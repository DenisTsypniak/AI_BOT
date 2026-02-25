from __future__ import annotations

import asyncio
import contextlib

import discord
from google import genai

from ...config import Settings
from .audio import _normalize_model
from .state import _NativeSessionState


class _NativeAudioBase:
    def __init__(self, discord_client: discord.Client, settings: Settings) -> None:
        self.client = discord_client
        self.settings = settings
        configured_model = _normalize_model(settings.gemini_live_model)
        self.models = self._dedupe_models(
            [
                configured_model,
                "models/gemini-2.5-flash-native-audio-latest",
                "models/gemini-2.5-flash-native-audio-preview-12-2025",
                "models/gemini-2.5-flash-native-audio-preview-09-2025",
            ]
        )

        self.voice_name = settings.gemini_live_voice
        self.input_rate = max(8000, settings.gemini_live_input_sample_rate)
        self.temperature = settings.gemini_live_temperature
        self.max_output_tokens = (
            int(settings.gemini_live_max_output_tokens)
            if settings.gemini_live_max_output_tokens > 0
            else None
        )
        self.vad_silence_ms = max(120, settings.gemini_live_vad_silence_ms)
        self.manual_vad_silence_rms_threshold = max(12, int(settings.voice_silence_rms * 0.35))

        self._api_client = genai.Client(api_key=settings.gemini_api_key)
        self._states: dict[int, _NativeSessionState] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._resolved_model_order: list[str] | None = None

    @staticmethod
    def _dedupe_models(models: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for model in models:
            normalized = _normalize_model(model)
            if normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)
        return out

    def _list_available_native_models_sync(self) -> list[str]:
        found: list[str] = []
        with contextlib.suppress(Exception):
            for model in self._api_client.models.list():
                name = getattr(model, "name", None)
                if not isinstance(name, str) or not name.strip():
                    continue
                normalized = _normalize_model(name)
                if "gemini-2.5-flash-native-audio" in normalized:
                    found.append(normalized)
        return self._dedupe_models(found)

    async def _resolve_connect_models(self) -> list[str]:
        if self._resolved_model_order is not None:
            return self._resolved_model_order

        preferred = list(self.models)
        discovered = await asyncio.to_thread(self._list_available_native_models_sync)
        if not discovered:
            self._resolved_model_order = preferred
            return preferred

        discovered_set = set(discovered)
        ordered = [model for model in preferred if model in discovered_set]
        for model in discovered:
            if model not in ordered:
                ordered.append(model)

        self._resolved_model_order = ordered or preferred
        return self._resolved_model_order

    def has_session(self, guild_id: int) -> bool:
        state = self._states.get(guild_id)
        return state is not None and not state.stop_event.is_set()

    def status_snapshot(self) -> dict[str, object]:
        sessions: list[dict[str, object]] = []
        for guild_id, state in self._states.items():
            sessions.append(
                {
                    "guild_id": guild_id,
                    "voice_channel_id": state.voice_channel_id,
                    "text_channel_id": state.text_channel_id,
                    "ready": bool(state.ready_event.is_set()),
                    "ready_error": state.ready_error or "",
                    "stop_requested": bool(state.stop_event.is_set()),
                    "input_queue": int(state.input_queue.qsize()),
                    "playback_queue": int(state.playback_queue.qsize()),
                    "playback_active": bool(state.playback_active),
                    "last_speaker_name": state.last_speaker_name or "",
                    "run_task_alive": bool(state.run_task is not None and not state.run_task.done()),
                }
            )
        return {
            "enabled": True,
            "model_candidates": list(self.models),
            "sessions": sessions,
            "session_count": len(sessions),
        }

    def push_pcm(self, guild_id: int, user_id: int, user_name: str, pcm_48k_stereo: bytes) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._enqueue_pcm, guild_id, user_id, user_name, pcm_48k_stereo)

    def _enqueue_pcm(self, guild_id: int, user_id: int, user_name: str, pcm_48k_stereo: bytes) -> None:
        state = self._states.get(guild_id)
        if state is None or state.stop_event.is_set():
            return
        if not pcm_48k_stereo:
            return

        state.last_speaker_id = user_id
        state.last_speaker_name = user_name or state.last_speaker_name

        if state.input_queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                state.input_queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            state.input_queue.put_nowait((user_id, user_name, pcm_48k_stereo))

    async def start_session(
        self,
        guild_id: int,
        voice_channel_id: int,
        text_channel_id: int,
        bot_user_id: int,
        system_prompt: str,
        preferred_language: str,
        send_transcripts_to_text: bool,
    ) -> None:
        await self.stop_session(guild_id)
        self._loop = asyncio.get_running_loop()

        state = _NativeSessionState(
            guild_id=guild_id,
            voice_channel_id=voice_channel_id,
            text_channel_id=text_channel_id,
            bot_user_id=bot_user_id,
            send_transcripts_to_text=send_transcripts_to_text,
        )
        self._states[guild_id] = state
        state.run_task = asyncio.create_task(
            self._run_session(state, system_prompt, preferred_language),
            name=f"native-audio-session-{guild_id}",
        )

        try:
            await asyncio.wait_for(state.ready_event.wait(), timeout=25)
        except asyncio.TimeoutError as exc:
            await self.stop_session(guild_id)
            raise RuntimeError("Gemini Native Audio session startup timed out") from exc

        if state.ready_error:
            await self.stop_session(guild_id)
            raise RuntimeError(state.ready_error)

    async def stop_session(self, guild_id: int) -> None:
        state = self._states.get(guild_id)
        if state is None:
            return
        state.stop_event.set()
        await self._interrupt_playback(guild_id)

        state = self._states.pop(guild_id, None)
        if state is None:
            return

        if state.run_task is not None:
            state.run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await state.run_task

    async def shutdown_all(self) -> None:
        for guild_id in list(self._states.keys()):
            await self.stop_session(guild_id)
