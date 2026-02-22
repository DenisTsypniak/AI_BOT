from __future__ import annotations

import asyncio
import contextlib
import logging
import time

try:
    import audioop
except ModuleNotFoundError:
    import audioop_lts as audioop  # type: ignore[import-not-found]

from .audio import StreamingPCMAudioSource, _parse_pcm_mime
from .state import _NativeSessionState

logger = logging.getLogger("gemini_native_audio")


class _NativeAudioPlaybackMixin:
    async def _playback_loop(self, state: _NativeSessionState) -> None:
        while not state.stop_event.is_set():
            try:
                item = await asyncio.wait_for(state.playback_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                # Decouple "source exists" from "bot is currently speaking".
                if state.playback_active and state.last_playback_feed_at > 0:
                    if time.monotonic() - state.last_playback_feed_at >= 0.45:
                        state.playback_active = False
                continue
            if item.turn_complete:
                await self._finish_turn_playback(state)
                continue
            await self._stream_model_audio(state, item.audio_bytes, item.mime_type)

    def _convert_model_audio_to_pcm48_stereo(self, audio_bytes: bytes, mime_type: str | None) -> bytes | None:
        if not audio_bytes:
            return None
        if mime_type and not mime_type.startswith("audio/pcm"):
            return None

        sample_rate, channels = _parse_pcm_mime(mime_type)
        try:
            if channels == 1:
                mono = audio_bytes
                if sample_rate != 48000:
                    mono, _ = audioop.ratecv(mono, 2, 1, sample_rate, 48000, None)
                return audioop.tostereo(mono, 2, 1, 1)
            if channels == 2:
                stereo = audio_bytes
                if sample_rate != 48000:
                    stereo, _ = audioop.ratecv(stereo, 2, 2, sample_rate, 48000, None)
                return stereo
        except Exception as exc:
            logger.debug("Audio convert failed: %s", exc)
        return None

    async def _ensure_active_source(self, state: _NativeSessionState) -> StreamingPCMAudioSource | None:
        guild = self.client.get_guild(state.guild_id)
        if guild is None or guild.voice_client is None:
            return None

        voice_client = guild.voice_client
        if voice_client.channel is None or voice_client.channel.id != state.voice_channel_id:
            return None

        async with state.playback_lock:
            if state.active_source is not None:
                if voice_client.is_playing() or voice_client.is_paused():
                    return state.active_source
                # Source object exists but playback has ended unexpectedly.
                state.active_source = None
                state.active_source_done = None
                state.playback_active = False
                state.last_playback_feed_at = 0.0
            if voice_client.is_playing() or voice_client.is_paused():
                return None

            loop = asyncio.get_running_loop()
            done = asyncio.Event()
            source = StreamingPCMAudioSource()

            def after_play(error: Exception | None) -> None:
                if error:
                    logger.error("Native audio playback error: %s", error)
                loop.call_soon_threadsafe(done.set)

            try:
                voice_client.play(source, after=after_play)
            except Exception as exc:
                logger.error("Failed to start native playback: %s", exc)
                return None

            state.active_source = source
            state.active_source_done = done
            state.playback_active = False
            state.last_playback_feed_at = 0.0
            return source

    async def _stream_model_audio(
        self,
        state: _NativeSessionState,
        audio_bytes: bytes,
        mime_type: str | None,
    ) -> None:
        pcm = self._convert_model_audio_to_pcm48_stereo(audio_bytes, mime_type)
        if not pcm:
            return
        for _ in range(10):
            source = await self._ensure_active_source(state)
            if source is not None:
                source.feed(pcm)
                state.playback_active = True
                state.last_playback_feed_at = time.monotonic()
                return
            await asyncio.sleep(0.02)
        logger.debug("Dropped model audio chunk for guild=%s (source unavailable)", state.guild_id)

    async def _finish_turn_playback(self, state: _NativeSessionState) -> None:
        # Keep source warm across turns to prevent audible clipping at boundaries.
        await asyncio.sleep(0.08)
        state.playback_active = False

    async def _interrupt_playback(self, guild_id: int) -> None:
        state = self._states.get(guild_id)
        if state is not None:
            while not state.playback_queue.empty():
                with contextlib.suppress(asyncio.QueueEmpty):
                    state.playback_queue.get_nowait()
            async with state.playback_lock:
                source = state.active_source
                done = state.active_source_done
                state.active_source = None
                state.active_source_done = None
                state.playback_active = False
                state.last_playback_feed_at = 0.0
                if source is not None:
                    source.force_stop()
                if done is not None:
                    done.set()

        guild = self.client.get_guild(guild_id)
        if guild is None or guild.voice_client is None:
            return
        vc = guild.voice_client
        if vc.is_playing() or vc.is_paused():
            vc.stop()
