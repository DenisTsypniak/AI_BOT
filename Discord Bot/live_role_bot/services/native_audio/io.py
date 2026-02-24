from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque

from google import genai
from google.genai import types

try:
    import audioop
except ModuleNotFoundError:
    import audioop_lts as audioop  # type: ignore[import-not-found]

from .audio import _convert_discord_pcm_to_live
from .state import _NativeSessionState, _PlaybackItem

logger = logging.getLogger("gemini_native_audio")


class _NativeAudioIOMixin:
    async def _sender_loop(self, state: _NativeSessionState, session: "genai.live.AsyncSession") -> None:
        speaking = False
        silence_ms = 0
        timeout_ms = 200
        start_voiced_ms = 0
        start_min_ms = 100
        preroll_pcm_48k: deque[bytes] = deque(maxlen=12)
        base_start_rms = max(46, int(self.settings.voice_silence_rms * 0.72))
        end_silence_rms_floor = max(12, self.manual_vad_silence_rms_threshold)
        adaptive_noise_floor = float(max(10, end_silence_rms_floor))

        async def _send_pcm_frame(pcm_48k: bytes) -> bool:
            pcm_live, state.rate_state = _convert_discord_pcm_to_live(
                pcm_48k_stereo=pcm_48k,
                target_rate=self.input_rate,
                rate_state=state.rate_state,
            )
            if not pcm_live:
                return False
            await session.send_realtime_input(
                audio=types.Blob(
                    data=pcm_live,
                    mime_type=f"audio/pcm;rate={self.input_rate}",
                )
            )
            return True

        while not state.stop_event.is_set():
            try:
                _, _, pcm_48k = await asyncio.wait_for(state.input_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                if speaking:
                    silence_ms += timeout_ms
                    if silence_ms >= self.vad_silence_ms:
                        await session.send_realtime_input(activity_end=types.ActivityEnd())
                        speaking = False
                        silence_ms = 0
                continue

            frame_ms = max(1, int((len(pcm_48k) / (48000 * 4)) * 1000))
            try:
                rms = int(audioop.rms(pcm_48k, 2))
            except Exception:
                rms = 0

            if not speaking:
                # Track ambient floor from quiet packets to avoid false starts on tiny noises.
                adaptive_noise_floor = (adaptive_noise_floor * 0.96) + (min(rms, base_start_rms) * 0.04)
                dynamic_start_rms = max(
                    base_start_rms,
                    int(adaptive_noise_floor * 2.1),
                    end_silence_rms_floor + 18,
                )
                # When bot is currently speaking, require stronger interruption to avoid self-echo loops.
                if state.playback_active:
                    dynamic_start_rms = max(dynamic_start_rms, int(base_start_rms * 1.35))

                preroll_pcm_48k.append(pcm_48k)
                if rms < dynamic_start_rms:
                    start_voiced_ms = 0
                    continue

                start_voiced_ms += frame_ms
                if start_voiced_ms < start_min_ms:
                    continue

                if state.playback_active:
                    logger.info("native-vad barge-in detected guild=%s; interrupting playback", state.guild_id)
                    await self._interrupt_playback(state.guild_id)

                await session.send_realtime_input(activity_start=types.ActivityStart())
                logger.info(
                    "native-vad activity_start guild=%s rms=%s threshold=%s",
                    state.guild_id,
                    rms,
                    dynamic_start_rms,
                )
                speaking = True
                silence_ms = 0
                start_voiced_ms = 0

                # Send a short preroll so first syllables are not clipped.
                while preroll_pcm_48k:
                    frame = preroll_pcm_48k.popleft()
                    await _send_pcm_frame(frame)
                continue

            dynamic_end_rms = max(end_silence_rms_floor, int(adaptive_noise_floor * 1.35))
            if rms <= dynamic_end_rms:
                silence_ms += frame_ms
                if silence_ms >= self.vad_silence_ms:
                    await session.send_realtime_input(activity_end=types.ActivityEnd())
                    logger.info(
                        "native-vad activity_end guild=%s silence_ms=%s threshold=%s",
                        state.guild_id,
                        silence_ms,
                        dynamic_end_rms,
                    )
                    speaking = False
                    silence_ms = 0
                    start_voiced_ms = 0
                    preroll_pcm_48k.clear()
                    continue
            else:
                silence_ms = 0

            await _send_pcm_frame(pcm_48k)

        if speaking:
            with contextlib.suppress(Exception):
                await session.send_realtime_input(activity_end=types.ActivityEnd())
                logger.info("native-vad activity_end forced on sender exit guild=%s", state.guild_id)

    async def _receiver_loop(self, state: _NativeSessionState, session: "genai.live.AsyncSession") -> None:
        while not state.stop_event.is_set():
            pending_audio = bytearray()
            pending_mime: str | None = None
            input_text: str | None = None
            output_text: str | None = None

            async for message in session.receive():
                server_content = message.server_content
                if server_content is None:
                    continue

                if server_content.input_transcription and server_content.input_transcription.text:
                    candidate_input = server_content.input_transcription.text.strip()
                    if candidate_input:
                        if not input_text:
                            input_text = candidate_input
                        elif len(candidate_input) >= len(input_text):
                            input_text = candidate_input
                    if server_content.input_transcription.finished and input_text:
                        await self._emit_user_transcript(state, input_text)

                if server_content.output_transcription and server_content.output_transcription.text:
                    candidate_output = server_content.output_transcription.text.strip()
                    if candidate_output:
                        if not output_text:
                            output_text = candidate_output
                        elif len(candidate_output) >= len(output_text):
                            output_text = candidate_output

                if server_content.model_turn and server_content.model_turn.parts:
                    for part in server_content.model_turn.parts:
                        inline = part.inline_data
                        if inline and isinstance(inline.data, bytes) and inline.data:
                            pending_audio.extend(inline.data)
                            if inline.mime_type:
                                pending_mime = inline.mime_type
                            if len(pending_audio) >= 1600:
                                await self._enqueue_playback_chunk(state, bytes(pending_audio), pending_mime, False)
                                pending_audio.clear()

                if server_content.turn_complete:
                    if pending_audio:
                        await self._enqueue_playback_chunk(state, bytes(pending_audio), pending_mime, False)
                        pending_audio.clear()
                    await self._enqueue_playback_chunk(state, b"", pending_mime, True)

                    if output_text:
                        await self._emit_assistant_transcript(state, output_text)

                    if state.send_transcripts_to_text and (input_text or output_text):
                        lines: list[str] = []
                        if input_text:
                            lines.append(f"[voice] {state.last_speaker_name}: {input_text}")
                        if output_text:
                            lines.append(f"AI: {output_text}")
                        if lines:
                            await self._send_debug(state, "\n".join(lines))

                    input_text = None
                    output_text = None

            if not state.stop_event.is_set():
                await asyncio.sleep(0.05)

    async def _enqueue_playback_chunk(
        self,
        state: _NativeSessionState,
        audio_bytes: bytes,
        mime_type: str | None,
        turn_complete: bool,
    ) -> None:
        if not audio_bytes and not turn_complete:
            return
        await state.playback_queue.put(
            _PlaybackItem(audio_bytes=audio_bytes, mime_type=mime_type, turn_complete=turn_complete)
        )
