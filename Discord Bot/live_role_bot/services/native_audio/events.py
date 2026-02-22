from __future__ import annotations

import asyncio
import contextlib
import logging
import re

from .state import _NativeSessionState

logger = logging.getLogger("gemini_native_audio")


class _NativeAudioEventsMixin:
    async def _send_debug(self, state: _NativeSessionState, text: str) -> None:
        channel = self.client.get_channel(state.text_channel_id)
        if channel is None or not hasattr(channel, "send"):
            return
        with contextlib.suppress(Exception):
            await channel.send(text)

    async def _emit_user_transcript(self, state: _NativeSessionState, text: str) -> None:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return
        if state.last_speaker_id is None:
            return
        if state.last_speaker_id == state.bot_user_id:
            return
        callback = getattr(self.client, "on_native_audio_user_transcript", None)
        if callback is None:
            return
        try:
            result = callback(state.guild_id, state.last_speaker_id, cleaned, "gemini_native_input", 0.96)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.debug("Native user transcript callback failed: %s", exc)

    async def _emit_assistant_transcript(self, state: _NativeSessionState, text: str) -> None:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return
        callback = getattr(self.client, "on_native_audio_assistant_transcript", None)
        if callback is None:
            return
        try:
            result = callback(state.guild_id, cleaned, "gemini_native_output")
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.debug("Native assistant transcript callback failed: %s", exc)
