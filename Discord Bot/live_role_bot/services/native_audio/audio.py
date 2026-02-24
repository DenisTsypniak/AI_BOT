from __future__ import annotations

import re
import threading
from typing import Any

import discord

try:
    import audioop
except ModuleNotFoundError:
    import audioop_lts as audioop  # type: ignore[import-not-found]


class StreamingPCMAudioSource(discord.AudioSource):
    FRAME_SIZE = 3840  # 20ms of 48kHz stereo 16-bit PCM.

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._cond = threading.Condition()
        self._closed = False

    def feed(self, pcm_48k_stereo: bytes) -> None:
        if not pcm_48k_stereo:
            return
        with self._cond:
            if self._closed:
                return
            self._buffer.extend(pcm_48k_stereo)
            self._cond.notify_all()

    def end_after_drain(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def force_stop(self) -> None:
        with self._cond:
            self._closed = True
            self._buffer.clear()
            self._cond.notify_all()

    def read(self) -> bytes:
        with self._cond:
            while True:
                if len(self._buffer) >= self.FRAME_SIZE:
                    out = bytes(self._buffer[: self.FRAME_SIZE])
                    del self._buffer[: self.FRAME_SIZE]
                    return out

                if self._closed:
                    if self._buffer:
                        out = bytes(self._buffer)
                        self._buffer.clear()
                        if len(out) < self.FRAME_SIZE:
                            out += b"\x00" * (self.FRAME_SIZE - len(out))
                        return out[: self.FRAME_SIZE]
                    return b""

                self._cond.wait(timeout=0.05)


def _normalize_model(model: str) -> str:
    cleaned = model.strip()
    if not cleaned:
        return "models/gemini-2.5-flash-native-audio-latest"
    if cleaned.startswith("models/"):
        return cleaned
    return f"models/{cleaned}"


def _extract_int_param(mime_type: str | None, key: str, default: int) -> int:
    if not mime_type:
        return default
    match = re.search(rf"{re.escape(key)}=(\d+)", mime_type)
    if not match:
        return default
    try:
        return int(match.group(1))
    except ValueError:
        return default


def _parse_pcm_mime(mime_type: str | None) -> tuple[int, int]:
    rate = _extract_int_param(mime_type, "rate", 24000)
    channels = max(1, _extract_int_param(mime_type, "channels", 1))
    return rate, channels


def _convert_discord_pcm_to_live(
    pcm_48k_stereo: bytes,
    target_rate: int,
    rate_state: Any,
) -> tuple[bytes, Any]:
    if not pcm_48k_stereo:
        return b"", rate_state
    mono = audioop.tomono(pcm_48k_stereo, 2, 0.5, 0.5)
    converted, new_state = audioop.ratecv(mono, 2, 1, 48000, target_rate, rate_state)
    return converted, new_state
