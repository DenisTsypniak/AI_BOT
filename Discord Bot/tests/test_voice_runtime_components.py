from __future__ import annotations

import asyncio
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("discord")

from live_role_bot.discord.common import PendingVoiceTurn  # noqa: E402
from live_role_bot.discord.mixins.voice_mixin import VoiceMixin  # noqa: E402


@dataclass
class _Settings:
    voice_silence_rms: int = 95
    voice_silence_ms: int = 520
    voice_min_turn_ms: int = 120
    voice_max_turn_seconds: int = 18


class _FakeVoiceBot(VoiceMixin):
    def __init__(self) -> None:
        self.settings = _Settings()
        self.voice_buffers = {}
        self.voice_text_channels: dict[int, int] = {}
        self.voice_turn_queue: asyncio.Queue[PendingVoiceTurn] = asyncio.Queue(maxsize=2)
        self.diag_events: list[dict[str, object]] = []

    def _record_voice_memory_diag(self, **kwargs: object) -> None:  # type: ignore[override]
        self.diag_events.append(dict(kwargs))


def _pcm_frames(sample: int, frames: int) -> bytes:
    chunk = struct.pack("<hh", sample, sample)
    return chunk * frames


def test_finalize_voice_turn_enqueues_item_with_binding() -> None:
    bot = _FakeVoiceBot()
    bot.voice_text_channels[1] = 777
    pcm = _pcm_frames(12000, 8000)  # ~166ms at 48k stereo

    bot._ingest_voice_pcm(1, 42, "User42", pcm, True, "local_stt")
    bot._finalize_voice_turn((1, 42))

    item = bot.voice_turn_queue.get_nowait()
    assert isinstance(item, PendingVoiceTurn)
    assert item.guild_id == 1
    assert item.channel_id == 777
    assert item.user_id == 42
    assert item.user_label == "User42"
    assert item.transcript_source == "local_stt"
    assert any(e.get("stage") == "voice_turn_queue" and e.get("outcome") == "queued" for e in bot.diag_events)


def test_finalize_voice_turn_drops_when_no_text_binding() -> None:
    bot = _FakeVoiceBot()
    pcm = _pcm_frames(10000, 8000)

    bot._ingest_voice_pcm(5, 9, "NoBind", pcm, True, "bridge_local_stt")
    bot._finalize_voice_turn((5, 9))

    assert bot.voice_turn_queue.empty()
    assert any(
        e.get("stage") == "voice_turn_finalize"
        and e.get("outcome") == "drop"
        and e.get("reason") == "no_voice_text_channel_binding"
        for e in bot.diag_events
    )


def test_finalize_voice_turn_queue_full_drops_oldest() -> None:
    bot = _FakeVoiceBot()
    bot.voice_turn_queue = asyncio.Queue(maxsize=1)
    bot.voice_text_channels[2] = 888

    old_item = PendingVoiceTurn(
        guild_id=2,
        channel_id=888,
        user_id=1,
        user_label="old",
        pcm_48k_stereo=b"old",
    )
    bot.voice_turn_queue.put_nowait(old_item)

    pcm = _pcm_frames(14000, 8000)
    bot._ingest_voice_pcm(2, 99, "NewUser", pcm, False, "bridge_local_stt")
    bot._finalize_voice_turn((2, 99))

    item = bot.voice_turn_queue.get_nowait()
    assert item.user_id == 99
    assert item.reply_enabled is False
    assert item.transcript_source == "bridge_local_stt"
    assert any(
        e.get("stage") == "voice_turn_queue"
        and e.get("outcome") == "drop"
        and e.get("reason") == "queue_full_drop_oldest"
        for e in bot.diag_events
    )


def test_ingest_voice_pcm_handles_malformed_pcm_without_crash() -> None:
    bot = _FakeVoiceBot()
    bot.voice_text_channels[3] = 999

    # Odd-length PCM raises audioop.error in rms(), but the bot should handle it safely.
    bot._ingest_voice_pcm(3, 55, "BadPCM", b"\x01", True, "local_stt")

    assert bot.voice_turn_queue.empty()
