from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .audio import StreamingPCMAudioSource


@dataclass(slots=True)
class _PlaybackItem:
    audio_bytes: bytes
    mime_type: str | None
    turn_complete: bool = False


@dataclass(slots=True)
class _NativeSessionState:
    guild_id: int
    voice_channel_id: int
    text_channel_id: int
    bot_user_id: int
    send_transcripts_to_text: bool
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    input_queue: asyncio.Queue[tuple[int, str, bytes]] = field(
        default_factory=lambda: asyncio.Queue(maxsize=500)
    )
    playback_queue: asyncio.Queue[_PlaybackItem] = field(default_factory=asyncio.Queue)
    run_task: asyncio.Task[None] | None = None
    ready_error: str | None = None
    rate_state: Any = None
    last_speaker_id: int | None = None
    last_speaker_name: str = "User"
    playback_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_source: StreamingPCMAudioSource | None = None
    active_source_done: asyncio.Event | None = None
    playback_active: bool = False
    last_playback_feed_at: float = 0.0
