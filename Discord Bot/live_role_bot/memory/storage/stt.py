from __future__ import annotations

import aiosqlite

from .utils import _clamp, _sqlite_memory_connection


class MemorySttMixin:
    async def save_stt_turn(
        self,
        guild_id: str,
        channel_id: str,
        user_id: str,
        duration_ms: int,
        rms: int,
        transcript: str | None,
        confidence: float,
        model_name: str,
        status: str,
        message_id: int | None = None,
    ) -> None:
        async with _sqlite_memory_connection(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO stt_turns (
                    message_id, guild_id, channel_id, user_id, duration_ms, rms,
                    transcript, confidence, model_name, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    guild_id,
                    channel_id,
                    user_id,
                    int(duration_ms),
                    int(rms),
                    transcript,
                    _clamp(float(confidence), 0.0, 1.0),
                    model_name,
                    status,
                ),
            )
            await db.commit()
