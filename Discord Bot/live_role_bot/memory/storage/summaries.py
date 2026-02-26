from __future__ import annotations

from typing import Dict, Optional

import aiosqlite


class MemorySummariesMixin:
    async def get_dialogue_summary(self, guild_id: str, user_id: str, channel_id: str) -> Optional[Dict[str, object]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT summary_text, source_user_messages, last_message_id, updated_at
                FROM dialogue_summaries
                WHERE guild_id = ? AND user_id = ? AND channel_id = ?
                """,
                (guild_id, user_id, channel_id),
            ) as cursor:
                row = await cursor.fetchone()

        if row is None:
            return None

        return {
            "summary_text": str(row["summary_text"]),
            "source_user_messages": int(row["source_user_messages"]),
            "last_message_id": int(row["last_message_id"] or 0),
            "updated_at": str(row["updated_at"]),
        }

    async def get_latest_dialogue_summary_by_user_id(
        self,
        user_id: str,
        *,
        exclude_guild_id: str | None = None,
    ) -> Optional[Dict[str, object]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if exclude_guild_id:
                query = """
                    SELECT guild_id, channel_id, summary_text, source_user_messages, last_message_id, updated_at
                    FROM dialogue_summaries
                    WHERE user_id = ? AND guild_id <> ?
                    ORDER BY updated_at DESC, source_user_messages DESC, last_message_id DESC
                    LIMIT 1
                """
                params = (user_id, exclude_guild_id)
            else:
                query = """
                    SELECT guild_id, channel_id, summary_text, source_user_messages, last_message_id, updated_at
                    FROM dialogue_summaries
                    WHERE user_id = ?
                    ORDER BY updated_at DESC, source_user_messages DESC, last_message_id DESC
                    LIMIT 1
                """
                params = (user_id,)
            async with db.execute(query, params) as cursor:
                row = await cursor.fetchone()

        if row is None:
            return None
        return {
            "guild_id": str(row["guild_id"]),
            "channel_id": str(row["channel_id"]),
            "summary_text": str(row["summary_text"]),
            "source_user_messages": int(row["source_user_messages"]),
            "last_message_id": int(row["last_message_id"] or 0),
            "updated_at": str(row["updated_at"]),
        }

    async def upsert_dialogue_summary(
        self,
        guild_id: str,
        user_id: str,
        channel_id: str,
        summary_text: str,
        source_user_messages: int,
        last_message_id: int,
    ) -> None:
        cleaned = summary_text.strip()
        if not cleaned:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO dialogue_summaries (
                    guild_id, user_id, channel_id, summary_text, source_user_messages, last_message_id, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(guild_id, user_id, channel_id) DO UPDATE SET
                    summary_text = excluded.summary_text,
                    source_user_messages = excluded.source_user_messages,
                    last_message_id = excluded.last_message_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    guild_id,
                    user_id,
                    channel_id,
                    cleaned,
                    max(0, int(source_user_messages)),
                    max(0, int(last_message_id)),
                ),
            )
            await db.commit()
