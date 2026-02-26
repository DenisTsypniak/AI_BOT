from __future__ import annotations

from typing import Dict, Optional

import aiosqlite

from .utils import _sqlite_memory_connection


class MemorySummariesMixin:
    async def get_dialogue_summary(self, guild_id: str, user_id: str, channel_id: str) -> Optional[Dict[str, object]]:
        async with _sqlite_memory_connection(self.db_path) as db:
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
        async with _sqlite_memory_connection(self.db_path) as db:
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

    async def get_global_biography_summary(
        self,
        subject_kind: str,
        subject_id: str,
    ) -> Optional[Dict[str, object]]:
        kind = str(subject_kind or "").strip().lower()
        subject = str(subject_id or "").strip()
        if not (kind and subject):
            return None

        async with _sqlite_memory_connection(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT subject_kind, subject_id, summary_text, source_fact_count, source_summary_count,
                       source_updated_at, last_source_guild_id, created_at, updated_at
                FROM global_biography_summaries
                WHERE subject_kind = ? AND subject_id = ?
                """,
                (kind, subject),
            ) as cursor:
                row = await cursor.fetchone()

        if row is None:
            return None
        return {
            "subject_kind": str(row["subject_kind"]),
            "subject_id": str(row["subject_id"]),
            "summary_text": str(row["summary_text"]),
            "source_fact_count": int(row["source_fact_count"] or 0),
            "source_summary_count": int(row["source_summary_count"] or 0),
            "source_updated_at": str(row["source_updated_at"] or ""),
            "last_source_guild_id": str(row["last_source_guild_id"] or ""),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    async def get_global_user_biography_summary(self, user_id: str) -> Optional[Dict[str, object]]:
        return await self.get_global_biography_summary("user", user_id)

    async def get_persona_biography_summary(self, persona_id: str) -> Optional[Dict[str, object]]:
        return await self.get_global_biography_summary("persona", persona_id)

    async def upsert_global_biography_summary(
        self,
        *,
        subject_kind: str,
        subject_id: str,
        summary_text: str,
        source_fact_count: int = 0,
        source_summary_count: int = 0,
        source_updated_at: str | None = None,
        last_source_guild_id: str | None = None,
    ) -> None:
        kind = str(subject_kind or "").strip().lower()
        subject = str(subject_id or "").strip()
        summary = summary_text.strip()
        if not (kind and subject and summary):
            return

        async with _sqlite_memory_connection(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO global_biography_summaries (
                    subject_kind, subject_id, summary_text, source_fact_count, source_summary_count,
                    source_updated_at, last_source_guild_id, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(subject_kind, subject_id) DO UPDATE SET
                    summary_text = excluded.summary_text,
                    source_fact_count = excluded.source_fact_count,
                    source_summary_count = excluded.source_summary_count,
                    source_updated_at = excluded.source_updated_at,
                    last_source_guild_id = excluded.last_source_guild_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    kind,
                    subject,
                    summary,
                    max(0, int(source_fact_count)),
                    max(0, int(source_summary_count)),
                    str(source_updated_at).strip() if source_updated_at else None,
                    str(last_source_guild_id).strip() if last_source_guild_id else None,
                ),
            )
            await db.commit()

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

        async with _sqlite_memory_connection(self.db_path) as db:
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
