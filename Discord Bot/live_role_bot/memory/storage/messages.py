from __future__ import annotations

from typing import Dict, List

import aiosqlite

from .utils import _clamp, _sqlite_memory_connection


class MemoryMessagesMixin:
    async def get_or_create_session(
        self,
        guild_id: str,
        channel_id: str,
        mode: str,
        role_id: str,
    ) -> int:
        async with _sqlite_memory_connection(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT session_id
                FROM sessions
                WHERE guild_id = ? AND channel_id = ? AND mode = ? AND status = 'active'
                ORDER BY session_id DESC
                LIMIT 1
                """,
                (guild_id, channel_id, mode),
            ) as cursor:
                row = await cursor.fetchone()

            if row is not None:
                return int(row["session_id"])

            cursor = await db.execute(
                """
                INSERT INTO sessions (guild_id, channel_id, mode, role_id, status, started_at)
                VALUES (?, ?, ?, ?, 'active', CURRENT_TIMESTAMP)
                """,
                (guild_id, channel_id, mode, role_id),
            )
            await db.commit()
            return int(cursor.lastrowid)

    async def save_message(
        self,
        session_id: int,
        guild_id: str,
        channel_id: str,
        user_id: str,
        author_label: str,
        role: str,
        modality: str,
        content_raw: str,
        content_clean: str,
        source: str,
        quality: float = 1.0,
    ) -> int:
        async with _sqlite_memory_connection(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO messages (
                    session_id, guild_id, channel_id, user_id, author_label,
                    role, modality, content_raw, content_clean, source, quality
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(session_id),
                    guild_id,
                    channel_id,
                    user_id,
                    author_label,
                    role,
                    modality,
                    content_raw,
                    content_clean,
                    source,
                    _clamp(float(quality), 0.0, 1.0),
                ),
            )
            await db.commit()
            return int(cursor.lastrowid)

    async def get_recent_session_messages(self, session_id: int, limit: int) -> List[Dict[str, object]]:
        async with _sqlite_memory_connection(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT message_id, user_id, author_label, role, modality, content_clean, source, quality, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY message_id DESC
                LIMIT ?
                """,
                (int(session_id), max(1, int(limit))),
            ) as cursor:
                rows = await cursor.fetchall()

        ordered = list(reversed(rows))
        return [
            {
                "message_id": int(row["message_id"]),
                "user_id": str(row["user_id"]),
                "author_label": str(row["author_label"]),
                "role": str(row["role"]),
                "modality": str(row["modality"]),
                "content": str(row["content_clean"]),
                "source": str(row["source"]),
                "quality": float(row["quality"]),
                "created_at": str(row["created_at"]),
            }
            for row in ordered
        ]

    async def count_user_messages_in_channel(self, guild_id: str, channel_id: str, user_id: str) -> int:
        async with _sqlite_memory_connection(self.db_path) as db:
            async with db.execute(
                """
                SELECT COUNT(*)
                FROM messages
                WHERE guild_id = ? AND channel_id = ? AND user_id = ? AND role = 'user'
                """,
                (guild_id, channel_id, user_id),
            ) as cursor:
                row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def get_recent_dialogue_messages(
        self,
        guild_id: str,
        channel_id: str,
        user_id: str,
        limit: int,
    ) -> List[Dict[str, object]]:
        async with _sqlite_memory_connection(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT message_id, role, user_id, author_label, content_clean
                FROM messages
                WHERE guild_id = ?
                  AND channel_id = ?
                  AND (
                    role = 'assistant'
                    OR (role = 'user' AND user_id = ?)
                  )
                ORDER BY message_id DESC
                LIMIT ?
                """,
                (guild_id, channel_id, user_id, max(1, int(limit))),
            ) as cursor:
                rows = await cursor.fetchall()

        ordered = list(reversed(rows))
        return [
            {
                "message_id": int(row["message_id"]),
                "role": str(row["role"]),
                "user_id": str(row["user_id"]),
                "author_label": str(row["author_label"]),
                "content": str(row["content_clean"]),
            }
            for row in ordered
        ]
