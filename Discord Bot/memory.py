from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def init(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    user_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS guild_prompts (
                    guild_id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS guild_voice_settings (
                    guild_id TEXT PRIMARY KEY,
                    auto_speak INTEGER NOT NULL,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_channel_created
                ON conversation_messages(channel_id, created_at DESC)
                """
            )
            await db.commit()

    async def save_message(
        self,
        guild_id: str,
        channel_id: str,
        user_id: str,
        role: str,
        content: str,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO conversation_messages (guild_id, channel_id, user_id, role, content)
                VALUES (?, ?, ?, ?, ?)
                """,
                (guild_id, channel_id, user_id, role, content),
            )
            await db.commit()

    async def get_recent_messages(self, channel_id: str, limit: int) -> List[Dict[str, str]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT role, content
                FROM conversation_messages
                WHERE channel_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (channel_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()

        ordered = list(reversed(rows))
        return [{"role": row["role"], "content": row["content"]} for row in ordered]

    async def clear_channel_history(self, channel_id: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM conversation_messages WHERE channel_id = ?",
                (channel_id,),
            )
            await db.commit()

    async def set_guild_prompt(self, guild_id: str, prompt: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO guild_prompts (guild_id, prompt, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(guild_id) DO UPDATE SET
                    prompt = excluded.prompt,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (guild_id, prompt),
            )
            await db.commit()

    async def get_guild_prompt(self, guild_id: str) -> Optional[str]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT prompt FROM guild_prompts WHERE guild_id = ?",
                (guild_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return row[0] if row else None

    async def reset_guild_prompt(self, guild_id: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM guild_prompts WHERE guild_id = ?", (guild_id,))
            await db.commit()

    async def set_voice_auto_speak(self, guild_id: str, enabled: bool) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO guild_voice_settings (guild_id, auto_speak, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(guild_id) DO UPDATE SET
                    auto_speak = excluded.auto_speak,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (guild_id, 1 if enabled else 0),
            )
            await db.commit()

    async def get_voice_auto_speak(self, guild_id: str) -> Optional[bool]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT auto_speak FROM guild_voice_settings WHERE guild_id = ?",
                (guild_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return bool(row[0])
