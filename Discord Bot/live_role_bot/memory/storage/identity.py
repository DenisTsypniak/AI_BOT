from __future__ import annotations

from typing import Dict, Optional

import aiosqlite


class MemoryIdentityMixin:
    async def upsert_user_identity(
        self,
        guild_id: str,
        user_id: str,
        discord_username: str,
        discord_global_name: str | None,
        guild_nick: str | None,
        combined_label: str,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO users (
                    guild_id, user_id, discord_username, discord_global_name, guild_nick, combined_label, first_seen, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(guild_id, user_id) DO UPDATE SET
                    discord_username = excluded.discord_username,
                    discord_global_name = excluded.discord_global_name,
                    guild_nick = excluded.guild_nick,
                    combined_label = excluded.combined_label,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    guild_id,
                    user_id,
                    discord_username,
                    discord_global_name,
                    guild_nick,
                    combined_label,
                ),
            )
            await db.commit()

    async def get_user_identity(self, guild_id: str, user_id: str) -> Optional[Dict[str, str | None]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT guild_id, user_id, discord_username, discord_global_name, guild_nick, combined_label
                FROM users
                WHERE guild_id = ? AND user_id = ?
                """,
                (guild_id, user_id),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "guild_id": str(row["guild_id"]),
            "user_id": str(row["user_id"]),
            "discord_username": str(row["discord_username"]),
            "discord_global_name": row["discord_global_name"],
            "guild_nick": row["guild_nick"],
            "combined_label": str(row["combined_label"]),
        }
