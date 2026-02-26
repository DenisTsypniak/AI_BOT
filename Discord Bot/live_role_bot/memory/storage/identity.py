from __future__ import annotations

from typing import Dict, Optional

import aiosqlite

from .utils import _sqlite_memory_connection


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
        primary_display_name = (discord_global_name or discord_username or "unknown").strip() or "unknown"
        async with _sqlite_memory_connection(self.db_path) as db:
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
            await db.execute(
                """
                INSERT INTO global_user_profiles (
                    user_id, discord_username, discord_global_name, primary_display_name, first_seen, updated_at
                )
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    discord_username = excluded.discord_username,
                    discord_global_name = COALESCE(excluded.discord_global_name, global_user_profiles.discord_global_name),
                    primary_display_name = COALESCE(excluded.discord_global_name, excluded.discord_username, global_user_profiles.primary_display_name),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    user_id,
                    discord_username,
                    discord_global_name,
                    primary_display_name,
                ),
            )
            await db.commit()

    async def get_user_identity(self, guild_id: str, user_id: str) -> Optional[Dict[str, str | None]]:
        async with _sqlite_memory_connection(self.db_path) as db:
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

    async def get_latest_user_identity_by_user_id(
        self,
        user_id: str,
        *,
        exclude_guild_id: str | None = None,
    ) -> Optional[Dict[str, str | None]]:
        async with _sqlite_memory_connection(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT user_id, discord_username, discord_global_name, primary_display_name
                FROM global_user_profiles
                WHERE user_id = ?
                """,
                (user_id,),
            ) as cursor:
                global_row = await cursor.fetchone()
            if global_row is not None:
                primary_display = str(global_row["primary_display_name"] or "").strip()
                username = str(global_row["discord_username"] or "").strip()
                return {
                    "guild_id": "*",
                    "user_id": str(global_row["user_id"]),
                    "discord_username": username,
                    "discord_global_name": global_row["discord_global_name"],
                    "guild_nick": None,
                    "combined_label": primary_display or username,
                }
            if exclude_guild_id:
                query = """
                    SELECT guild_id, user_id, discord_username, discord_global_name, guild_nick, combined_label
                    FROM users
                    WHERE user_id = ? AND guild_id <> ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                """
                params = (user_id, exclude_guild_id)
            else:
                query = """
                    SELECT guild_id, user_id, discord_username, discord_global_name, guild_nick, combined_label
                    FROM users
                    WHERE user_id = ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                """
                params = (user_id,)
            async with db.execute(query, params) as cursor:
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
