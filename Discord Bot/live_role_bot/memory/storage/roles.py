from __future__ import annotations

from typing import Dict, Optional

import aiosqlite

from .utils import _sqlite_memory_connection


class MemoryRolesMixin:
    async def ensure_role_profile(
        self,
        role_id: str,
        name: str,
        goal: str,
        style: str,
        constraints: str,
    ) -> None:
        async with _sqlite_memory_connection(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO role_profiles (role_id, name, goal, style, constraints, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(role_id) DO UPDATE SET
                    name = excluded.name,
                    goal = excluded.goal,
                    style = excluded.style,
                    constraints = excluded.constraints,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (role_id, name, goal, style, constraints),
            )
            await db.commit()

    async def ensure_guild_settings(self, guild_id: str, default_role_id: str) -> None:
        async with _sqlite_memory_connection(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO guild_settings (guild_id, default_role_id, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(guild_id) DO UPDATE SET
                    default_role_id = excluded.default_role_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (guild_id, default_role_id),
            )
            await db.commit()

    async def get_guild_role_id(self, guild_id: str) -> str | None:
        async with _sqlite_memory_connection(self.db_path) as db:
            async with db.execute(
                "SELECT default_role_id FROM guild_settings WHERE guild_id = ?",
                (guild_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return str(row[0])

    async def get_role_profile(self, role_id: str) -> Optional[Dict[str, str]]:
        async with _sqlite_memory_connection(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT role_id, name, goal, style, constraints
                FROM role_profiles
                WHERE role_id = ?
                """,
                (role_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "role_id": str(row["role_id"]),
            "name": str(row["name"]),
            "goal": str(row["goal"]),
            "style": str(row["style"]),
            "constraints": str(row["constraints"]),
        }
