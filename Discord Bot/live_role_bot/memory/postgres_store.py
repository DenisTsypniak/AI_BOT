from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

try:
    import asyncpg
except Exception:  # pragma: no cover - optional dependency at runtime
    asyncpg = None  # type: ignore[assignment]

from .storage.utils import _clamp


class PostgresMemoryStore:
    """Postgres-backed memory store implementing the same API as MemoryStore."""

    SCHEMA_VERSION = 7
    backend_name = "postgres"

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn.strip()
        if not self.dsn:
            raise ValueError("MEMORY_POSTGRES_DSN cannot be empty")
        self._pool: "asyncpg.Pool | None" = None
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def _ensure_pool(self) -> "asyncpg.Pool":
        if asyncpg is None:
            raise RuntimeError(
                "Postgres memory backend requires asyncpg. Install with: pip install asyncpg"
            )
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=self.dsn,
                min_size=1,
                max_size=6,
                command_timeout=30.0,
            )
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        self._initialized = False

    async def ping(self) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute("SELECT 1")

    async def init(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    version = await self._get_schema_version(conn)
                    if version != self.SCHEMA_VERSION:
                        await self._reset_schema(conn)
                        await self._set_schema_version(conn, self.SCHEMA_VERSION)
                    else:
                        await self._create_schema(conn)
            self._initialized = True

    async def _get_schema_version(self, conn: "asyncpg.Connection") -> int:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        row = await conn.fetchrow("SELECT value FROM memory_meta WHERE key = 'schema_version'")
        if row is None:
            return 0
        try:
            return int(str(row["value"]))
        except Exception:
            return 0

    async def _set_schema_version(self, conn: "asyncpg.Connection", version: int) -> None:
        await conn.execute(
            """
            INSERT INTO memory_meta (key, value, updated_at)
            VALUES ('schema_version', $1, NOW())
            ON CONFLICT(key) DO UPDATE SET
                value = EXCLUDED.value,
                updated_at = NOW()
            """,
            str(int(version)),
        )

    async def _reset_schema(self, conn: "asyncpg.Connection") -> None:
        tables = (
            "fact_evidence",
            "stt_turns",
            "messages",
            "sessions",
            "dialogue_summaries",
            "user_facts",
            "guild_settings",
            "role_profiles",
            "users",
            "memory_meta",
            # legacy names kept for cleanup parity with sqlite reset
            "conversation_messages",
            "guild_prompts",
            "guild_voice_settings",
            "user_profile_facts",
            "user_memory_items",
            "guild_user_identities",
            "voice_turn_logs",
            "chat_messages",
            "user_identities",
            "tone_observations",
            "character_profiles",
            "voice_turn_audit",
        )
        for table in tables:
            await conn.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
        await self._create_schema(conn)

    async def _create_schema(self, conn: "asyncpg.Connection") -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS users (
                guild_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                discord_username TEXT NOT NULL,
                discord_global_name TEXT,
                guild_nick TEXT,
                combined_label TEXT NOT NULL,
                first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (guild_id, user_id)
            );

            CREATE TABLE IF NOT EXISTS role_profiles (
                role_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                goal TEXT NOT NULL,
                style TEXT NOT NULL,
                constraints TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS guild_settings (
                guild_id TEXT PRIMARY KEY,
                default_role_id TEXT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                FOREIGN KEY(default_role_id) REFERENCES role_profiles(role_id)
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id BIGSERIAL PRIMARY KEY,
                guild_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                role_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                ended_at TIMESTAMPTZ,
                FOREIGN KEY(role_id) REFERENCES role_profiles(role_id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                message_id BIGSERIAL PRIMARY KEY,
                session_id BIGINT NOT NULL,
                guild_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                author_label TEXT NOT NULL,
                role TEXT NOT NULL,
                modality TEXT NOT NULL,
                content_raw TEXT NOT NULL,
                content_clean TEXT NOT NULL,
                source TEXT NOT NULL,
                quality DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS stt_turns (
                turn_id BIGSERIAL PRIMARY KEY,
                message_id BIGINT,
                guild_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                rms INTEGER NOT NULL,
                transcript TEXT,
                confidence DOUBLE PRECISION NOT NULL,
                model_name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                FOREIGN KEY(message_id) REFERENCES messages(message_id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS user_facts (
                fact_id BIGSERIAL PRIMARY KEY,
                guild_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                fact_key TEXT NOT NULL,
                fact_value TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                importance DOUBLE PRECISION NOT NULL,
                status TEXT NOT NULL,
                pinned BOOLEAN NOT NULL DEFAULT FALSE,
                evidence_count INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(guild_id, user_id, fact_key)
            );

            CREATE TABLE IF NOT EXISTS fact_evidence (
                evidence_id BIGSERIAL PRIMARY KEY,
                fact_id BIGINT NOT NULL,
                message_id BIGINT,
                extractor TEXT NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                FOREIGN KEY(fact_id) REFERENCES user_facts(fact_id) ON DELETE CASCADE,
                FOREIGN KEY(message_id) REFERENCES messages(message_id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS dialogue_summaries (
                summary_id BIGSERIAL PRIMARY KEY,
                guild_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                source_user_messages INTEGER NOT NULL,
                last_message_id BIGINT,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(guild_id, user_id, channel_id)
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_lookup
            ON sessions(guild_id, channel_id, mode, status, session_id DESC);

            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, message_id DESC);

            CREATE INDEX IF NOT EXISTS idx_messages_dialogue
            ON messages(guild_id, channel_id, message_id DESC);

            CREATE INDEX IF NOT EXISTS idx_facts_lookup
            ON user_facts(guild_id, user_id, pinned DESC, confidence DESC, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_summary_lookup
            ON dialogue_summaries(guild_id, user_id, channel_id);

            CREATE INDEX IF NOT EXISTS idx_stt_lookup
            ON stt_turns(guild_id, channel_id, user_id, created_at DESC);
            """
        )

    async def upsert_user_identity(
        self,
        guild_id: str,
        user_id: str,
        discord_username: str,
        discord_global_name: str | None,
        guild_nick: str | None,
        combined_label: str,
    ) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (
                    guild_id, user_id, discord_username, discord_global_name, guild_nick, combined_label, first_seen, updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW())
                ON CONFLICT(guild_id, user_id) DO UPDATE SET
                    discord_username = EXCLUDED.discord_username,
                    discord_global_name = EXCLUDED.discord_global_name,
                    guild_nick = EXCLUDED.guild_nick,
                    combined_label = EXCLUDED.combined_label,
                    updated_at = NOW()
                """,
                guild_id,
                user_id,
                discord_username,
                discord_global_name,
                guild_nick,
                combined_label,
            )

    async def get_user_identity(self, guild_id: str, user_id: str) -> Optional[Dict[str, str | None]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT guild_id, user_id, discord_username, discord_global_name, guild_nick, combined_label
                FROM users
                WHERE guild_id = $1 AND user_id = $2
                """,
                guild_id,
                user_id,
            )
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

    async def ensure_role_profile(
        self,
        role_id: str,
        name: str,
        goal: str,
        style: str,
        constraints: str,
    ) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO role_profiles (role_id, name, goal, style, constraints, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
                ON CONFLICT(role_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    goal = EXCLUDED.goal,
                    style = EXCLUDED.style,
                    constraints = EXCLUDED.constraints,
                    updated_at = NOW()
                """,
                role_id,
                name,
                goal,
                style,
                constraints,
            )

    async def ensure_guild_settings(self, guild_id: str, default_role_id: str) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO guild_settings (guild_id, default_role_id, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT(guild_id) DO UPDATE SET
                    default_role_id = EXCLUDED.default_role_id,
                    updated_at = NOW()
                """,
                guild_id,
                default_role_id,
            )

    async def get_guild_role_id(self, guild_id: str) -> str | None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT default_role_id FROM guild_settings WHERE guild_id = $1",
                guild_id,
            )
        if row is None:
            return None
        return str(row["default_role_id"])

    async def get_role_profile(self, role_id: str) -> Optional[Dict[str, str]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT role_id, name, goal, style, constraints
                FROM role_profiles
                WHERE role_id = $1
                """,
                role_id,
            )
        if row is None:
            return None
        return {
            "role_id": str(row["role_id"]),
            "name": str(row["name"]),
            "goal": str(row["goal"]),
            "style": str(row["style"]),
            "constraints": str(row["constraints"]),
        }

    async def get_or_create_session(
        self,
        guild_id: str,
        channel_id: str,
        mode: str,
        role_id: str,
    ) -> int:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT session_id
                FROM sessions
                WHERE guild_id = $1 AND channel_id = $2 AND mode = $3 AND status = 'active'
                ORDER BY session_id DESC
                LIMIT 1
                """,
                guild_id,
                channel_id,
                mode,
            )
            if row is not None:
                return int(row["session_id"])

            session_id = await conn.fetchval(
                """
                INSERT INTO sessions (guild_id, channel_id, mode, role_id, status, started_at)
                VALUES ($1, $2, $3, $4, 'active', NOW())
                RETURNING session_id
                """,
                guild_id,
                channel_id,
                mode,
                role_id,
            )
            return int(session_id)

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
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            message_id = await conn.fetchval(
                """
                INSERT INTO messages (
                    session_id, guild_id, channel_id, user_id, author_label,
                    role, modality, content_raw, content_clean, source, quality
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING message_id
                """,
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
            )
            return int(message_id)

    async def get_recent_session_messages(self, session_id: int, limit: int) -> List[Dict[str, object]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT message_id, user_id, author_label, role, modality, content_clean, source, quality, created_at
                FROM messages
                WHERE session_id = $1
                ORDER BY message_id DESC
                LIMIT $2
                """,
                int(session_id),
                max(1, int(limit)),
            )
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
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            value = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM messages
                WHERE guild_id = $1 AND channel_id = $2 AND user_id = $3 AND role = 'user'
                """,
                guild_id,
                channel_id,
                user_id,
            )
        return int(value or 0)

    async def get_recent_dialogue_messages(
        self,
        guild_id: str,
        channel_id: str,
        user_id: str,
        limit: int,
    ) -> List[Dict[str, object]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT message_id, role, user_id, author_label, content_clean
                FROM messages
                WHERE guild_id = $1
                  AND channel_id = $2
                  AND (
                    role = 'assistant'
                    OR (role = 'user' AND user_id = $3)
                  )
                ORDER BY message_id DESC
                LIMIT $4
                """,
                guild_id,
                channel_id,
                user_id,
                max(1, int(limit)),
            )
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

    async def get_dialogue_summary(self, guild_id: str, user_id: str, channel_id: str) -> Optional[Dict[str, object]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT summary_text, source_user_messages, last_message_id, updated_at
                FROM dialogue_summaries
                WHERE guild_id = $1 AND user_id = $2 AND channel_id = $3
                """,
                guild_id,
                user_id,
                channel_id,
            )
        if row is None:
            return None
        return {
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
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO dialogue_summaries (
                    guild_id, user_id, channel_id, summary_text, source_user_messages, last_message_id, updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, NOW())
                ON CONFLICT(guild_id, user_id, channel_id) DO UPDATE SET
                    summary_text = EXCLUDED.summary_text,
                    source_user_messages = EXCLUDED.source_user_messages,
                    last_message_id = EXCLUDED.last_message_id,
                    updated_at = NOW()
                """,
                guild_id,
                user_id,
                channel_id,
                cleaned,
                max(0, int(source_user_messages)),
                max(0, int(last_message_id)),
            )

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
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO stt_turns (
                    message_id, guild_id, channel_id, user_id, duration_ms, rms,
                    transcript, confidence, model_name, status
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
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
            )

    async def upsert_user_fact(
        self,
        guild_id: str,
        user_id: str,
        fact_key: str,
        fact_value: str,
        fact_type: str,
        confidence: float,
        importance: float,
        message_id: int | None,
        extractor: str,
    ) -> int:
        key = fact_key.strip().casefold()
        value = fact_value.strip()
        if not key or not value:
            return 0

        fact_type_clean = fact_type.strip().lower() or "fact"
        conf = _clamp(float(confidence), 0.0, 1.0)
        imp = _clamp(float(importance), 0.0, 1.0)

        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    SELECT fact_id, confidence, importance, evidence_count, status, pinned
                    FROM user_facts
                    WHERE guild_id = $1 AND user_id = $2 AND fact_key = $3
                    """,
                    guild_id,
                    user_id,
                    key,
                )

                if row is None:
                    status = "confirmed" if conf >= 0.78 else "candidate"
                    fact_id = await conn.fetchval(
                        """
                        INSERT INTO user_facts (
                            guild_id, user_id, fact_key, fact_value, fact_type,
                            confidence, importance, status, pinned, evidence_count,
                            created_at, updated_at, last_seen_at
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, FALSE, 1, NOW(), NOW(), NOW())
                        RETURNING fact_id
                        """,
                        guild_id,
                        user_id,
                        key,
                        value[:280],
                        fact_type_clean,
                        conf,
                        imp,
                        status,
                    )
                    fact_id = int(fact_id)
                else:
                    fact_id = int(row["fact_id"])
                    prior_conf = float(row["confidence"])
                    prior_imp = float(row["importance"])
                    prior_count = int(row["evidence_count"])
                    prior_status = str(row["status"])
                    pinned = bool(row["pinned"])

                    merged_conf = _clamp(max(prior_conf * 0.86, conf), 0.0, 1.0)
                    merged_imp = _clamp(max(prior_imp, imp), 0.0, 1.0)
                    merged_count = prior_count + 1

                    if pinned:
                        merged_status = "pinned"
                    elif merged_count >= 2 or merged_conf >= 0.78:
                        merged_status = "confirmed"
                    elif prior_status == "confirmed":
                        merged_status = "confirmed"
                    else:
                        merged_status = "candidate"

                    await conn.execute(
                        """
                        UPDATE user_facts
                        SET fact_value = $1,
                            fact_type = $2,
                            confidence = $3,
                            importance = $4,
                            status = $5,
                            evidence_count = $6,
                            updated_at = NOW(),
                            last_seen_at = NOW()
                        WHERE fact_id = $7
                        """,
                        value[:280],
                        fact_type_clean,
                        merged_conf,
                        merged_imp,
                        merged_status,
                        merged_count,
                        fact_id,
                    )

                await conn.execute(
                    """
                    INSERT INTO fact_evidence (fact_id, message_id, extractor, confidence, created_at)
                    VALUES ($1, $2, $3, $4, NOW())
                    """,
                    fact_id,
                    message_id,
                    extractor,
                    conf,
                )

                return fact_id

    async def get_user_facts(self, guild_id: str, user_id: str, limit: int) -> List[Dict[str, object]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT fact_id, fact_key, fact_value, fact_type, confidence, importance,
                       status, pinned, evidence_count, updated_at, last_seen_at
                FROM user_facts
                WHERE guild_id = $1 AND user_id = $2
                ORDER BY
                    pinned DESC,
                    CASE status
                        WHEN 'pinned' THEN 3
                        WHEN 'confirmed' THEN 2
                        ELSE 1
                    END DESC,
                    importance DESC,
                    confidence DESC,
                    evidence_count DESC,
                    updated_at DESC
                LIMIT $3
                """,
                guild_id,
                user_id,
                max(1, int(limit)),
            )
        return [
            {
                "fact_id": int(row["fact_id"]),
                "fact_key": str(row["fact_key"]),
                "fact_value": str(row["fact_value"]),
                "fact_type": str(row["fact_type"]),
                "confidence": float(row["confidence"]),
                "importance": float(row["importance"]),
                "status": str(row["status"]),
                "pinned": bool(row["pinned"]),
                "evidence_count": int(row["evidence_count"]),
                "updated_at": str(row["updated_at"]),
                "last_seen_at": str(row["last_seen_at"]),
            }
            for row in rows
        ]
