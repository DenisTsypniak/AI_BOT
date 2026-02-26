from __future__ import annotations

from pathlib import Path

import aiosqlite


class MemorySchemaMixin:
    SCHEMA_VERSION = 7

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def init(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA foreign_keys=ON")
            async with db.execute("PRAGMA user_version") as cursor:
                row = await cursor.fetchone()
            version = int(row[0]) if row else 0

            if version != self.SCHEMA_VERSION:
                await self._reset_schema(db)
                await db.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")
            else:
                await self._create_schema(db)

            await db.commit()

    async def _reset_schema(self, db: aiosqlite.Connection) -> None:
        tables = (
            "global_fact_evidence",
            "global_user_facts",
            "global_user_profiles",
            "conversation_messages",
            "guild_prompts",
            "guild_voice_settings",
            "user_profile_facts",
            "user_memory_items",
            "guild_user_identities",
            "voice_turn_logs",
            "guild_settings",
            "chat_messages",
            "user_identities",
            "user_facts",
            "tone_observations",
            "character_profiles",
            "dialogue_summaries",
            "voice_turn_audit",
            "sessions",
            "messages",
            "stt_turns",
            "fact_evidence",
            "role_profiles",
            "users",
        )
        for table in tables:
            await db.execute(f"DROP TABLE IF EXISTS {table}")
        await self._create_schema(db)

    async def _create_schema(self, db: aiosqlite.Connection) -> None:
        await db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                guild_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                discord_username TEXT NOT NULL,
                discord_global_name TEXT,
                guild_nick TEXT,
                combined_label TEXT NOT NULL,
                first_seen DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (guild_id, user_id)
            );

            CREATE TABLE IF NOT EXISTS global_user_profiles (
                user_id TEXT PRIMARY KEY,
                discord_username TEXT NOT NULL,
                discord_global_name TEXT,
                primary_display_name TEXT NOT NULL,
                first_seen DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS role_profiles (
                role_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                goal TEXT NOT NULL,
                style TEXT NOT NULL,
                constraints TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS guild_settings (
                guild_id TEXT PRIMARY KEY,
                default_role_id TEXT NOT NULL,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(default_role_id) REFERENCES role_profiles(role_id)
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                role_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                ended_at DATETIME,
                FOREIGN KEY(role_id) REFERENCES role_profiles(role_id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                guild_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                author_label TEXT NOT NULL,
                role TEXT NOT NULL,
                modality TEXT NOT NULL,
                content_raw TEXT NOT NULL,
                content_clean TEXT NOT NULL,
                source TEXT NOT NULL,
                quality REAL NOT NULL DEFAULT 1.0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS stt_turns (
                turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER,
                guild_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                rms INTEGER NOT NULL,
                transcript TEXT,
                confidence REAL NOT NULL,
                model_name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(message_id) REFERENCES messages(message_id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS user_facts (
                fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                fact_key TEXT NOT NULL,
                fact_value TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                importance REAL NOT NULL,
                status TEXT NOT NULL,
                pinned INTEGER NOT NULL DEFAULT 0,
                evidence_count INTEGER NOT NULL DEFAULT 1,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_seen_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(guild_id, user_id, fact_key)
            );

            CREATE TABLE IF NOT EXISTS global_user_facts (
                global_fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                fact_key TEXT NOT NULL,
                fact_value TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                importance REAL NOT NULL,
                status TEXT NOT NULL,
                pinned INTEGER NOT NULL DEFAULT 0,
                evidence_count INTEGER NOT NULL DEFAULT 1,
                first_seen_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_seen_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_source_guild_id TEXT,
                last_source_message_id INTEGER,
                UNIQUE(user_id, fact_key)
            );

            CREATE TABLE IF NOT EXISTS fact_evidence (
                evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id INTEGER NOT NULL,
                message_id INTEGER,
                extractor TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(fact_id) REFERENCES user_facts(fact_id) ON DELETE CASCADE,
                FOREIGN KEY(message_id) REFERENCES messages(message_id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS global_fact_evidence (
                global_evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                global_fact_id INTEGER NOT NULL,
                source_guild_id TEXT,
                source_message_id INTEGER,
                extractor TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(global_fact_id) REFERENCES global_user_facts(global_fact_id) ON DELETE CASCADE,
                FOREIGN KEY(source_message_id) REFERENCES messages(message_id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS dialogue_summaries (
                summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                source_user_messages INTEGER NOT NULL,
                last_message_id INTEGER,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
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

            CREATE INDEX IF NOT EXISTS idx_global_users_updated
            ON global_user_profiles(updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_global_facts_lookup
            ON global_user_facts(user_id, pinned DESC, confidence DESC, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_global_fact_evidence_lookup
            ON global_fact_evidence(global_fact_id, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_summary_lookup
            ON dialogue_summaries(guild_id, user_id, channel_id);

            CREATE INDEX IF NOT EXISTS idx_stt_lookup
            ON stt_turns(guild_id, channel_id, user_id, created_at DESC);
            """
        )
