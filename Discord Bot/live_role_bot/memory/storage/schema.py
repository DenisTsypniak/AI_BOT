from __future__ import annotations

import os
from pathlib import Path

import aiosqlite

from .utils import _sqlite_memory_connection


class MemorySchemaMixin:
    SCHEMA_VERSION = 9

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _allow_destructive_reset_on_mismatch() -> bool:
        raw = os.getenv("MEMORY_SQLITE_RESET_ON_SCHEMA_MISMATCH", "")
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

    async def _has_user_tables(self, db: aiosqlite.Connection) -> bool:
        async with db.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            LIMIT 1
            """
        ) as cursor:
            row = await cursor.fetchone()
        return bool(row)

    async def init(self) -> None:
        async with _sqlite_memory_connection(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute("PRAGMA user_version") as cursor:
                row = await cursor.fetchone()
            version = int(row[0]) if row else 0
            has_tables = await self._has_user_tables(db)

            if version > self.SCHEMA_VERSION:
                if has_tables and self._allow_destructive_reset_on_mismatch():
                    await self._reset_schema(db)
                    await db.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")
                    await db.commit()
                    return
                raise RuntimeError(
                    "SQLite schema version mismatch detected (database is newer than this bot build). "
                    f"Found user_version={version}, supported={self.SCHEMA_VERSION}. "
                    "Set MEMORY_SQLITE_RESET_ON_SCHEMA_MISMATCH=1 to allow destructive reset."
                )

            if not has_tables:
                await self._create_schema(db)
                await db.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")
            elif version != self.SCHEMA_VERSION and self._allow_destructive_reset_on_mismatch():
                await self._reset_schema(db)
                await db.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")
            else:
                await self._create_schema(db)
                await self._migrate_schema(db, version)
                if version != self.SCHEMA_VERSION:
                    await db.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")

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
            "global_biography_summaries",
            "memory_extractor_candidates",
            "memory_extractor_runs",
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

    async def _table_columns(self, db: aiosqlite.Connection, table_name: str) -> set[str]:
        cols: set[str] = set()
        try:
            async with db.execute(f"PRAGMA table_info({table_name})") as cursor:
                rows = await cursor.fetchall()
        except Exception:
            return cols
        for row in rows:
            try:
                cols.add(str(row[1]))
            except Exception:
                continue
        return cols

    async def _add_column_if_missing(self, db: aiosqlite.Connection, table_name: str, column_sql: str) -> None:
        column_name = str(column_sql.split()[0]).strip()
        if not column_name:
            return
        cols = await self._table_columns(db, table_name)
        if column_name in cols:
            return
        await db.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}")

    async def _migrate_schema(self, db: aiosqlite.Connection, from_version: int) -> None:
        if from_version < 8:
            await self._migrate_v8_fact_metadata_schema(db)
        if from_version < 9:
            await self._migrate_v9_memory_extractor_audit_schema(db)
        # Re-run idempotent migration to self-heal partial deployments.
        await self._migrate_v8_fact_metadata_schema(db)
        await self._migrate_v9_memory_extractor_audit_schema(db)

    async def _migrate_v8_fact_metadata_schema(self, db: aiosqlite.Connection) -> None:
        for table in ("user_facts", "global_user_facts"):
            await self._add_column_if_missing(db, table, "about_target TEXT NOT NULL DEFAULT 'self'")
            await self._add_column_if_missing(db, table, "directness TEXT NOT NULL DEFAULT 'explicit'")
            await self._add_column_if_missing(db, table, "evidence_quote TEXT NOT NULL DEFAULT ''")
        for table in ("fact_evidence", "global_fact_evidence"):
            await self._add_column_if_missing(db, table, "about_target TEXT NOT NULL DEFAULT 'self'")
            await self._add_column_if_missing(db, table, "directness TEXT NOT NULL DEFAULT 'explicit'")
            await self._add_column_if_missing(db, table, "evidence_quote TEXT NOT NULL DEFAULT ''")

        await db.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_facts_status_directness_lookup
            ON user_facts(guild_id, user_id, status, directness, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_global_facts_status_directness_lookup
            ON global_user_facts(user_id, status, directness, updated_at DESC);
            """
        )

    async def _migrate_v9_memory_extractor_audit_schema(self, db: aiosqlite.Connection) -> None:
        await db.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_extractor_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                speaker_user_id TEXT NOT NULL,
                fact_owner_kind TEXT NOT NULL,
                fact_owner_id TEXT NOT NULL,
                speaker_role TEXT NOT NULL,
                modality TEXT NOT NULL,
                source TEXT NOT NULL,
                backend_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                dry_run INTEGER NOT NULL DEFAULT 0,
                llm_attempted INTEGER NOT NULL DEFAULT 0,
                llm_ok INTEGER NOT NULL DEFAULT 0,
                json_valid INTEGER NOT NULL DEFAULT 0,
                fallback_used INTEGER NOT NULL DEFAULT 0,
                latency_ms INTEGER NOT NULL DEFAULT 0,
                candidate_count INTEGER NOT NULL DEFAULT 0,
                accepted_count INTEGER NOT NULL DEFAULT 0,
                saved_count INTEGER NOT NULL DEFAULT 0,
                filtered_count INTEGER NOT NULL DEFAULT 0,
                error_text TEXT NOT NULL DEFAULT '',
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS memory_extractor_candidates (
                candidate_row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                fact_key TEXT NOT NULL,
                fact_value TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                about_target TEXT NOT NULL DEFAULT 'self',
                directness TEXT NOT NULL DEFAULT 'explicit',
                evidence_quote TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL,
                importance REAL NOT NULL,
                moderation_action TEXT NOT NULL DEFAULT 'unknown',
                moderation_reason TEXT NOT NULL DEFAULT '',
                selected_for_apply INTEGER NOT NULL DEFAULT 0,
                saved_to_memory INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(run_id) REFERENCES memory_extractor_runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_memory_extractor_runs_created
            ON memory_extractor_runs(created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_memory_extractor_runs_backend_created
            ON memory_extractor_runs(backend_name, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_memory_extractor_runs_dryrun_created
            ON memory_extractor_runs(dry_run, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_memory_extractor_candidates_run
            ON memory_extractor_candidates(run_id, candidate_row_id DESC);

            CREATE INDEX IF NOT EXISTS idx_memory_extractor_candidates_action_created
            ON memory_extractor_candidates(moderation_action, created_at DESC);
            """
        )

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
                about_target TEXT NOT NULL DEFAULT 'self',
                directness TEXT NOT NULL DEFAULT 'explicit',
                evidence_quote TEXT NOT NULL DEFAULT '',
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
                about_target TEXT NOT NULL DEFAULT 'self',
                directness TEXT NOT NULL DEFAULT 'explicit',
                evidence_quote TEXT NOT NULL DEFAULT '',
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
                about_target TEXT NOT NULL DEFAULT 'self',
                directness TEXT NOT NULL DEFAULT 'explicit',
                evidence_quote TEXT NOT NULL DEFAULT '',
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
                about_target TEXT NOT NULL DEFAULT 'self',
                directness TEXT NOT NULL DEFAULT 'explicit',
                evidence_quote TEXT NOT NULL DEFAULT '',
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

            CREATE TABLE IF NOT EXISTS global_biography_summaries (
                subject_kind TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                source_fact_count INTEGER NOT NULL DEFAULT 0,
                source_summary_count INTEGER NOT NULL DEFAULT 0,
                source_updated_at DATETIME,
                last_source_guild_id TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (subject_kind, subject_id)
            );

            CREATE TABLE IF NOT EXISTS memory_extractor_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                speaker_user_id TEXT NOT NULL,
                fact_owner_kind TEXT NOT NULL,
                fact_owner_id TEXT NOT NULL,
                speaker_role TEXT NOT NULL,
                modality TEXT NOT NULL,
                source TEXT NOT NULL,
                backend_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                dry_run INTEGER NOT NULL DEFAULT 0,
                llm_attempted INTEGER NOT NULL DEFAULT 0,
                llm_ok INTEGER NOT NULL DEFAULT 0,
                json_valid INTEGER NOT NULL DEFAULT 0,
                fallback_used INTEGER NOT NULL DEFAULT 0,
                latency_ms INTEGER NOT NULL DEFAULT 0,
                candidate_count INTEGER NOT NULL DEFAULT 0,
                accepted_count INTEGER NOT NULL DEFAULT 0,
                saved_count INTEGER NOT NULL DEFAULT 0,
                filtered_count INTEGER NOT NULL DEFAULT 0,
                error_text TEXT NOT NULL DEFAULT '',
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS memory_extractor_candidates (
                candidate_row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                fact_key TEXT NOT NULL,
                fact_value TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                about_target TEXT NOT NULL DEFAULT 'self',
                directness TEXT NOT NULL DEFAULT 'explicit',
                evidence_quote TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL,
                importance REAL NOT NULL,
                moderation_action TEXT NOT NULL DEFAULT 'unknown',
                moderation_reason TEXT NOT NULL DEFAULT '',
                selected_for_apply INTEGER NOT NULL DEFAULT 0,
                saved_to_memory INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(run_id) REFERENCES memory_extractor_runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_lookup
            ON sessions(guild_id, channel_id, mode, status, session_id DESC);

            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, message_id DESC);

            CREATE INDEX IF NOT EXISTS idx_messages_dialogue
            ON messages(guild_id, channel_id, message_id DESC);

            CREATE INDEX IF NOT EXISTS idx_facts_lookup
            ON user_facts(guild_id, user_id, pinned DESC, confidence DESC, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_facts_status_directness_lookup
            ON user_facts(guild_id, user_id, status, directness, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_global_users_updated
            ON global_user_profiles(updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_global_facts_lookup
            ON global_user_facts(user_id, pinned DESC, confidence DESC, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_global_facts_status_directness_lookup
            ON global_user_facts(user_id, status, directness, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_global_fact_evidence_lookup
            ON global_fact_evidence(global_fact_id, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_global_fact_evidence_source_lookup
            ON global_fact_evidence(global_fact_id, source_guild_id, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_summary_lookup
            ON dialogue_summaries(guild_id, user_id, channel_id);

            CREATE INDEX IF NOT EXISTS idx_global_biography_summaries_lookup
            ON global_biography_summaries(subject_kind, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_memory_extractor_runs_created
            ON memory_extractor_runs(created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_memory_extractor_runs_backend_created
            ON memory_extractor_runs(backend_name, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_memory_extractor_runs_dryrun_created
            ON memory_extractor_runs(dry_run, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_memory_extractor_candidates_run
            ON memory_extractor_candidates(run_id, candidate_row_id DESC);

            CREATE INDEX IF NOT EXISTS idx_memory_extractor_candidates_action_created
            ON memory_extractor_candidates(moderation_action, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_stt_lookup
            ON stt_turns(guild_id, channel_id, user_id, created_at DESC);
            """
        )
