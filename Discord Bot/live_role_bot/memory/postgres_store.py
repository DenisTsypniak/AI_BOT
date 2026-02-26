from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import asyncpg
except Exception:  # pragma: no cover - optional dependency at runtime
    asyncpg = None  # type: ignore[assignment]

from .storage.utils import (
    _clamp,
    apply_memory_fact_promotion_policy,
    merge_memory_fact_metadata_state,
    merge_memory_fact_state,
    normalize_memory_fact_about_target,
    normalize_memory_fact_directness,
    sanitize_memory_fact_evidence_quote,
)


logger = logging.getLogger("live_role_bot")


def _tokenize_text(text: str) -> set[str]:
    words = re.findall(r"[\w']{2,}", (text or "").casefold(), flags=re.UNICODE)
    stop = {
        "the",
        "and",
        "for",
        "that",
        "this",
        "you",
        "your",
        "with",
        "have",
        "just",
        "like",
    }
    return {word for word in words if word not in stop}


class PostgresMemoryStore:
    """Postgres-backed memory store implementing the same API as MemoryStore."""

    SCHEMA_VERSION = 12
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
                    if version > self.SCHEMA_VERSION:
                        raise RuntimeError(
                            f"Postgres memory schema version {version} is newer than supported {self.SCHEMA_VERSION}. "
                            "Upgrade the bot before starting."
                        )
                    if 0 < version < 7:
                        raise RuntimeError(
                            f"Unsupported legacy Postgres memory schema version {version}. "
                            "Manual migration is required before upgrading to persona-growth schema (v8+)."
                        )
                    await self._create_schema(conn)
                    await self._migrate_schema(conn, version)
                    await self._backfill_global_user_memory_tables(conn)
                    if version != self.SCHEMA_VERSION:
                        await self._set_schema_version(conn, self.SCHEMA_VERSION)
            self._initialized = True

    async def _migrate_schema(self, conn: "asyncpg.Connection", from_version: int) -> None:
        # v8: Persona Growth Engine MVP schema (non-destructive additive migration).
        if from_version < 8:
            await self._create_persona_mvp_schema(conn)
        # Keep idempotent create calls in init to self-heal partial deployments.
        await self._create_persona_mvp_schema(conn)
        # v9: Relationship policy evidence + episodic memory candidates + ingest idempotency.
        await self._create_persona_phase23_schema(conn)
        # v10: Trait evidence + reflection apply-path support (additive).
        await self._create_persona_phase6_schema(conn)
        # v11: Memory fact metadata for implicit fact pipeline (additive).
        await self._create_memory_fact_pipeline_v11_schema(conn)
        # v12: Extractor diagnostics + dry-run candidate audit tables.
        await self._create_memory_extractor_audit_v12_schema(conn)

    async def _create_memory_fact_pipeline_v11_schema(self, conn: "asyncpg.Connection") -> None:
        await conn.execute(
            """
            ALTER TABLE user_facts
            ADD COLUMN IF NOT EXISTS about_target TEXT NOT NULL DEFAULT 'self';

            ALTER TABLE user_facts
            ADD COLUMN IF NOT EXISTS directness TEXT NOT NULL DEFAULT 'explicit';

            ALTER TABLE user_facts
            ADD COLUMN IF NOT EXISTS evidence_quote TEXT NOT NULL DEFAULT '';

            ALTER TABLE global_user_facts
            ADD COLUMN IF NOT EXISTS about_target TEXT NOT NULL DEFAULT 'self';

            ALTER TABLE global_user_facts
            ADD COLUMN IF NOT EXISTS directness TEXT NOT NULL DEFAULT 'explicit';

            ALTER TABLE global_user_facts
            ADD COLUMN IF NOT EXISTS evidence_quote TEXT NOT NULL DEFAULT '';

            ALTER TABLE fact_evidence
            ADD COLUMN IF NOT EXISTS about_target TEXT NOT NULL DEFAULT 'self';

            ALTER TABLE fact_evidence
            ADD COLUMN IF NOT EXISTS directness TEXT NOT NULL DEFAULT 'explicit';

            ALTER TABLE fact_evidence
            ADD COLUMN IF NOT EXISTS evidence_quote TEXT NOT NULL DEFAULT '';

            ALTER TABLE global_fact_evidence
            ADD COLUMN IF NOT EXISTS about_target TEXT NOT NULL DEFAULT 'self';

            ALTER TABLE global_fact_evidence
            ADD COLUMN IF NOT EXISTS directness TEXT NOT NULL DEFAULT 'explicit';

            ALTER TABLE global_fact_evidence
            ADD COLUMN IF NOT EXISTS evidence_quote TEXT NOT NULL DEFAULT '';

            CREATE INDEX IF NOT EXISTS idx_facts_status_directness_lookup
            ON user_facts(guild_id, user_id, status, directness, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_global_facts_status_directness_lookup
            ON global_user_facts(user_id, status, directness, updated_at DESC);
            """
        )

    async def _create_memory_extractor_audit_v12_schema(self, conn: "asyncpg.Connection") -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_extractor_runs (
                run_id BIGSERIAL PRIMARY KEY,
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
                dry_run BOOLEAN NOT NULL DEFAULT FALSE,
                llm_attempted BOOLEAN NOT NULL DEFAULT FALSE,
                llm_ok BOOLEAN NOT NULL DEFAULT FALSE,
                json_valid BOOLEAN NOT NULL DEFAULT FALSE,
                fallback_used BOOLEAN NOT NULL DEFAULT FALSE,
                latency_ms INTEGER NOT NULL DEFAULT 0,
                candidate_count INTEGER NOT NULL DEFAULT 0,
                accepted_count INTEGER NOT NULL DEFAULT 0,
                saved_count INTEGER NOT NULL DEFAULT 0,
                filtered_count INTEGER NOT NULL DEFAULT 0,
                error_text TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS memory_extractor_candidates (
                candidate_row_id BIGSERIAL PRIMARY KEY,
                run_id BIGINT NOT NULL,
                fact_key TEXT NOT NULL,
                fact_value TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                about_target TEXT NOT NULL DEFAULT 'self',
                directness TEXT NOT NULL DEFAULT 'explicit',
                evidence_quote TEXT NOT NULL DEFAULT '',
                confidence DOUBLE PRECISION NOT NULL,
                importance DOUBLE PRECISION NOT NULL,
                moderation_action TEXT NOT NULL DEFAULT 'unknown',
                moderation_reason TEXT NOT NULL DEFAULT '',
                selected_for_apply BOOLEAN NOT NULL DEFAULT FALSE,
                saved_to_memory BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                FOREIGN KEY (run_id) REFERENCES memory_extractor_runs(run_id) ON DELETE CASCADE
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

    async def _create_persona_mvp_schema(self, conn: "asyncpg.Connection") -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS persona_global_state (
                persona_id TEXT PRIMARY KEY,
                core_dna_hash TEXT NOT NULL,
                core_dna_source TEXT NOT NULL,
                policy_version INTEGER NOT NULL,
                current_era_id BIGINT,
                reflection_cursor_message_id BIGINT NOT NULL DEFAULT 0,
                last_reflection_at TIMESTAMPTZ,
                last_decay_at TIMESTAMPTZ,
                overlay_summary TEXT NOT NULL DEFAULT '',
                total_messages_seen BIGINT NOT NULL DEFAULT 0,
                eligible_messages_seen BIGINT NOT NULL DEFAULT 0,
                unique_users_seen INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS persona_trait_catalog (
                trait_key TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                description TEXT NOT NULL,
                default_anchor_value DOUBLE PRECISION NOT NULL,
                min_value DOUBLE PRECISION NOT NULL,
                max_value DOUBLE PRECISION NOT NULL,
                max_abs_drift DOUBLE PRECISION NOT NULL,
                max_step_per_reflection DOUBLE PRECISION NOT NULL,
                plasticity DOUBLE PRECISION NOT NULL,
                protected_mode TEXT NOT NULL,
                prompt_exposure TEXT NOT NULL,
                anchor_source TEXT NOT NULL DEFAULT 'neutral',
                notes TEXT NOT NULL DEFAULT '',
                enabled BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS persona_traits (
                persona_id TEXT NOT NULL,
                trait_key TEXT NOT NULL,
                anchor_value DOUBLE PRECISION NOT NULL,
                current_value DOUBLE PRECISION NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                drift_velocity DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                evidence_count INTEGER NOT NULL DEFAULT 0,
                support_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                contradiction_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                status TEXT NOT NULL DEFAULT 'emerging',
                prompt_exposure TEXT NOT NULL DEFAULT 'relevant',
                last_changed_at TIMESTAMPTZ,
                last_reconfirmed_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (persona_id, trait_key),
                FOREIGN KEY (persona_id) REFERENCES persona_global_state(persona_id) ON DELETE CASCADE,
                FOREIGN KEY (trait_key) REFERENCES persona_trait_catalog(trait_key) ON DELETE RESTRICT
            );

            CREATE TABLE IF NOT EXISTS persona_relationships (
                persona_id TEXT NOT NULL,
                guild_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_label_cache TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                consent_scope TEXT NOT NULL DEFAULT 'full',
                familiarity DOUBLE PRECISION NOT NULL DEFAULT 0.10,
                trust DOUBLE PRECISION NOT NULL DEFAULT 0.35,
                warmth DOUBLE PRECISION NOT NULL DEFAULT 0.45,
                banter_license DOUBLE PRECISION NOT NULL DEFAULT 0.25,
                support_sensitivity DOUBLE PRECISION NOT NULL DEFAULT 0.55,
                boundary_sensitivity DOUBLE PRECISION NOT NULL DEFAULT 0.60,
                topic_alignment DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                confidence DOUBLE PRECISION NOT NULL DEFAULT 0.20,
                interaction_count INTEGER NOT NULL DEFAULT 0,
                voice_turn_count INTEGER NOT NULL DEFAULT 0,
                text_turn_count INTEGER NOT NULL DEFAULT 0,
                effective_influence_weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                relationship_summary TEXT NOT NULL DEFAULT '',
                inside_joke_summary TEXT NOT NULL DEFAULT '',
                preferred_style_notes TEXT NOT NULL DEFAULT '',
                risk_flags JSONB NOT NULL DEFAULT '{}'::jsonb,
                last_interaction_at TIMESTAMPTZ,
                last_reflection_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (persona_id, guild_id, user_id),
                FOREIGN KEY (persona_id) REFERENCES persona_global_state(persona_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS persona_reflections (
                reflection_id BIGSERIAL PRIMARY KEY,
                persona_id TEXT NOT NULL,
                dedupe_key TEXT,
                trigger_type TEXT NOT NULL,
                trigger_reason TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                window_start_message_id BIGINT,
                window_end_message_id BIGINT,
                input_counts JSONB NOT NULL DEFAULT '{}'::jsonb,
                proposal_json JSONB,
                validator_report_json JSONB,
                applied_changes_json JSONB,
                rejection_reason TEXT NOT NULL DEFAULT '',
                model_name TEXT NOT NULL DEFAULT '',
                prompt_version TEXT NOT NULL DEFAULT '',
                duration_ms INTEGER,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                started_at TIMESTAMPTZ,
                finished_at TIMESTAMPTZ,
                applied_at TIMESTAMPTZ,
                FOREIGN KEY (persona_id) REFERENCES persona_global_state(persona_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS persona_audit_log (
                audit_id BIGSERIAL PRIMARY KEY,
                persona_id TEXT NOT NULL,
                actor_type TEXT NOT NULL,
                actor_user_id TEXT,
                action TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                before_json JSONB,
                after_json JSONB,
                diff_json JSONB,
                evidence_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
                reason TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                FOREIGN KEY (persona_id) REFERENCES persona_global_state(persona_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS persona_user_memory_prefs (
                guild_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                memory_mode TEXT NOT NULL DEFAULT 'full',
                allow_episodic_callbacks BOOLEAN NOT NULL DEFAULT TRUE,
                allow_personality_influence BOOLEAN NOT NULL DEFAULT TRUE,
                allow_sensitive_storage BOOLEAN NOT NULL DEFAULT FALSE,
                retention_days INTEGER,
                updated_by_admin_user_id TEXT,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (guild_id, user_id)
            );

            CREATE INDEX IF NOT EXISTS idx_persona_traits_lookup
            ON persona_traits(persona_id, status, confidence DESC, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_persona_relationships_lookup
            ON persona_relationships(persona_id, guild_id, confidence DESC, updated_at DESC);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_persona_reflections_dedupe
            ON persona_reflections(dedupe_key)
            WHERE dedupe_key IS NOT NULL AND dedupe_key <> '';

            CREATE INDEX IF NOT EXISTS idx_persona_reflections_lookup
            ON persona_reflections(persona_id, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_persona_audit_lookup
            ON persona_audit_log(persona_id, created_at DESC);
            """
        )

    async def _backfill_global_user_memory_tables(self, conn: "asyncpg.Connection") -> None:
        # Idempotent backfill so existing per-guild memory becomes visible in global card tables.
        await conn.execute(
            """
            INSERT INTO global_user_profiles (
                user_id, discord_username, discord_global_name, primary_display_name, first_seen, updated_at
            )
            SELECT latest.user_id,
                   latest.discord_username,
                   latest.discord_global_name,
                   COALESCE(NULLIF(latest.discord_global_name, ''), NULLIF(latest.discord_username, ''), 'unknown') AS primary_display_name,
                   first_seen_map.first_seen,
                   latest.updated_at
            FROM (
                SELECT DISTINCT ON (user_id)
                    user_id, discord_username, discord_global_name, updated_at
                FROM users
                ORDER BY user_id, updated_at DESC
            ) AS latest
            JOIN (
                SELECT user_id, MIN(first_seen) AS first_seen
                FROM users
                GROUP BY user_id
            ) AS first_seen_map USING (user_id)
            ON CONFLICT(user_id) DO UPDATE SET
                discord_username = EXCLUDED.discord_username,
                discord_global_name = COALESCE(EXCLUDED.discord_global_name, global_user_profiles.discord_global_name),
                primary_display_name = COALESCE(EXCLUDED.discord_global_name, EXCLUDED.discord_username, global_user_profiles.primary_display_name),
                updated_at = GREATEST(global_user_profiles.updated_at, EXCLUDED.updated_at)
            """
        )

        await conn.execute(
            """
            WITH ranked AS (
                SELECT uf.*,
                       ROW_NUMBER() OVER (
                           PARTITION BY uf.user_id, uf.fact_key
                           ORDER BY
                               uf.pinned DESC,
                               CASE uf.status
                                   WHEN 'pinned' THEN 3
                                   WHEN 'confirmed' THEN 2
                                   ELSE 1
                               END DESC,
                               uf.importance DESC,
                               uf.confidence DESC,
                               uf.evidence_count DESC,
                               uf.updated_at DESC
                       ) AS rn
                FROM user_facts uf
            )
            INSERT INTO global_user_facts (
                user_id, fact_key, fact_value, fact_type, about_target, directness, evidence_quote,
                confidence, importance, status, pinned, evidence_count,
                first_seen_at, updated_at, last_seen_at, last_source_guild_id, last_source_message_id
            )
            SELECT
                r.user_id, r.fact_key, r.fact_value, r.fact_type,
                COALESCE(r.about_target, 'self'),
                COALESCE(r.directness, 'explicit'),
                COALESCE(r.evidence_quote, ''),
                r.confidence, r.importance, r.status, r.pinned, r.evidence_count,
                r.created_at, r.updated_at, r.last_seen_at, r.guild_id, NULL
            FROM ranked r
            WHERE r.rn = 1
            ON CONFLICT(user_id, fact_key) DO UPDATE SET
                fact_value = CASE
                    WHEN global_user_facts.pinned THEN global_user_facts.fact_value
                    WHEN LOWER(BTRIM(global_user_facts.fact_value)) = LOWER(BTRIM(EXCLUDED.fact_value)) THEN EXCLUDED.fact_value
                    WHEN EXCLUDED.confidence >= global_user_facts.confidence + 0.18 THEN EXCLUDED.fact_value
                    WHEN (
                        NOT (
                            global_user_facts.status IN ('confirmed', 'pinned')
                            OR global_user_facts.confidence >= 0.78
                        )
                        AND EXCLUDED.confidence >= GREATEST(0.55, global_user_facts.confidence + 0.05)
                    ) THEN EXCLUDED.fact_value
                    WHEN global_user_facts.confidence < 0.45 AND EXCLUDED.confidence >= 0.55 THEN EXCLUDED.fact_value
                    ELSE global_user_facts.fact_value
                END,
                fact_type = CASE
                    WHEN global_user_facts.pinned THEN global_user_facts.fact_type
                    WHEN LOWER(BTRIM(global_user_facts.fact_value)) = LOWER(BTRIM(EXCLUDED.fact_value)) THEN EXCLUDED.fact_type
                    WHEN EXCLUDED.confidence >= global_user_facts.confidence + 0.18 THEN EXCLUDED.fact_type
                    WHEN (
                        NOT (
                            global_user_facts.status IN ('confirmed', 'pinned')
                            OR global_user_facts.confidence >= 0.78
                        )
                        AND EXCLUDED.confidence >= GREATEST(0.55, global_user_facts.confidence + 0.05)
                    ) THEN EXCLUDED.fact_type
                    WHEN global_user_facts.confidence < 0.45 AND EXCLUDED.confidence >= 0.55 THEN EXCLUDED.fact_type
                    ELSE global_user_facts.fact_type
                END,
                about_target = CASE
                    WHEN global_user_facts.about_target = 'unknown' THEN COALESCE(EXCLUDED.about_target, 'self')
                    ELSE global_user_facts.about_target
                END,
                directness = CASE
                    WHEN global_user_facts.directness = 'explicit' THEN global_user_facts.directness
                    WHEN EXCLUDED.directness = 'explicit' THEN EXCLUDED.directness
                    WHEN global_user_facts.directness = 'implicit' THEN global_user_facts.directness
                    ELSE COALESCE(EXCLUDED.directness, global_user_facts.directness)
                END,
                evidence_quote = CASE
                    WHEN NULLIF(BTRIM(EXCLUDED.evidence_quote), '') IS NOT NULL THEN EXCLUDED.evidence_quote
                    ELSE global_user_facts.evidence_quote
                END,
                confidence = GREATEST(global_user_facts.confidence * 0.9, EXCLUDED.confidence),
                importance = GREATEST(global_user_facts.importance, EXCLUDED.importance),
                status = CASE
                    WHEN global_user_facts.pinned THEN 'pinned'
                    WHEN global_user_facts.status = 'confirmed' OR EXCLUDED.status = 'confirmed' THEN 'confirmed'
                    ELSE EXCLUDED.status
                END,
                pinned = (global_user_facts.pinned OR EXCLUDED.pinned),
                evidence_count = GREATEST(global_user_facts.evidence_count, EXCLUDED.evidence_count),
                updated_at = GREATEST(global_user_facts.updated_at, EXCLUDED.updated_at),
                last_seen_at = GREATEST(global_user_facts.last_seen_at, EXCLUDED.last_seen_at),
                last_source_guild_id = COALESCE(EXCLUDED.last_source_guild_id, global_user_facts.last_source_guild_id)
            """
        )

    async def _create_persona_phase23_schema(self, conn: "asyncpg.Connection") -> None:
        await conn.execute(
            """
            ALTER TABLE persona_relationships
            ADD COLUMN IF NOT EXISTS daily_influence_budget_day DATE;

            ALTER TABLE persona_relationships
            ADD COLUMN IF NOT EXISTS daily_influence_budget_used DOUBLE PRECISION NOT NULL DEFAULT 0.0;

            ALTER TABLE persona_relationships
            ADD COLUMN IF NOT EXISTS last_message_id BIGINT;

            ALTER TABLE persona_relationships
            ADD COLUMN IF NOT EXISTS consistency_score DOUBLE PRECISION NOT NULL DEFAULT 0.50;

            ALTER TABLE persona_relationships
            ADD COLUMN IF NOT EXISTS last_episode_candidate_at TIMESTAMPTZ;

            CREATE TABLE IF NOT EXISTS persona_ingested_messages (
                persona_id TEXT NOT NULL,
                message_id BIGINT NOT NULL,
                guild_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                modality TEXT NOT NULL,
                source TEXT NOT NULL,
                quality DOUBLE PRECISION NOT NULL,
                event_kind TEXT NOT NULL DEFAULT 'user_message',
                ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (persona_id, message_id),
                FOREIGN KEY (persona_id) REFERENCES persona_global_state(persona_id) ON DELETE CASCADE,
                FOREIGN KEY (message_id) REFERENCES messages(message_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS persona_relationship_evidence (
                relationship_evidence_id BIGSERIAL PRIMARY KEY,
                persona_id TEXT NOT NULL,
                guild_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                message_id BIGINT,
                dimension_key TEXT NOT NULL,
                delta DOUBLE PRECISION NOT NULL,
                signal_confidence DOUBLE PRECISION NOT NULL,
                quality_weight DOUBLE PRECISION NOT NULL,
                influence_weight DOUBLE PRECISION NOT NULL,
                reason_text TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                FOREIGN KEY (persona_id) REFERENCES persona_global_state(persona_id) ON DELETE CASCADE,
                FOREIGN KEY (message_id) REFERENCES messages(message_id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS persona_episodes (
                episode_id BIGSERIAL PRIMARY KEY,
                persona_id TEXT NOT NULL,
                guild_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                source_dedupe_key TEXT NOT NULL,
                episode_type TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                callback_line TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'candidate',
                privacy_level TEXT NOT NULL DEFAULT 'participants_only',
                callback_safety TEXT NOT NULL DEFAULT 'safe',
                valence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                importance DOUBLE PRECISION NOT NULL,
                vividness DOUBLE PRECISION NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                source_message_count INTEGER NOT NULL DEFAULT 1,
                first_message_id BIGINT,
                last_message_id BIGINT,
                recall_count INTEGER NOT NULL DEFAULT 0,
                last_recalled_at TIMESTAMPTZ,
                last_confirmed_at TIMESTAMPTZ,
                last_participant_user_id TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                FOREIGN KEY (persona_id) REFERENCES persona_global_state(persona_id) ON DELETE CASCADE,
                UNIQUE (source_dedupe_key)
            );

            CREATE TABLE IF NOT EXISTS persona_episode_participants (
                episode_id BIGINT NOT NULL,
                guild_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                participant_role TEXT NOT NULL DEFAULT 'speaker',
                influence_weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                mention_count INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (episode_id, user_id),
                FOREIGN KEY (episode_id) REFERENCES persona_episodes(episode_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS persona_episode_evidence (
                episode_evidence_id BIGSERIAL PRIMARY KEY,
                episode_id BIGINT NOT NULL,
                message_id BIGINT,
                user_id TEXT,
                extractor TEXT NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                snippet TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                FOREIGN KEY (episode_id) REFERENCES persona_episodes(episode_id) ON DELETE CASCADE,
                FOREIGN KEY (message_id) REFERENCES messages(message_id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_persona_ingested_messages_lookup
            ON persona_ingested_messages(persona_id, ingested_at DESC);

            CREATE INDEX IF NOT EXISTS idx_persona_relationship_evidence_lookup
            ON persona_relationship_evidence(persona_id, guild_id, user_id, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_persona_episodes_lookup
            ON persona_episodes(persona_id, guild_id, status, importance DESC, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_persona_episodes_channel_lookup
            ON persona_episodes(persona_id, guild_id, channel_id, updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_persona_episode_participants_user_lookup
            ON persona_episode_participants(guild_id, user_id, episode_id DESC);

            CREATE INDEX IF NOT EXISTS idx_persona_episode_evidence_lookup
            ON persona_episode_evidence(episode_id, created_at DESC);
            """
        )

    async def _create_persona_phase6_schema(self, conn: "asyncpg.Connection") -> None:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS persona_trait_evidence (
                trait_evidence_id BIGSERIAL PRIMARY KEY,
                persona_id TEXT NOT NULL,
                trait_key TEXT NOT NULL,
                reflection_id BIGINT,
                signal_kind TEXT NOT NULL DEFAULT 'reflection_apply',
                direction DOUBLE PRECISION NOT NULL,
                magnitude DOUBLE PRECISION NOT NULL,
                signal_confidence DOUBLE PRECISION NOT NULL,
                quality_weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                influence_weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                reason_text TEXT NOT NULL DEFAULT '',
                evidence_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                FOREIGN KEY (persona_id) REFERENCES persona_global_state(persona_id) ON DELETE CASCADE,
                FOREIGN KEY (trait_key) REFERENCES persona_trait_catalog(trait_key) ON DELETE CASCADE,
                FOREIGN KEY (reflection_id) REFERENCES persona_reflections(reflection_id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_persona_trait_evidence_lookup
            ON persona_trait_evidence(persona_id, trait_key, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_persona_trait_evidence_reflection_lookup
            ON persona_trait_evidence(reflection_id, created_at DESC);
            """
        )

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
            "global_fact_evidence",
            "global_user_facts",
            "global_user_profiles",
            "fact_evidence",
            "stt_turns",
            "messages",
            "sessions",
            "dialogue_summaries",
            "global_biography_summaries",
            "memory_extractor_candidates",
            "memory_extractor_runs",
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

            CREATE TABLE IF NOT EXISTS global_user_profiles (
                user_id TEXT PRIMARY KEY,
                discord_username TEXT NOT NULL,
                discord_global_name TEXT,
                primary_display_name TEXT NOT NULL,
                first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
                about_target TEXT NOT NULL DEFAULT 'self',
                directness TEXT NOT NULL DEFAULT 'explicit',
                evidence_quote TEXT NOT NULL DEFAULT '',
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

            CREATE TABLE IF NOT EXISTS global_user_facts (
                global_fact_id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                fact_key TEXT NOT NULL,
                fact_value TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                about_target TEXT NOT NULL DEFAULT 'self',
                directness TEXT NOT NULL DEFAULT 'explicit',
                evidence_quote TEXT NOT NULL DEFAULT '',
                confidence DOUBLE PRECISION NOT NULL,
                importance DOUBLE PRECISION NOT NULL,
                status TEXT NOT NULL,
                pinned BOOLEAN NOT NULL DEFAULT FALSE,
                evidence_count INTEGER NOT NULL DEFAULT 1,
                first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_source_guild_id TEXT,
                last_source_message_id BIGINT,
                UNIQUE(user_id, fact_key)
            );

            CREATE TABLE IF NOT EXISTS fact_evidence (
                evidence_id BIGSERIAL PRIMARY KEY,
                fact_id BIGINT NOT NULL,
                message_id BIGINT,
                extractor TEXT NOT NULL,
                about_target TEXT NOT NULL DEFAULT 'self',
                directness TEXT NOT NULL DEFAULT 'explicit',
                evidence_quote TEXT NOT NULL DEFAULT '',
                confidence DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                FOREIGN KEY(fact_id) REFERENCES user_facts(fact_id) ON DELETE CASCADE,
                FOREIGN KEY(message_id) REFERENCES messages(message_id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS global_fact_evidence (
                global_evidence_id BIGSERIAL PRIMARY KEY,
                global_fact_id BIGINT NOT NULL,
                source_guild_id TEXT,
                source_message_id BIGINT,
                extractor TEXT NOT NULL,
                about_target TEXT NOT NULL DEFAULT 'self',
                directness TEXT NOT NULL DEFAULT 'explicit',
                evidence_quote TEXT NOT NULL DEFAULT '',
                confidence DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                FOREIGN KEY(global_fact_id) REFERENCES global_user_facts(global_fact_id) ON DELETE CASCADE,
                FOREIGN KEY(source_message_id) REFERENCES messages(message_id) ON DELETE SET NULL
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

            CREATE TABLE IF NOT EXISTS global_biography_summaries (
                subject_kind TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                source_fact_count INTEGER NOT NULL DEFAULT 0,
                source_summary_count INTEGER NOT NULL DEFAULT 0,
                source_updated_at TIMESTAMPTZ,
                last_source_guild_id TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY(subject_kind, subject_id)
            );

            CREATE TABLE IF NOT EXISTS memory_extractor_runs (
                run_id BIGSERIAL PRIMARY KEY,
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
                dry_run BOOLEAN NOT NULL DEFAULT FALSE,
                llm_attempted BOOLEAN NOT NULL DEFAULT FALSE,
                llm_ok BOOLEAN NOT NULL DEFAULT FALSE,
                json_valid BOOLEAN NOT NULL DEFAULT FALSE,
                fallback_used BOOLEAN NOT NULL DEFAULT FALSE,
                latency_ms INTEGER NOT NULL DEFAULT 0,
                candidate_count INTEGER NOT NULL DEFAULT 0,
                accepted_count INTEGER NOT NULL DEFAULT 0,
                saved_count INTEGER NOT NULL DEFAULT 0,
                filtered_count INTEGER NOT NULL DEFAULT 0,
                error_text TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS memory_extractor_candidates (
                candidate_row_id BIGSERIAL PRIMARY KEY,
                run_id BIGINT NOT NULL,
                fact_key TEXT NOT NULL,
                fact_value TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                about_target TEXT NOT NULL DEFAULT 'self',
                directness TEXT NOT NULL DEFAULT 'explicit',
                evidence_quote TEXT NOT NULL DEFAULT '',
                confidence DOUBLE PRECISION NOT NULL,
                importance DOUBLE PRECISION NOT NULL,
                moderation_action TEXT NOT NULL DEFAULT 'unknown',
                moderation_reason TEXT NOT NULL DEFAULT '',
                selected_for_apply BOOLEAN NOT NULL DEFAULT FALSE,
                saved_to_memory BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
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

            CREATE INDEX IF NOT EXISTS idx_global_users_updated
            ON global_user_profiles(updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_global_facts_lookup
            ON global_user_facts(user_id, pinned DESC, confidence DESC, updated_at DESC);

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
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
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
                await conn.execute(
                    """
                    INSERT INTO global_user_profiles (
                        user_id, discord_username, discord_global_name, primary_display_name, first_seen, updated_at
                    )
                    VALUES ($1, $2, $3, $4, NOW(), NOW())
                    ON CONFLICT(user_id) DO UPDATE SET
                        discord_username = EXCLUDED.discord_username,
                        discord_global_name = COALESCE(EXCLUDED.discord_global_name, global_user_profiles.discord_global_name),
                        primary_display_name = COALESCE(EXCLUDED.discord_global_name, EXCLUDED.discord_username, global_user_profiles.primary_display_name),
                        updated_at = NOW()
                    """,
                    user_id,
                    discord_username,
                    discord_global_name,
                    primary_display_name,
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

    async def get_latest_user_identity_by_user_id(
        self,
        user_id: str,
        *,
        exclude_guild_id: str | None = None,
    ) -> Optional[Dict[str, str | None]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            global_row = await conn.fetchrow(
                """
                SELECT user_id, discord_username, discord_global_name, primary_display_name
                FROM global_user_profiles
                WHERE user_id = $1
                """,
                user_id,
            )
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
                row = await conn.fetchrow(
                    """
                    SELECT guild_id, user_id, discord_username, discord_global_name, guild_nick, combined_label
                    FROM users
                    WHERE user_id = $1
                      AND guild_id <> $2
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    user_id,
                    exclude_guild_id,
                )
            else:
                row = await conn.fetchrow(
                    """
                    SELECT guild_id, user_id, discord_username, discord_global_name, guild_nick, combined_label
                    FROM users
                    WHERE user_id = $1
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
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

    async def get_latest_dialogue_summary_by_user_id(
        self,
        user_id: str,
        *,
        exclude_guild_id: str | None = None,
    ) -> Optional[Dict[str, object]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            if exclude_guild_id:
                row = await conn.fetchrow(
                    """
                    SELECT guild_id, channel_id, summary_text, source_user_messages, last_message_id, updated_at
                    FROM dialogue_summaries
                    WHERE user_id = $1
                      AND guild_id <> $2
                    ORDER BY updated_at DESC, source_user_messages DESC, last_message_id DESC
                    LIMIT 1
                    """,
                    user_id,
                    exclude_guild_id,
                )
            else:
                row = await conn.fetchrow(
                    """
                    SELECT guild_id, channel_id, summary_text, source_user_messages, last_message_id, updated_at
                    FROM dialogue_summaries
                    WHERE user_id = $1
                    ORDER BY updated_at DESC, source_user_messages DESC, last_message_id DESC
                    LIMIT 1
                    """,
                    user_id,
                )
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
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT subject_kind, subject_id, summary_text, source_fact_count, source_summary_count,
                       source_updated_at, last_source_guild_id, created_at, updated_at
                FROM global_biography_summaries
                WHERE subject_kind = $1 AND subject_id = $2
                """,
                kind,
                subject,
            )
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
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO global_biography_summaries (
                    subject_kind, subject_id, summary_text, source_fact_count, source_summary_count,
                    source_updated_at, last_source_guild_id, created_at, updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6::timestamptz, $7, NOW(), NOW())
                ON CONFLICT(subject_kind, subject_id) DO UPDATE SET
                    summary_text = EXCLUDED.summary_text,
                    source_fact_count = EXCLUDED.source_fact_count,
                    source_summary_count = EXCLUDED.source_summary_count,
                    source_updated_at = EXCLUDED.source_updated_at,
                    last_source_guild_id = EXCLUDED.last_source_guild_id,
                    updated_at = NOW()
                """,
                kind,
                subject,
                summary,
                max(0, int(source_fact_count)),
                max(0, int(source_summary_count)),
                str(source_updated_at).strip() if source_updated_at else None,
                str(last_source_guild_id).strip() if last_source_guild_id else None,
            )

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

    async def record_memory_extractor_run(
        self,
        *,
        guild_id: str,
        channel_id: str,
        speaker_user_id: str,
        fact_owner_kind: str,
        fact_owner_id: str,
        speaker_role: str,
        modality: str,
        source: str,
        backend_name: str,
        model_name: str,
        dry_run: bool,
        llm_attempted: bool,
        llm_ok: bool,
        json_valid: bool,
        fallback_used: bool,
        latency_ms: int,
        candidate_count: int,
        accepted_count: int,
        saved_count: int,
        filtered_count: int,
        error_text: str,
        candidates: list[dict[str, object]] | None = None,
    ) -> int:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                run_id = await conn.fetchval(
                    """
                    INSERT INTO memory_extractor_runs (
                        guild_id, channel_id, speaker_user_id, fact_owner_kind, fact_owner_id, speaker_role,
                        modality, source, backend_name, model_name, dry_run, llm_attempted, llm_ok, json_valid,
                        fallback_used, latency_ms, candidate_count, accepted_count, saved_count, filtered_count, error_text,
                        created_at
                    )
                    VALUES (
                        $1, $2, $3, $4, $5, $6,
                        $7, $8, $9, $10, $11, $12, $13, $14,
                        $15, $16, $17, $18, $19, $20, $21,
                        NOW()
                    )
                    RETURNING run_id
                    """,
                    str(guild_id or ""),
                    str(channel_id or ""),
                    str(speaker_user_id or ""),
                    str(fact_owner_kind or "user"),
                    str(fact_owner_id or ""),
                    str(speaker_role or "user"),
                    str(modality or "text"),
                    str(source or "unknown"),
                    str(backend_name or ""),
                    str(model_name or ""),
                    bool(dry_run),
                    bool(llm_attempted),
                    bool(llm_ok),
                    bool(json_valid),
                    bool(fallback_used),
                    max(0, int(latency_ms)),
                    max(0, int(candidate_count)),
                    max(0, int(accepted_count)),
                    max(0, int(saved_count)),
                    max(0, int(filtered_count)),
                    str(error_text or "")[:400],
                )
                run_id = int(run_id or 0)
                for row in list(candidates or []):
                    if not isinstance(row, dict):
                        continue
                    await conn.execute(
                        """
                        INSERT INTO memory_extractor_candidates (
                            run_id, fact_key, fact_value, fact_type, about_target, directness, evidence_quote,
                            confidence, importance, moderation_action, moderation_reason, selected_for_apply, saved_to_memory,
                            created_at
                        )
                        VALUES (
                            $1, $2, $3, $4, $5, $6, $7,
                            $8, $9, $10, $11, $12, $13,
                            NOW()
                        )
                        """,
                        run_id,
                        str(row.get("fact_key") or "")[:140],
                        str(row.get("fact_value") or "")[:280],
                        str(row.get("fact_type") or "fact")[:80],
                        str(row.get("about_target") or "self")[:32],
                        str(row.get("directness") or "explicit")[:32],
                        str(row.get("evidence_quote") or "")[:220],
                        _clamp(float(row.get("confidence") or 0.0), 0.0, 1.0),
                        _clamp(float(row.get("importance") or 0.0), 0.0, 1.0),
                        str(row.get("moderation_action") or "unknown")[:64],
                        str(row.get("moderation_reason") or "")[:200],
                        bool(row.get("selected_for_apply")),
                        bool(row.get("saved_to_memory")),
                    )
                return run_id

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
        *,
        about_target: str = "self",
        directness: str = "explicit",
        evidence_quote: str = "",
    ) -> int:
        key = fact_key.strip().casefold()
        value = fact_value.strip()
        if not key or not value:
            return 0

        fact_type_clean = fact_type.strip().lower() or "fact"
        conf = _clamp(float(confidence), 0.0, 1.0)
        imp = _clamp(float(importance), 0.0, 1.0)
        about_target_clean = normalize_memory_fact_about_target(
            about_target,
            default="assistant_self" if str(user_id).startswith("persona::") else "self",
        )
        directness_clean = normalize_memory_fact_directness(directness, default="explicit")
        evidence_quote_clean = sanitize_memory_fact_evidence_quote(evidence_quote)

        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    SELECT fact_id, fact_value, fact_type, confidence, importance, evidence_count, status, pinned,
                           COALESCE(about_target, 'self') AS about_target,
                           COALESCE(directness, 'explicit') AS directness,
                           COALESCE(evidence_quote, '') AS evidence_quote
                    FROM user_facts
                    WHERE guild_id = $1 AND user_id = $2 AND fact_key = $3
                    """,
                    guild_id,
                    user_id,
                    key,
                )

                if row is None:
                    status = apply_memory_fact_promotion_policy(
                        current_status="confirmed" if conf >= 0.78 else "candidate",
                        prior_status="candidate",
                        pinned=False,
                        confidence=conf,
                        importance=imp,
                        evidence_count=1,
                        directness=directness_clean,
                        value_conflict=False,
                        value_replaced=False,
                    )
                    fact_id = await conn.fetchval(
                        """
                        INSERT INTO user_facts (
                            guild_id, user_id, fact_key, fact_value, fact_type,
                            about_target, directness, evidence_quote,
                            confidence, importance, status, pinned, evidence_count,
                            created_at, updated_at, last_seen_at
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, FALSE, 1, NOW(), NOW(), NOW())
                        RETURNING fact_id
                        """,
                        guild_id,
                        user_id,
                        key,
                        value[:280],
                        fact_type_clean,
                        about_target_clean,
                        directness_clean,
                        evidence_quote_clean,
                        conf,
                        imp,
                        status,
                    )
                    fact_id = int(fact_id)
                else:
                    fact_id = int(row["fact_id"])
                    merge = merge_memory_fact_state(
                        prior_value=str(row["fact_value"] or ""),
                        prior_fact_type=str(row["fact_type"] or "fact"),
                        prior_confidence=float(row["confidence"]),
                        prior_importance=float(row["importance"]),
                        prior_evidence_count=int(row["evidence_count"]),
                        prior_status=str(row["status"]),
                        pinned=bool(row["pinned"]),
                        incoming_value=value,
                        incoming_fact_type=fact_type_clean,
                        incoming_confidence=conf,
                        incoming_importance=imp,
                    )
                    if merge.value_conflict:
                        logger.debug(
                            "Memory fact conflict (postgres local): guild=%s user=%s key=%s replaced=%s prior_conf=%.3f incoming_conf=%.3f",
                            guild_id,
                            user_id,
                            key,
                            merge.value_replaced,
                            float(row["confidence"]),
                            conf,
                        )

                    meta = merge_memory_fact_metadata_state(
                        prior_value=str(row["fact_value"] or ""),
                        incoming_value=value,
                        prior_about_target=str(row["about_target"] or "self"),
                        incoming_about_target=about_target_clean,
                        prior_directness=str(row["directness"] or "explicit"),
                        incoming_directness=directness_clean,
                        prior_evidence_quote=str(row["evidence_quote"] or ""),
                        incoming_evidence_quote=evidence_quote_clean,
                        value_conflict=merge.value_conflict,
                        value_replaced=merge.value_replaced,
                    )
                    next_status = apply_memory_fact_promotion_policy(
                        current_status=merge.status,
                        prior_status=str(row["status"] or "candidate"),
                        pinned=bool(row["pinned"]),
                        confidence=merge.confidence,
                        importance=merge.importance,
                        evidence_count=merge.evidence_count,
                        directness=meta.directness,
                        value_conflict=merge.value_conflict,
                        value_replaced=merge.value_replaced,
                    )

                    await conn.execute(
                        """
                        UPDATE user_facts
                        SET fact_value = $1,
                            fact_type = $2,
                            about_target = $3,
                            directness = $4,
                            evidence_quote = $5,
                            confidence = $6,
                            importance = $7,
                            status = $8,
                            evidence_count = $9,
                            updated_at = NOW(),
                            last_seen_at = NOW()
                        WHERE fact_id = $10
                        """,
                        merge.fact_value,
                        merge.fact_type,
                        meta.about_target,
                        meta.directness,
                        meta.evidence_quote,
                        merge.confidence,
                        merge.importance,
                        next_status,
                        merge.evidence_count,
                        fact_id,
                    )

                await conn.execute(
                    """
                    INSERT INTO fact_evidence (
                        fact_id, message_id, extractor, about_target, directness, evidence_quote, confidence, created_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                    """,
                    fact_id,
                    message_id,
                    extractor,
                    about_target_clean,
                    directness_clean,
                    evidence_quote_clean,
                    conf,
                )

                global_row = await conn.fetchrow(
                    """
                    SELECT global_fact_id, fact_value, fact_type, confidence, importance, evidence_count, status, pinned,
                           COALESCE(about_target, 'self') AS about_target,
                           COALESCE(directness, 'explicit') AS directness,
                           COALESCE(evidence_quote, '') AS evidence_quote
                    FROM global_user_facts
                    WHERE user_id = $1 AND fact_key = $2
                    """,
                    user_id,
                    key,
                )
                if global_row is None:
                    global_status = apply_memory_fact_promotion_policy(
                        current_status="confirmed" if conf >= 0.78 else "candidate",
                        prior_status="candidate",
                        pinned=False,
                        confidence=conf,
                        importance=imp,
                        evidence_count=1,
                        directness=directness_clean,
                        value_conflict=False,
                        value_replaced=False,
                    )
                    global_fact_id = await conn.fetchval(
                        """
                        INSERT INTO global_user_facts (
                            user_id, fact_key, fact_value, fact_type,
                            about_target, directness, evidence_quote,
                            confidence, importance, status, pinned, evidence_count,
                            first_seen_at, updated_at, last_seen_at,
                            last_source_guild_id, last_source_message_id
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, FALSE, 1, NOW(), NOW(), NOW(), $11, $12)
                        RETURNING global_fact_id
                        """,
                        user_id,
                        key,
                        value[:280],
                        fact_type_clean,
                        about_target_clean,
                        directness_clean,
                        evidence_quote_clean,
                        conf,
                        imp,
                        global_status,
                        guild_id,
                        message_id,
                    )
                    global_fact_id = int(global_fact_id)
                else:
                    global_fact_id = int(global_row["global_fact_id"])
                    g_merge = merge_memory_fact_state(
                        prior_value=str(global_row["fact_value"] or ""),
                        prior_fact_type=str(global_row["fact_type"] or "fact"),
                        prior_confidence=float(global_row["confidence"]),
                        prior_importance=float(global_row["importance"]),
                        prior_evidence_count=int(global_row["evidence_count"]),
                        prior_status=str(global_row["status"]),
                        pinned=bool(global_row["pinned"]),
                        incoming_value=value,
                        incoming_fact_type=fact_type_clean,
                        incoming_confidence=conf,
                        incoming_importance=imp,
                    )
                    if g_merge.value_conflict:
                        logger.debug(
                            "Memory fact conflict (postgres global): user=%s key=%s source_guild=%s replaced=%s prior_conf=%.3f incoming_conf=%.3f",
                            user_id,
                            key,
                            guild_id,
                            g_merge.value_replaced,
                            float(global_row["confidence"]),
                            conf,
                        )

                    g_meta = merge_memory_fact_metadata_state(
                        prior_value=str(global_row["fact_value"] or ""),
                        incoming_value=value,
                        prior_about_target=str(global_row["about_target"] or "self"),
                        incoming_about_target=about_target_clean,
                        prior_directness=str(global_row["directness"] or "explicit"),
                        incoming_directness=directness_clean,
                        prior_evidence_quote=str(global_row["evidence_quote"] or ""),
                        incoming_evidence_quote=evidence_quote_clean,
                        value_conflict=g_merge.value_conflict,
                        value_replaced=g_merge.value_replaced,
                    )
                    g_status = apply_memory_fact_promotion_policy(
                        current_status=g_merge.status,
                        prior_status=str(global_row["status"] or "candidate"),
                        pinned=bool(global_row["pinned"]),
                        confidence=g_merge.confidence,
                        importance=g_merge.importance,
                        evidence_count=g_merge.evidence_count,
                        directness=g_meta.directness,
                        value_conflict=g_merge.value_conflict,
                        value_replaced=g_merge.value_replaced,
                    )

                    await conn.execute(
                        """
                        UPDATE global_user_facts
                        SET fact_value = $1,
                            fact_type = $2,
                            about_target = $3,
                            directness = $4,
                            evidence_quote = $5,
                            confidence = $6,
                            importance = $7,
                            status = $8,
                            evidence_count = $9,
                            updated_at = NOW(),
                            last_seen_at = NOW(),
                            last_source_guild_id = $10,
                            last_source_message_id = $11
                        WHERE global_fact_id = $12
                        """,
                        g_merge.fact_value,
                        g_merge.fact_type,
                        g_meta.about_target,
                        g_meta.directness,
                        g_meta.evidence_quote,
                        g_merge.confidence,
                        g_merge.importance,
                        g_status,
                        g_merge.evidence_count,
                        guild_id,
                        message_id,
                        global_fact_id,
                    )

                await conn.execute(
                    """
                    INSERT INTO global_fact_evidence (
                        global_fact_id, source_guild_id, source_message_id, extractor,
                        about_target, directness, evidence_quote, confidence, created_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                    """,
                    global_fact_id,
                    guild_id,
                    message_id,
                    extractor,
                    about_target_clean,
                    directness_clean,
                    evidence_quote_clean,
                    conf,
                )

                return fact_id

    async def get_user_facts(self, guild_id: str, user_id: str, limit: int) -> List[Dict[str, object]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT fact_id, fact_key, fact_value, fact_type, confidence, importance,
                       COALESCE(about_target, 'self') AS about_target,
                       COALESCE(directness, 'explicit') AS directness,
                       COALESCE(evidence_quote, '') AS evidence_quote,
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
                "about_target": str(row["about_target"] or "self"),
                "directness": str(row["directness"] or "explicit"),
                "evidence_quote": str(row["evidence_quote"] or ""),
                "status": str(row["status"]),
                "pinned": bool(row["pinned"]),
                "evidence_count": int(row["evidence_count"]),
                "updated_at": str(row["updated_at"]),
                "last_seen_at": str(row["last_seen_at"]),
            }
            for row in rows
        ]

    async def get_user_facts_global_by_user_id(
        self,
        user_id: str,
        limit: int,
        *,
        exclude_guild_id: str | None = None,
    ) -> List[Dict[str, object]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            if exclude_guild_id:
                rows = await conn.fetch(
                    """
                    SELECT guf.global_fact_id, guf.user_id, guf.fact_key, guf.fact_value, guf.fact_type, guf.confidence, guf.importance,
                           COALESCE(guf.about_target, 'self') AS about_target,
                           COALESCE(guf.directness, 'explicit') AS directness,
                           COALESCE(guf.evidence_quote, '') AS evidence_quote,
                           guf.status, guf.pinned, guf.evidence_count, guf.updated_at, guf.last_seen_at, guf.last_source_guild_id
                    FROM global_user_facts AS guf
                    WHERE guf.user_id = $1
                      AND (
                            guf.last_source_guild_id IS NULL
                         OR guf.last_source_guild_id <> $2
                         OR EXISTS (
                                SELECT 1
                                FROM global_fact_evidence AS gfe
                                WHERE gfe.global_fact_id = guf.global_fact_id
                                  AND gfe.source_guild_id IS NOT NULL
                                  AND gfe.source_guild_id <> $2
                            )
                      )
                    ORDER BY
                        guf.pinned DESC,
                        CASE guf.status
                            WHEN 'pinned' THEN 3
                            WHEN 'confirmed' THEN 2
                            ELSE 1
                        END DESC,
                        guf.importance DESC,
                        guf.confidence DESC,
                        guf.evidence_count DESC,
                        guf.updated_at DESC
                    LIMIT $3
                    """,
                    user_id,
                    exclude_guild_id,
                    max(1, int(limit)),
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT global_fact_id, user_id, fact_key, fact_value, fact_type, confidence, importance,
                           COALESCE(about_target, 'self') AS about_target,
                           COALESCE(directness, 'explicit') AS directness,
                           COALESCE(evidence_quote, '') AS evidence_quote,
                           status, pinned, evidence_count, updated_at, last_seen_at, last_source_guild_id
                    FROM global_user_facts
                    WHERE user_id = $1
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
                    LIMIT $2
                    """,
                    user_id,
                    max(1, int(limit)),
                )

        if rows:
            return [
                {
                    "fact_id": int(row["global_fact_id"]),
                    "guild_id": str(row["last_source_guild_id"] or ""),
                    "fact_key": str(row["fact_key"]),
                    "fact_value": str(row["fact_value"]),
                    "fact_type": str(row["fact_type"]),
                    "confidence": float(row["confidence"]),
                    "importance": float(row["importance"]),
                    "about_target": str(row["about_target"] or "self"),
                    "directness": str(row["directness"] or "explicit"),
                    "evidence_quote": str(row["evidence_quote"] or ""),
                    "status": str(row["status"]),
                    "pinned": bool(row["pinned"]),
                    "evidence_count": int(row["evidence_count"]),
                    "updated_at": str(row["updated_at"]),
                    "last_seen_at": str(row["last_seen_at"]),
                    "global": True,
                }
                for row in rows
            ]

        # Fallback to legacy per-guild aggregation if global tables are empty (migration warmup).
        async with pool.acquire() as conn:
            if exclude_guild_id:
                rows = await conn.fetch(
                    """
                    SELECT fact_id, guild_id, fact_key, fact_value, fact_type, confidence, importance,
                           COALESCE(about_target, 'self') AS about_target,
                           COALESCE(directness, 'explicit') AS directness,
                           COALESCE(evidence_quote, '') AS evidence_quote,
                           status, pinned, evidence_count, updated_at, last_seen_at
                    FROM user_facts
                    WHERE user_id = $1
                      AND guild_id <> $2
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
                    user_id,
                    exclude_guild_id,
                    max(1, int(limit) * 4),
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT fact_id, guild_id, fact_key, fact_value, fact_type, confidence, importance,
                           COALESCE(about_target, 'self') AS about_target,
                           COALESCE(directness, 'explicit') AS directness,
                           COALESCE(evidence_quote, '') AS evidence_quote,
                           status, pinned, evidence_count, updated_at, last_seen_at
                    FROM user_facts
                    WHERE user_id = $1
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
                    LIMIT $2
                    """,
                    user_id,
                    max(1, int(limit) * 4),
                )
        deduped: list[Dict[str, object]] = []
        seen_keys: set[str] = set()
        for row in rows:
            fact_key = str(row["fact_key"])
            if not fact_key or fact_key in seen_keys:
                continue
            seen_keys.add(fact_key)
            deduped.append(
                {
                    "fact_id": int(row["fact_id"]),
                    "guild_id": str(row["guild_id"]),
                    "fact_key": fact_key,
                    "fact_value": str(row["fact_value"]),
                    "fact_type": str(row["fact_type"]),
                    "confidence": float(row["confidence"]),
                    "importance": float(row["importance"]),
                    "about_target": str(row["about_target"] or "self"),
                    "directness": str(row["directness"] or "explicit"),
                    "evidence_quote": str(row["evidence_quote"] or ""),
                    "status": str(row["status"]),
                    "pinned": bool(row["pinned"]),
                    "evidence_count": int(row["evidence_count"]),
                    "updated_at": str(row["updated_at"]),
                    "last_seen_at": str(row["last_seen_at"]),
                }
            )
            if len(deduped) >= max(1, int(limit)):
                break
        return deduped

    async def ensure_persona_mvp_bootstrap(
        self,
        persona_id: str,
        core_dna_hash: str,
        core_dna_source: str,
        policy_version: int,
        trait_catalog_entries: list[dict[str, object]],
    ) -> None:
        persona_key = str(persona_id or "").strip()
        if not persona_key:
            raise ValueError("persona_id cannot be empty")

        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO persona_global_state (
                        persona_id, core_dna_hash, core_dna_source, policy_version, overlay_summary, updated_at
                    )
                    VALUES ($1, $2, $3, $4, '', NOW())
                    ON CONFLICT(persona_id) DO UPDATE SET
                        core_dna_hash = EXCLUDED.core_dna_hash,
                        core_dna_source = EXCLUDED.core_dna_source,
                        policy_version = EXCLUDED.policy_version,
                        updated_at = NOW()
                    """,
                    persona_key,
                    str(core_dna_hash or "")[:128],
                    str(core_dna_source or "")[:400],
                    max(1, int(policy_version)),
                )

                for raw in trait_catalog_entries:
                    if not isinstance(raw, dict):
                        continue
                    trait_key = str(raw.get("trait_key") or "").strip().lower()
                    if not trait_key:
                        continue
                    label = str(raw.get("label") or trait_key).strip()[:120]
                    description = str(raw.get("description") or "").strip()[:500]
                    notes = str(raw.get("notes") or "").strip()[:600]

                    def _f(name: str, default: float) -> float:
                        try:
                            return float(raw.get(name, default))
                        except (TypeError, ValueError):
                            return float(default)

                    min_value = _f("min_value", 0.0)
                    max_value = _f("max_value", 1.0)
                    if max_value < min_value:
                        min_value, max_value = max_value, min_value
                    default_anchor = _clamp(_f("default_anchor_value", 0.5), min_value, max_value)
                    max_abs_drift = max(0.0, _f("max_abs_drift", 0.2))
                    max_step = max(0.0, _f("max_step_per_reflection", 0.03))
                    plasticity = _clamp(_f("plasticity", 0.5), 0.0, 1.0)
                    protected_mode = str(raw.get("protected_mode") or "soft").strip().lower()
                    if protected_mode not in {"locked", "bounded", "soft"}:
                        protected_mode = "soft"
                    prompt_exposure = str(raw.get("prompt_exposure") or "relevant").strip().lower()
                    if prompt_exposure not in {"always", "relevant", "never"}:
                        prompt_exposure = "relevant"
                    anchor_source = str(raw.get("anchor_source") or "neutral").strip().lower()
                    enabled = bool(raw.get("enabled", True))

                    await conn.execute(
                        """
                        INSERT INTO persona_trait_catalog (
                            trait_key, label, description, default_anchor_value,
                            min_value, max_value, max_abs_drift, max_step_per_reflection, plasticity,
                            protected_mode, prompt_exposure, anchor_source, notes, enabled, updated_at
                        )
                        VALUES (
                            $1, $2, $3, $4,
                            $5, $6, $7, $8, $9,
                            $10, $11, $12, $13, $14, NOW()
                        )
                        ON CONFLICT(trait_key) DO UPDATE SET
                            label = EXCLUDED.label,
                            description = EXCLUDED.description,
                            default_anchor_value = EXCLUDED.default_anchor_value,
                            min_value = EXCLUDED.min_value,
                            max_value = EXCLUDED.max_value,
                            max_abs_drift = EXCLUDED.max_abs_drift,
                            max_step_per_reflection = EXCLUDED.max_step_per_reflection,
                            plasticity = EXCLUDED.plasticity,
                            protected_mode = EXCLUDED.protected_mode,
                            prompt_exposure = EXCLUDED.prompt_exposure,
                            anchor_source = EXCLUDED.anchor_source,
                            notes = EXCLUDED.notes,
                            enabled = EXCLUDED.enabled,
                            updated_at = NOW()
                        """,
                        trait_key,
                        label,
                        description,
                        default_anchor,
                        min_value,
                        max_value,
                        max_abs_drift,
                        max_step,
                        plasticity,
                        protected_mode,
                        prompt_exposure,
                        anchor_source,
                        notes,
                        enabled,
                    )

                await conn.execute(
                    """
                    INSERT INTO persona_traits (
                        persona_id, trait_key, anchor_value, current_value, confidence, status, prompt_exposure,
                        created_at, updated_at
                    )
                    SELECT
                        $1 AS persona_id,
                        c.trait_key,
                        c.default_anchor_value AS anchor_value,
                        c.default_anchor_value AS current_value,
                        CASE WHEN c.protected_mode = 'locked' THEN 1.0 ELSE 0.35 END AS confidence,
                        CASE WHEN c.protected_mode = 'locked' THEN 'frozen' ELSE 'emerging' END AS status,
                        c.prompt_exposure,
                        NOW(),
                        NOW()
                    FROM persona_trait_catalog c
                    WHERE c.enabled = TRUE
                    ON CONFLICT(persona_id, trait_key) DO NOTHING
                    """,
                    persona_key,
                )

    async def get_persona_global_state(self, persona_id: str) -> Optional[Dict[str, object]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT persona_id, core_dna_hash, core_dna_source, policy_version, current_era_id,
                       reflection_cursor_message_id, last_reflection_at, last_decay_at, overlay_summary,
                       total_messages_seen, eligible_messages_seen, unique_users_seen, created_at, updated_at
                FROM persona_global_state
                WHERE persona_id = $1
                """,
                persona_id,
            )
        if row is None:
            return None

        return {
            "persona_id": str(row["persona_id"]),
            "core_dna_hash": str(row["core_dna_hash"]),
            "core_dna_source": str(row["core_dna_source"]),
            "policy_version": int(row["policy_version"]),
            "current_era_id": int(row["current_era_id"]) if row["current_era_id"] is not None else None,
            "reflection_cursor_message_id": int(row["reflection_cursor_message_id"] or 0),
            "last_reflection_at": str(row["last_reflection_at"]) if row["last_reflection_at"] is not None else "",
            "last_decay_at": str(row["last_decay_at"]) if row["last_decay_at"] is not None else "",
            "overlay_summary": str(row["overlay_summary"] or ""),
            "total_messages_seen": int(row["total_messages_seen"] or 0),
            "eligible_messages_seen": int(row["eligible_messages_seen"] or 0),
            "unique_users_seen": int(row["unique_users_seen"] or 0),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    async def list_persona_traits(self, persona_id: str, limit: int = 12) -> List[Dict[str, object]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    t.persona_id,
                    t.trait_key,
                    t.anchor_value,
                    t.current_value,
                    t.confidence,
                    t.drift_velocity,
                    t.evidence_count,
                    t.support_score,
                    t.contradiction_score,
                    t.status,
                    t.prompt_exposure,
                    t.last_changed_at,
                    t.last_reconfirmed_at,
                    t.updated_at,
                    c.label,
                    c.description,
                    c.min_value,
                    c.max_value,
                    c.max_abs_drift,
                    c.max_step_per_reflection,
                    c.plasticity,
                    c.protected_mode,
                    c.anchor_source,
                    c.notes
                FROM persona_traits t
                JOIN persona_trait_catalog c ON c.trait_key = t.trait_key
                WHERE t.persona_id = $1
                ORDER BY
                    CASE t.status
                        WHEN 'frozen' THEN 4
                        WHEN 'stable' THEN 3
                        WHEN 'contested' THEN 2
                        ELSE 1
                    END DESC,
                    t.confidence DESC,
                    ABS(t.current_value - t.anchor_value) DESC,
                    t.updated_at DESC
                LIMIT $2
                """,
                persona_id,
                max(1, int(limit)),
            )
        return [
            {
                "persona_id": str(row["persona_id"]),
                "trait_key": str(row["trait_key"]),
                "label": str(row["label"]),
                "description": str(row["description"]),
                "anchor_value": float(row["anchor_value"]),
                "current_value": float(row["current_value"]),
                "confidence": float(row["confidence"]),
                "drift_velocity": float(row["drift_velocity"]),
                "evidence_count": int(row["evidence_count"]),
                "support_score": float(row["support_score"]),
                "contradiction_score": float(row["contradiction_score"]),
                "status": str(row["status"]),
                "prompt_exposure": str(row["prompt_exposure"]),
                "min_value": float(row["min_value"]),
                "max_value": float(row["max_value"]),
                "max_abs_drift": float(row["max_abs_drift"]),
                "max_step_per_reflection": float(row["max_step_per_reflection"]),
                "plasticity": float(row["plasticity"]),
                "protected_mode": str(row["protected_mode"]),
                "anchor_source": str(row["anchor_source"]),
                "notes": str(row["notes"] or ""),
                "last_changed_at": str(row["last_changed_at"]) if row["last_changed_at"] is not None else "",
                "last_reconfirmed_at": str(row["last_reconfirmed_at"]) if row["last_reconfirmed_at"] is not None else "",
                "updated_at": str(row["updated_at"]),
            }
            for row in rows
        ]

    async def get_persona_relationship(
        self,
        persona_id: str,
        guild_id: str,
        user_id: str,
    ) -> Optional[Dict[str, object]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT persona_id, guild_id, user_id, user_label_cache, status, consent_scope,
                       familiarity, trust, warmth, banter_license, support_sensitivity,
                       boundary_sensitivity, topic_alignment, confidence, interaction_count,
                       voice_turn_count, text_turn_count, effective_influence_weight,
                       relationship_summary, inside_joke_summary, preferred_style_notes,
                       risk_flags, last_interaction_at, last_reflection_at, created_at, updated_at
                FROM persona_relationships
                WHERE persona_id = $1 AND guild_id = $2 AND user_id = $3
                """,
                persona_id,
                guild_id,
                user_id,
            )
        if row is None:
            return None
        risk_flags_raw = row["risk_flags"]
        if isinstance(risk_flags_raw, dict):
            risk_flags = dict(risk_flags_raw)
        else:
            try:
                parsed = json.loads(str(risk_flags_raw or "{}"))
            except Exception:
                parsed = {}
            risk_flags = parsed if isinstance(parsed, dict) else {}
        return {
            "persona_id": str(row["persona_id"]),
            "guild_id": str(row["guild_id"]),
            "user_id": str(row["user_id"]),
            "user_label_cache": str(row["user_label_cache"] or ""),
            "status": str(row["status"]),
            "consent_scope": str(row["consent_scope"]),
            "familiarity": float(row["familiarity"]),
            "trust": float(row["trust"]),
            "warmth": float(row["warmth"]),
            "banter_license": float(row["banter_license"]),
            "support_sensitivity": float(row["support_sensitivity"]),
            "boundary_sensitivity": float(row["boundary_sensitivity"]),
            "topic_alignment": float(row["topic_alignment"]),
            "confidence": float(row["confidence"]),
            "interaction_count": int(row["interaction_count"]),
            "voice_turn_count": int(row["voice_turn_count"]),
            "text_turn_count": int(row["text_turn_count"]),
            "effective_influence_weight": float(row["effective_influence_weight"]),
            "relationship_summary": str(row["relationship_summary"] or ""),
            "inside_joke_summary": str(row["inside_joke_summary"] or ""),
            "preferred_style_notes": str(row["preferred_style_notes"] or ""),
            "risk_flags": risk_flags,
            "last_interaction_at": str(row["last_interaction_at"]) if row["last_interaction_at"] is not None else "",
            "last_reflection_at": str(row["last_reflection_at"]) if row["last_reflection_at"] is not None else "",
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    async def touch_persona_relationship(
        self,
        persona_id: str,
        guild_id: str,
        user_id: str,
        *,
        user_label: str = "",
        modality: str = "text",
    ) -> None:
        mode = str(modality or "text").strip().lower()
        is_voice = mode == "voice"
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            pref_mode_raw = await conn.fetchval(
                """
                SELECT memory_mode
                FROM persona_user_memory_prefs
                WHERE guild_id = $1 AND user_id = $2
                """,
                guild_id,
                user_id,
            )
            pref_mode = str(pref_mode_raw or "full").strip().lower()
            if pref_mode == "none":
                return
            consent_scope = "minimal" if pref_mode == "minimal" else "full"
            await conn.execute(
                """
                INSERT INTO persona_relationships (
                    persona_id, guild_id, user_id, user_label_cache, status, consent_scope,
                    interaction_count, voice_turn_count, text_turn_count, last_interaction_at, updated_at
                )
                VALUES (
                    $1, $2, $3, $4, 'active', $7,
                    1, $5, $6, NOW(), NOW()
                )
                ON CONFLICT(persona_id, guild_id, user_id) DO UPDATE SET
                    user_label_cache = CASE
                        WHEN EXCLUDED.user_label_cache <> '' THEN EXCLUDED.user_label_cache
                        ELSE persona_relationships.user_label_cache
                    END,
                    interaction_count = persona_relationships.interaction_count + 1,
                    voice_turn_count = persona_relationships.voice_turn_count + EXCLUDED.voice_turn_count,
                    text_turn_count = persona_relationships.text_turn_count + EXCLUDED.text_turn_count,
                    consent_scope = EXCLUDED.consent_scope,
                    familiarity = LEAST(1.0, persona_relationships.familiarity + 0.01),
                    confidence = LEAST(1.0, persona_relationships.confidence + 0.005),
                    last_interaction_at = NOW(),
                    updated_at = NOW()
                """,
                persona_id,
                guild_id,
                user_id,
                str(user_label or "")[:160],
                1 if is_voice else 0,
                0 if is_voice else 1,
                consent_scope,
            )
            await conn.execute(
                """
                UPDATE persona_global_state
                SET total_messages_seen = total_messages_seen + 1,
                    unique_users_seen = (
                        SELECT COUNT(*)
                        FROM persona_relationships pr
                        WHERE pr.persona_id = $1
                    ),
                    updated_at = NOW()
                WHERE persona_id = $1
                """,
                persona_id,
            )

    async def get_persona_user_memory_pref(self, guild_id: str, user_id: str) -> Optional[Dict[str, object]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT guild_id, user_id, memory_mode, allow_episodic_callbacks,
                       allow_personality_influence, allow_sensitive_storage,
                       retention_days, updated_by_admin_user_id, updated_at
                FROM persona_user_memory_prefs
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
            "memory_mode": str(row["memory_mode"]),
            "allow_episodic_callbacks": bool(row["allow_episodic_callbacks"]),
            "allow_personality_influence": bool(row["allow_personality_influence"]),
            "allow_sensitive_storage": bool(row["allow_sensitive_storage"]),
            "retention_days": int(row["retention_days"]) if row["retention_days"] is not None else None,
            "updated_by_admin_user_id": str(row["updated_by_admin_user_id"] or ""),
            "updated_at": str(row["updated_at"]),
        }

    async def set_persona_user_memory_pref(
        self,
        guild_id: str,
        user_id: str,
        *,
        memory_mode: str,
        allow_episodic_callbacks: bool = True,
        allow_personality_influence: bool = True,
        allow_sensitive_storage: bool = False,
        retention_days: int | None = None,
        updated_by_admin_user_id: str | None = None,
    ) -> None:
        mode = str(memory_mode or "full").strip().lower()
        if mode not in {"full", "minimal", "none"}:
            mode = "full"
        retention_value = None if retention_days is None else max(1, int(retention_days))
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO persona_user_memory_prefs (
                    guild_id, user_id, memory_mode, allow_episodic_callbacks,
                    allow_personality_influence, allow_sensitive_storage,
                    retention_days, updated_by_admin_user_id, updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                ON CONFLICT(guild_id, user_id) DO UPDATE SET
                    memory_mode = EXCLUDED.memory_mode,
                    allow_episodic_callbacks = EXCLUDED.allow_episodic_callbacks,
                    allow_personality_influence = EXCLUDED.allow_personality_influence,
                    allow_sensitive_storage = EXCLUDED.allow_sensitive_storage,
                    retention_days = EXCLUDED.retention_days,
                    updated_by_admin_user_id = EXCLUDED.updated_by_admin_user_id,
                    updated_at = NOW()
                """,
                guild_id,
                user_id,
                mode,
                bool(allow_episodic_callbacks),
                bool(allow_personality_influence),
                bool(allow_sensitive_storage),
                retention_value,
                (str(updated_by_admin_user_id).strip()[:60] if updated_by_admin_user_id else None),
            )
            consent_scope = "minimal" if mode == "minimal" else "full"
            status = "limited" if mode == "none" else ("limited" if mode == "minimal" else "active")
            await conn.execute(
                """
                UPDATE persona_relationships
                SET consent_scope = $1,
                    status = $2,
                    updated_at = NOW()
                WHERE guild_id = $3 AND user_id = $4
                """,
                consent_scope,
                status,
                guild_id,
                user_id,
            )

    async def create_persona_reflection(
        self,
        persona_id: str,
        *,
        trigger_type: str,
        status: str,
        dedupe_key: str | None = None,
        trigger_reason: str = "",
        window_start_message_id: int | None = None,
        window_end_message_id: int | None = None,
        input_counts: dict[str, object] | None = None,
        proposal_json: dict[str, object] | None = None,
        validator_report_json: dict[str, object] | None = None,
        model_name: str = "",
        prompt_version: str = "",
    ) -> int:
        dedupe = (str(dedupe_key).strip()[:200] if dedupe_key else None)
        payload_input = json.dumps(input_counts or {}, ensure_ascii=True)
        payload_proposal = json.dumps(proposal_json or {}, ensure_ascii=True) if proposal_json is not None else None
        payload_validator = (
            json.dumps(validator_report_json or {}, ensure_ascii=True) if validator_report_json is not None else None
        )
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO persona_reflections (
                    persona_id, dedupe_key, trigger_type, trigger_reason, status,
                    window_start_message_id, window_end_message_id,
                    input_counts, proposal_json, validator_report_json,
                    model_name, prompt_version, created_at
                )
                VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7,
                    $8::jsonb, $9::jsonb, $10::jsonb,
                    $11, $12, NOW()
                )
                ON CONFLICT DO NOTHING
                RETURNING reflection_id
                """,
                persona_id,
                dedupe,
                str(trigger_type or "scheduled")[:40],
                str(trigger_reason or "")[:240],
                str(status or "queued")[:40],
                (int(window_start_message_id) if window_start_message_id is not None else None),
                (int(window_end_message_id) if window_end_message_id is not None else None),
                payload_input,
                payload_proposal,
                payload_validator,
                str(model_name or "")[:120],
                str(prompt_version or "")[:80],
            )
            if row is not None:
                return int(row["reflection_id"])
            if dedupe:
                existing = await conn.fetchval(
                    "SELECT reflection_id FROM persona_reflections WHERE dedupe_key = $1 LIMIT 1",
                    dedupe,
                )
                if existing is not None:
                    return int(existing)
        return 0

    async def update_persona_reflection_status(
        self,
        reflection_id: int,
        *,
        status: str,
        rejection_reason: str = "",
        validator_report_json: dict[str, object] | None = None,
        proposal_json: dict[str, object] | None = None,
        applied_changes_json: dict[str, object] | None = None,
        duration_ms: int | None = None,
        model_name: str | None = None,
        prompt_version: str | None = None,
    ) -> None:
        payload_validator = (
            json.dumps(validator_report_json or {}, ensure_ascii=True) if validator_report_json is not None else None
        )
        payload_proposal = json.dumps(proposal_json or {}, ensure_ascii=True) if proposal_json is not None else None
        payload_applied = (
            json.dumps(applied_changes_json or {}, ensure_ascii=True) if applied_changes_json is not None else None
        )
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE persona_reflections
                SET status = $1,
                    rejection_reason = $2,
                    validator_report_json = COALESCE($3::jsonb, validator_report_json),
                    proposal_json = COALESCE($4::jsonb, proposal_json),
                    applied_changes_json = COALESCE($5::jsonb, applied_changes_json),
                    duration_ms = COALESCE($6, duration_ms),
                    model_name = COALESCE($8, model_name),
                    prompt_version = COALESCE($9, prompt_version),
                    finished_at = NOW(),
                    applied_at = CASE WHEN $1 = 'applied' THEN NOW() ELSE applied_at END
                WHERE reflection_id = $7
                """,
                str(status or "failed")[:40],
                str(rejection_reason or "")[:240],
                payload_validator,
                payload_proposal,
                payload_applied,
                (int(duration_ms) if duration_ms is not None else None),
                int(reflection_id),
                (str(model_name).strip()[:120] if model_name else None),
                (str(prompt_version).strip()[:80] if prompt_version else None),
            )

    async def list_persona_reflections(self, persona_id: str, limit: int = 10) -> List[Dict[str, object]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT reflection_id, persona_id, dedupe_key, trigger_type, trigger_reason, status,
                       window_start_message_id, window_end_message_id, input_counts,
                       rejection_reason, model_name, prompt_version, duration_ms,
                       created_at, started_at, finished_at, applied_at
                FROM persona_reflections
                WHERE persona_id = $1
                ORDER BY reflection_id DESC
                LIMIT $2
                """,
                persona_id,
                max(1, int(limit)),
            )
        out: List[Dict[str, object]] = []
        for row in rows:
            input_counts_raw = row["input_counts"]
            if isinstance(input_counts_raw, dict):
                input_counts = dict(input_counts_raw)
            else:
                try:
                    parsed_counts = json.loads(str(input_counts_raw or "{}"))
                except Exception:
                    parsed_counts = {}
                input_counts = parsed_counts if isinstance(parsed_counts, dict) else {}
            out.append(
                {
                    "reflection_id": int(row["reflection_id"]),
                    "persona_id": str(row["persona_id"]),
                    "dedupe_key": str(row["dedupe_key"] or ""),
                    "trigger_type": str(row["trigger_type"]),
                    "trigger_reason": str(row["trigger_reason"] or ""),
                    "status": str(row["status"]),
                    "window_start_message_id": (
                        int(row["window_start_message_id"]) if row["window_start_message_id"] is not None else None
                    ),
                    "window_end_message_id": (
                        int(row["window_end_message_id"]) if row["window_end_message_id"] is not None else None
                    ),
                    "input_counts": dict(input_counts),
                    "rejection_reason": str(row["rejection_reason"] or ""),
                    "model_name": str(row["model_name"] or ""),
                    "prompt_version": str(row["prompt_version"] or ""),
                    "duration_ms": int(row["duration_ms"]) if row["duration_ms"] is not None else None,
                    "created_at": str(row["created_at"]),
                    "started_at": str(row["started_at"]) if row["started_at"] is not None else "",
                    "finished_at": str(row["finished_at"]) if row["finished_at"] is not None else "",
                    "applied_at": str(row["applied_at"]) if row["applied_at"] is not None else "",
                }
            )
        return out

    @staticmethod
    def _coerce_json_object(value: object, default: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        if isinstance(value, dict):
            return dict(value)
        try:
            parsed = json.loads(str(value or "{}"))
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return dict(parsed)
        return dict(default or {})

    @staticmethod
    def _coerce_json_list(value: object, default: Optional[List[object]] = None) -> List[object]:
        if isinstance(value, list):
            return list(value)
        try:
            parsed = json.loads(str(value or "[]"))
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return list(parsed)
        return list(default or [])

    async def get_persona_reflection_details(
        self,
        persona_id: str,
        *,
        reflection_id: int | None = None,
    ) -> Optional[Dict[str, object]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            if reflection_id is None:
                row = await conn.fetchrow(
                    """
                    SELECT reflection_id, persona_id, dedupe_key, trigger_type, trigger_reason, status,
                           window_start_message_id, window_end_message_id,
                           input_counts, proposal_json, validator_report_json, applied_changes_json,
                           rejection_reason, model_name, prompt_version, duration_ms,
                           created_at, started_at, finished_at, applied_at
                    FROM persona_reflections
                    WHERE persona_id = $1
                    ORDER BY reflection_id DESC
                    LIMIT 1
                    """,
                    persona_id,
                )
            else:
                row = await conn.fetchrow(
                    """
                    SELECT reflection_id, persona_id, dedupe_key, trigger_type, trigger_reason, status,
                           window_start_message_id, window_end_message_id,
                           input_counts, proposal_json, validator_report_json, applied_changes_json,
                           rejection_reason, model_name, prompt_version, duration_ms,
                           created_at, started_at, finished_at, applied_at
                    FROM persona_reflections
                    WHERE persona_id = $1 AND reflection_id = $2
                    LIMIT 1
                    """,
                    persona_id,
                    int(reflection_id),
                )
        if row is None:
            return None
        return {
            "reflection_id": int(row["reflection_id"]),
            "persona_id": str(row["persona_id"]),
            "dedupe_key": str(row["dedupe_key"] or ""),
            "trigger_type": str(row["trigger_type"] or ""),
            "trigger_reason": str(row["trigger_reason"] or ""),
            "status": str(row["status"] or ""),
            "window_start_message_id": int(row["window_start_message_id"]) if row["window_start_message_id"] is not None else None,
            "window_end_message_id": int(row["window_end_message_id"]) if row["window_end_message_id"] is not None else None,
            "input_counts": self._coerce_json_object(row["input_counts"]),
            "proposal_json": self._coerce_json_object(row["proposal_json"]),
            "validator_report_json": self._coerce_json_object(row["validator_report_json"]),
            "applied_changes_json": self._coerce_json_object(row["applied_changes_json"]),
            "rejection_reason": str(row["rejection_reason"] or ""),
            "model_name": str(row["model_name"] or ""),
            "prompt_version": str(row["prompt_version"] or ""),
            "duration_ms": int(row["duration_ms"]) if row["duration_ms"] is not None else None,
            "created_at": str(row["created_at"]),
            "started_at": str(row["started_at"]) if row["started_at"] is not None else "",
            "finished_at": str(row["finished_at"]) if row["finished_at"] is not None else "",
            "applied_at": str(row["applied_at"]) if row["applied_at"] is not None else "",
        }

    async def list_persona_trait_evidence(
        self,
        persona_id: str,
        trait_key: str,
        *,
        limit: int = 10,
    ) -> List[Dict[str, object]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    te.trait_evidence_id,
                    te.persona_id,
                    te.trait_key,
                    te.reflection_id,
                    te.signal_kind,
                    te.direction,
                    te.magnitude,
                    te.signal_confidence,
                    te.quality_weight,
                    te.influence_weight,
                    te.reason_text,
                    te.evidence_refs,
                    te.created_at,
                    r.status AS reflection_status,
                    r.trigger_type AS reflection_trigger_type,
                    r.created_at AS reflection_created_at
                FROM persona_trait_evidence te
                LEFT JOIN persona_reflections r ON r.reflection_id = te.reflection_id
                WHERE te.persona_id = $1 AND te.trait_key = $2
                ORDER BY te.created_at DESC
                LIMIT $3
                """,
                persona_id,
                str(trait_key or "").strip(),
                max(1, min(int(limit), 30)),
            )
        return [
            {
                "trait_evidence_id": int(row["trait_evidence_id"]),
                "persona_id": str(row["persona_id"]),
                "trait_key": str(row["trait_key"]),
                "reflection_id": int(row["reflection_id"]) if row["reflection_id"] is not None else None,
                "signal_kind": str(row["signal_kind"] or ""),
                "direction": float(row["direction"] or 0.0),
                "magnitude": float(row["magnitude"] or 0.0),
                "signal_confidence": float(row["signal_confidence"] or 0.0),
                "quality_weight": float(row["quality_weight"] or 0.0),
                "influence_weight": float(row["influence_weight"] or 0.0),
                "reason_text": str(row["reason_text"] or ""),
                "evidence_refs": self._coerce_json_list(row["evidence_refs"]),
                "created_at": str(row["created_at"]),
                "reflection_status": str(row["reflection_status"] or ""),
                "reflection_trigger_type": str(row["reflection_trigger_type"] or ""),
                "reflection_created_at": str(row["reflection_created_at"]) if row["reflection_created_at"] is not None else "",
            }
            for row in rows
        ]

    async def apply_persona_reflection_changes(
        self,
        persona_id: str,
        reflection_id: int,
        *,
        accepted_trait_candidates: list[dict[str, object]] | None = None,
        contested_trait_candidates: list[dict[str, object]] | None = None,
        accepted_episode_promotions: list[dict[str, object]] | None = None,
        overlay_summary: str | None = None,
    ) -> Dict[str, object]:
        trait_candidates = [row for row in (accepted_trait_candidates or []) if isinstance(row, dict)]
        contested_candidates = [row for row in (contested_trait_candidates or []) if isinstance(row, dict)]
        episode_promotions = [row for row in (accepted_episode_promotions or []) if isinstance(row, dict)]
        payload_overlay = str(overlay_summary or "").strip()[:760]

        applied_traits: List[Dict[str, object]] = []
        skipped_traits: List[Dict[str, object]] = []
        applied_contested_traits: List[Dict[str, object]] = []
        skipped_contested_traits: List[Dict[str, object]] = []
        applied_episodes: List[Dict[str, object]] = []
        skipped_episodes: List[Dict[str, object]] = []

        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                reflection_status = await conn.fetchval(
                    """
                    SELECT status
                    FROM persona_reflections
                    WHERE reflection_id = $1 AND persona_id = $2
                    FOR UPDATE
                    """,
                    int(reflection_id),
                    persona_id,
                )
                if reflection_status is None:
                    return {
                        "ok": False,
                        "error": "reflection_not_found",
                        "reflection_id": int(reflection_id),
                    }
                if str(reflection_status) == "applied":
                    return {
                        "ok": False,
                        "error": "reflection_already_applied",
                        "reflection_id": int(reflection_id),
                    }

                for raw in trait_candidates:
                    trait_key = str(raw.get("trait_key") or "").strip()
                    if not trait_key:
                        skipped_traits.append({"reason": "missing_trait_key"})
                        continue
                    row = await conn.fetchrow(
                        """
                        SELECT
                            t.current_value,
                            t.anchor_value,
                            t.confidence,
                            t.evidence_count,
                            t.support_score,
                            t.contradiction_score,
                            t.status,
                            c.min_value,
                            c.max_value,
                            c.max_abs_drift,
                            c.max_step_per_reflection,
                            c.protected_mode
                        FROM persona_traits t
                        JOIN persona_trait_catalog c ON c.trait_key = t.trait_key
                        WHERE t.persona_id = $1 AND t.trait_key = $2
                        FOR UPDATE
                        """,
                        persona_id,
                        trait_key,
                    )
                    if row is None:
                        skipped_traits.append({"trait_key": trait_key, "reason": "trait_not_found"})
                        continue

                    protected_mode = str(row["protected_mode"] or "soft")
                    if protected_mode == "locked":
                        skipped_traits.append({"trait_key": trait_key, "reason": "locked_trait"})
                        continue

                    try:
                        delta = float(raw.get("delta", 0.0))
                        candidate_conf = float(raw.get("confidence", 0.0))
                    except (TypeError, ValueError):
                        skipped_traits.append({"trait_key": trait_key, "reason": "invalid_candidate_values"})
                        continue

                    max_step = abs(float(row["max_step_per_reflection"] or 0.03))
                    bounded_delta = _clamp(delta, -max_step, max_step)
                    current_value = float(row["current_value"] or 0.0)
                    anchor_value = float(row["anchor_value"] or 0.0)
                    min_value = float(row["min_value"] or 0.0)
                    max_value = float(row["max_value"] or 1.0)
                    max_abs_drift = abs(float(row["max_abs_drift"] or 0.0))

                    target_value = _clamp(current_value + bounded_delta, min_value, max_value)
                    drift = target_value - anchor_value
                    if max_abs_drift >= 0.0:
                        drift = _clamp(drift, -max_abs_drift, max_abs_drift)
                    target_value = _clamp(anchor_value + drift, min_value, max_value)
                    applied_delta = target_value - current_value
                    if abs(applied_delta) < 1e-9:
                        skipped_traits.append({"trait_key": trait_key, "reason": "no_effect"})
                        continue

                    prev_conf = float(row["confidence"] or 0.0)
                    new_conf = _clamp(max(prev_conf, (prev_conf * 0.92) + (_clamp(candidate_conf, 0.0, 1.0) * 0.16) + 0.01), 0.0, 1.0)
                    new_evidence_count = int(row["evidence_count"] or 0) + 1
                    new_support_score = float(row["support_score"] or 0.0) + abs(applied_delta) * max(0.2, _clamp(candidate_conf, 0.0, 1.0))
                    new_contradiction = float(row["contradiction_score"] or 0.0)
                    prev_status = str(row["status"] or "emerging")
                    contradiction_dominant = new_contradiction >= max(0.045, new_support_score * 0.72)
                    if prev_status == "frozen":
                        new_status = "frozen"
                    elif contradiction_dominant:
                        new_status = "contested"
                    elif new_conf >= 0.68 and new_evidence_count >= 3:
                        new_status = "stable"
                    elif prev_status == "contested" and new_conf >= 0.55 and new_support_score > (new_contradiction * 1.10):
                        new_status = "emerging"
                    else:
                        new_status = prev_status if prev_status in {"stable", "emerging", "contested"} else "emerging"

                    await conn.execute(
                        """
                        UPDATE persona_traits
                        SET current_value = $3,
                            confidence = $4,
                            drift_velocity = $5,
                            evidence_count = $6,
                            support_score = $7,
                            contradiction_score = $8,
                            status = $9,
                            last_changed_at = NOW(),
                            last_reconfirmed_at = NOW(),
                            updated_at = NOW()
                        WHERE persona_id = $1 AND trait_key = $2
                        """,
                        persona_id,
                        trait_key,
                        float(target_value),
                        float(new_conf),
                        float(applied_delta),
                        int(new_evidence_count),
                        float(new_support_score),
                        float(new_contradiction),
                        str(new_status),
                    )

                    evidence_refs_payload = json.dumps(
                        [
                            {"kind": "reflection_id", "value": int(reflection_id)},
                            {"kind": "proposal_evidence", "value": raw.get("reason") or ""},
                            {"kind": "candidate_evidence", "value": raw.get("evidence") or {}},
                        ],
                        ensure_ascii=True,
                    )
                    await conn.execute(
                        """
                        INSERT INTO persona_trait_evidence (
                            persona_id, trait_key, reflection_id, signal_kind, direction, magnitude,
                            signal_confidence, quality_weight, influence_weight, reason_text, evidence_refs, created_at
                        )
                        VALUES (
                            $1, $2, $3, 'reflection_apply', $4, $5,
                            $6, 1.0, 1.0, $7, $8::jsonb, NOW()
                        )
                        """,
                        persona_id,
                        trait_key,
                        int(reflection_id),
                        1.0 if applied_delta >= 0 else -1.0,
                        abs(float(applied_delta)),
                        _clamp(candidate_conf, 0.0, 1.0),
                        str(raw.get("reason") or "")[:220],
                        evidence_refs_payload,
                    )

                    applied_traits.append(
                        {
                            "trait_key": trait_key,
                            "previous_value": round(current_value, 6),
                            "new_value": round(target_value, 6),
                            "applied_delta": round(applied_delta, 6),
                            "confidence": round(new_conf, 6),
                            "status": new_status,
                            "reason": str(raw.get("reason") or "")[:180],
                        }
                    )

                for raw in contested_candidates:
                    trait_key = str(raw.get("trait_key") or "").strip()
                    if not trait_key:
                        skipped_contested_traits.append({"reason": "missing_trait_key"})
                        continue
                    row = await conn.fetchrow(
                        """
                        SELECT
                            t.current_value,
                            t.anchor_value,
                            t.confidence,
                            t.evidence_count,
                            t.support_score,
                            t.contradiction_score,
                            t.status,
                            c.protected_mode
                        FROM persona_traits t
                        JOIN persona_trait_catalog c ON c.trait_key = t.trait_key
                        WHERE t.persona_id = $1 AND t.trait_key = $2
                        FOR UPDATE
                        """,
                        persona_id,
                        trait_key,
                    )
                    if row is None:
                        skipped_contested_traits.append({"trait_key": trait_key, "reason": "trait_not_found"})
                        continue
                    protected_mode = str(row["protected_mode"] or "soft")
                    prev_status = str(row["status"] or "emerging")
                    if protected_mode == "locked" or prev_status == "frozen":
                        skipped_contested_traits.append(
                            {
                                "trait_key": trait_key,
                                "reason": "locked_or_frozen",
                                "status": prev_status,
                            }
                        )
                        continue

                    try:
                        proposed_delta = float(raw.get("proposed_delta", raw.get("delta", 0.0)))
                        candidate_conf = float(raw.get("confidence", 0.0))
                    except (TypeError, ValueError):
                        skipped_contested_traits.append({"trait_key": trait_key, "reason": "invalid_candidate_values"})
                        continue
                    magnitude = abs(float(proposed_delta))
                    if magnitude < 1e-9:
                        skipped_contested_traits.append({"trait_key": trait_key, "reason": "no_effect"})
                        continue

                    prev_conf = float(row["confidence"] or 0.0)
                    prev_support = float(row["support_score"] or 0.0)
                    prev_contradiction = float(row["contradiction_score"] or 0.0)
                    contradiction_inc = magnitude * max(0.18, _clamp(candidate_conf, 0.0, 1.0))
                    next_contradiction = max(0.0, prev_contradiction + contradiction_inc)
                    next_support = max(0.0, prev_support * 0.998)
                    next_conf = _clamp(prev_conf - min(0.05, 0.008 + (_clamp(candidate_conf, 0.0, 1.0) * 0.02)), 0.05, 1.0)

                    if prev_status == "stable":
                        next_status = "contested"
                    elif next_contradiction >= max(0.045, next_support * 0.65):
                        next_status = "contested"
                    else:
                        next_status = prev_status if prev_status in {"emerging", "stable", "contested"} else "emerging"

                    await conn.execute(
                        """
                        UPDATE persona_traits
                        SET confidence = $3,
                            support_score = $4,
                            contradiction_score = $5,
                            status = $6,
                            updated_at = NOW()
                        WHERE persona_id = $1 AND trait_key = $2
                        """,
                        persona_id,
                        trait_key,
                        float(next_conf),
                        float(next_support),
                        float(next_contradiction),
                        str(next_status),
                    )

                    evidence_refs_payload = json.dumps(
                        [
                            {"kind": "reflection_id", "value": int(reflection_id)},
                            {"kind": "conflict_kind", "value": str(raw.get("conflict_kind") or "direction_flip")},
                            {"kind": "current_drift", "value": raw.get("current_drift")},
                            {"kind": "proposal_evidence", "value": raw.get("reason") or ""},
                            {"kind": "candidate_evidence", "value": raw.get("evidence") or {}},
                        ],
                        ensure_ascii=True,
                    )
                    await conn.execute(
                        """
                        INSERT INTO persona_trait_evidence (
                            persona_id, trait_key, reflection_id, signal_kind, direction, magnitude,
                            signal_confidence, quality_weight, influence_weight, reason_text, evidence_refs, created_at
                        )
                        VALUES (
                            $1, $2, $3, 'reflection_conflict', $4, $5,
                            $6, 1.0, 1.0, $7, $8::jsonb, NOW()
                        )
                        """,
                        persona_id,
                        trait_key,
                        int(reflection_id),
                        1.0 if proposed_delta >= 0 else -1.0,
                        float(magnitude),
                        _clamp(candidate_conf, 0.0, 1.0),
                        str(raw.get("reason") or "")[:220],
                        evidence_refs_payload,
                    )

                    applied_contested_traits.append(
                        {
                            "trait_key": trait_key,
                            "previous_status": prev_status,
                            "status": next_status,
                            "previous_confidence": round(prev_conf, 6),
                            "confidence": round(next_conf, 6),
                            "support_score": round(next_support, 6),
                            "contradiction_score": round(next_contradiction, 6),
                            "proposed_delta": round(proposed_delta, 6),
                            "reason": str(raw.get("reason") or "")[:180],
                            "conflict_kind": str(raw.get("conflict_kind") or "direction_flip"),
                        }
                    )

                for raw in episode_promotions:
                    try:
                        episode_id = int(raw.get("episode_id") or 0)
                        conf = float(raw.get("confidence", 0.0))
                    except (TypeError, ValueError):
                        skipped_episodes.append({"reason": "invalid_episode_candidate"})
                        continue
                    to_status = str(raw.get("to_status") or "confirmed").strip().lower()
                    if episode_id <= 0:
                        skipped_episodes.append({"reason": "invalid_episode_id"})
                        continue
                    if to_status != "confirmed":
                        skipped_episodes.append({"episode_id": episode_id, "reason": "unsupported_to_status"})
                        continue

                    row = await conn.fetchrow(
                        """
                        UPDATE persona_episodes
                        SET status = 'confirmed',
                            confidence = GREATEST(confidence, $3),
                            vividness = GREATEST(vividness, LEAST(1.0, 0.40 + ($3 * 0.45))),
                            last_confirmed_at = NOW(),
                            updated_at = NOW()
                        WHERE persona_id = $1
                          AND episode_id = $2
                          AND status <> 'forgotten'
                        RETURNING episode_id, episode_type, status, confidence, callback_line
                        """,
                        persona_id,
                        episode_id,
                        _clamp(conf, 0.0, 1.0),
                    )
                    if row is None:
                        skipped_episodes.append({"episode_id": episode_id, "reason": "episode_not_found"})
                        continue
                    applied_episodes.append(
                        {
                            "episode_id": int(row["episode_id"]),
                            "episode_type": str(row["episode_type"] or ""),
                            "status": str(row["status"] or "confirmed"),
                            "confidence": float(row["confidence"] or 0.0),
                            "callback_line": str(row["callback_line"] or ""),
                        }
                    )

                overlay_applied = False
                if payload_overlay:
                    await conn.execute(
                        """
                        UPDATE persona_global_state
                        SET overlay_summary = $2,
                            updated_at = NOW()
                        WHERE persona_id = $1
                        """,
                        persona_id,
                        payload_overlay,
                    )
                    overlay_applied = True
                else:
                    overlay_applied = False

        return {
            "ok": True,
            "reflection_id": int(reflection_id),
            "applied_trait_updates": applied_traits,
            "skipped_trait_updates": skipped_traits,
            "applied_contested_trait_updates": applied_contested_traits,
            "skipped_contested_trait_updates": skipped_contested_traits,
            "applied_episode_promotions": applied_episodes,
            "skipped_episode_promotions": skipped_episodes,
            "overlay_summary_applied": bool(overlay_applied),
            "overlay_summary": payload_overlay,
        }

    async def append_persona_audit_log(
        self,
        persona_id: str,
        *,
        actor_type: str,
        actor_user_id: str | None,
        action: str,
        entity_type: str,
        entity_id: str,
        before_json: dict[str, object] | None = None,
        after_json: dict[str, object] | None = None,
        diff_json: dict[str, object] | None = None,
        evidence_refs: list[dict[str, object]] | None = None,
        reason: str = "",
    ) -> int:
        payload_before = json.dumps(before_json or {}, ensure_ascii=True) if before_json is not None else None
        payload_after = json.dumps(after_json or {}, ensure_ascii=True) if after_json is not None else None
        payload_diff = json.dumps(diff_json or {}, ensure_ascii=True) if diff_json is not None else None
        payload_evidence = json.dumps(evidence_refs or [], ensure_ascii=True)
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            value = await conn.fetchval(
                """
                INSERT INTO persona_audit_log (
                    persona_id, actor_type, actor_user_id, action, entity_type, entity_id,
                    before_json, after_json, diff_json, evidence_refs, reason, created_at
                )
                VALUES (
                    $1, $2, $3, $4, $5, $6,
                    $7::jsonb, $8::jsonb, $9::jsonb, $10::jsonb, $11, NOW()
                )
                RETURNING audit_id
                """,
                persona_id,
                str(actor_type or "system")[:30],
                (str(actor_user_id).strip()[:60] if actor_user_id else None),
                str(action or "")[:80],
                str(entity_type or "")[:80],
                str(entity_id or "")[:160],
                payload_before,
                payload_after,
                payload_diff,
                payload_evidence,
                str(reason or "")[:300],
            )
        return int(value or 0)

    async def ingest_persona_user_message(
        self,
        persona_id: str,
        guild_id: str,
        channel_id: str,
        user_id: str,
        message_id: int,
        user_text: str,
        *,
        user_label: str = "",
        modality: str = "text",
        source: str = "unknown",
        quality: float = 1.0,
        relationship_update: dict[str, object] | None = None,
        episode_candidate: dict[str, object] | None = None,
        daily_influence_cap: float = 1.0,
        eligible_for_growth: bool = True,
    ) -> Dict[str, object]:
        persona_key = str(persona_id or "").strip()
        guild_key = str(guild_id or "").strip()
        channel_key = str(channel_id or "").strip()
        user_key = str(user_id or "").strip()
        text = str(user_text or "").strip()
        if not (persona_key and guild_key and channel_key and user_key and text):
            return {"applied": False, "reason": "invalid_input"}

        rel = relationship_update if isinstance(relationship_update, dict) else {}
        episode = episode_candidate if isinstance(episode_candidate, dict) else None
        signal_conf = _clamp(float(rel.get("signal_confidence", 0.45)), 0.0, 1.0)
        quality_weight = _clamp(float(rel.get("quality_weight", quality)), 0.0, 1.0)
        influence_weight_in = max(0.0, float(rel.get("influence_weight", 0.5)))
        toxicity_risk = _clamp(float(rel.get("toxicity_risk", 0.0)), 0.0, 1.0)
        consistency_target = _clamp(float(rel.get("consistency_score", 0.5)), 0.0, 1.0)
        reason_text = str(rel.get("reason_text", "") or "").strip()[:240]
        deltas: dict[str, float] = {}
        for dim in (
            "familiarity",
            "trust",
            "warmth",
            "banter_license",
            "support_sensitivity",
            "boundary_sensitivity",
            "topic_alignment",
        ):
            try:
                deltas[dim] = float(rel.get(dim, 0.0))
            except (TypeError, ValueError):
                deltas[dim] = 0.0

        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                pref = await conn.fetchrow(
                    """
                    SELECT memory_mode, allow_episodic_callbacks, allow_personality_influence
                    FROM persona_user_memory_prefs
                    WHERE guild_id = $1 AND user_id = $2
                    """,
                    guild_key,
                    user_key,
                )
                memory_mode = str((pref["memory_mode"] if pref else "full") or "full").strip().lower()
                if memory_mode == "none":
                    return {"applied": False, "skipped": True, "reason": "memory_mode_none"}
                allow_callbacks = bool(pref["allow_episodic_callbacks"]) if pref is not None else True
                allow_personality_influence = bool(pref["allow_personality_influence"]) if pref is not None else True

                inserted = await conn.fetchval(
                    """
                    INSERT INTO persona_ingested_messages (
                        persona_id, message_id, guild_id, channel_id, user_id, modality, source, quality, event_kind
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'user_message')
                    ON CONFLICT(persona_id, message_id) DO NOTHING
                    RETURNING message_id
                    """,
                    persona_key,
                    int(message_id),
                    guild_key,
                    channel_key,
                    user_key,
                    str(modality or "text")[:20],
                    str(source or "unknown")[:80],
                    _clamp(float(quality), 0.0, 1.0),
                )
                if inserted is None:
                    return {"applied": False, "deduped": True, "reason": "duplicate_message_ingest"}

                await conn.execute(
                    """
                    INSERT INTO persona_relationships (
                        persona_id, guild_id, user_id, user_label_cache, status, consent_scope,
                        last_interaction_at, updated_at
                    )
                    VALUES ($1, $2, $3, $4, 'active', $5, NOW(), NOW())
                    ON CONFLICT(persona_id, guild_id, user_id) DO NOTHING
                    """,
                    persona_key,
                    guild_key,
                    user_key,
                    str(user_label or "")[:160],
                    ("minimal" if memory_mode == "minimal" else "full"),
                )
                current = await conn.fetchrow(
                    """
                    SELECT familiarity, trust, warmth, banter_license, support_sensitivity, boundary_sensitivity,
                           topic_alignment, confidence, effective_influence_weight, interaction_count,
                           voice_turn_count, text_turn_count, daily_influence_budget_day, daily_influence_budget_used,
                           consistency_score, risk_flags
                    FROM persona_relationships
                    WHERE persona_id = $1 AND guild_id = $2 AND user_id = $3
                    """,
                    persona_key,
                    guild_key,
                    user_key,
                )
                if current is None:
                    return {"applied": False, "reason": "relationship_missing_after_insert"}

                today = await conn.fetchval("SELECT CURRENT_DATE")
                prev_budget_day = current["daily_influence_budget_day"]
                budget_used = 0.0 if prev_budget_day != today else float(current["daily_influence_budget_used"] or 0.0)
                cap_value = max(0.1, float(daily_influence_cap))
                global_weight_raw = influence_weight_in if allow_personality_influence else 0.0
                applied_global_weight = _clamp(global_weight_raw, 0.0, max(0.0, cap_value - budget_used))
                next_budget_used = _clamp(budget_used + applied_global_weight, 0.0, cap_value)

                local_scale = _clamp(signal_conf * quality_weight, 0.0, 1.0)
                if memory_mode == "minimal":
                    local_scale *= 0.72
                if toxicity_risk >= 0.65:
                    local_scale *= 0.35
                applied_deltas: dict[str, float] = {}
                for dim, raw in deltas.items():
                    max_step = 0.08 if dim in {"familiarity", "trust", "warmth"} else 0.06
                    applied_deltas[dim] = _clamp(raw * local_scale, -max_step, max_step)

                def _next(dim: str, lo: float = 0.0, hi: float = 1.0) -> float:
                    return _clamp(float(current[dim]) + applied_deltas.get(dim, 0.0), lo, hi)

                risk_raw = current["risk_flags"]
                if isinstance(risk_raw, dict):
                    risk_flags = dict(risk_raw)
                else:
                    try:
                        risk_flags_any = json.loads(str(risk_raw or "{}"))
                    except Exception:
                        risk_flags_any = {}
                    risk_flags = risk_flags_any if isinstance(risk_flags_any, dict) else {}
                risk_flags.update(
                    {
                        "toxicity_risk": round(float(toxicity_risk), 3),
                        "last_modality": str(modality or "text")[:20],
                        "last_source": str(source or "unknown")[:80],
                    }
                )

                next_conf = _clamp(float(current["confidence"]) + 0.003 + (0.006 * local_scale), 0.0, 1.0)
                next_eff_weight = _clamp(
                    (float(current["effective_influence_weight"]) * 0.88) + (applied_global_weight * 0.12),
                    0.0,
                    2.5,
                )
                next_consistency = _clamp(
                    (float(current["consistency_score"]) * 0.92) + (consistency_target * 0.08),
                    0.0,
                    1.0,
                )
                consent_scope = "minimal" if memory_mode == "minimal" else "full"
                rel_status = "limited" if memory_mode == "minimal" else "active"
                is_voice = str(modality or "text").strip().lower() == "voice"
                await conn.execute(
                    """
                    UPDATE persona_relationships
                    SET user_label_cache = CASE WHEN $4 <> '' THEN $4 ELSE user_label_cache END,
                        status = $5,
                        consent_scope = $6,
                        familiarity = $7,
                        trust = $8,
                        warmth = $9,
                        banter_license = $10,
                        support_sensitivity = $11,
                        boundary_sensitivity = $12,
                        topic_alignment = $13,
                        confidence = $14,
                        effective_influence_weight = $15,
                        interaction_count = interaction_count + 1,
                        voice_turn_count = voice_turn_count + $16,
                        text_turn_count = text_turn_count + $17,
                        risk_flags = COALESCE(risk_flags, '{}'::jsonb) || $18::jsonb,
                        last_interaction_at = NOW(),
                        updated_at = NOW(),
                        daily_influence_budget_day = $19,
                        daily_influence_budget_used = $20,
                        last_message_id = $21,
                        consistency_score = $22
                    WHERE persona_id = $1 AND guild_id = $2 AND user_id = $3
                    """,
                    persona_key,
                    guild_key,
                    user_key,
                    str(user_label or "")[:160],
                    rel_status,
                    consent_scope,
                    _next("familiarity"),
                    _next("trust"),
                    _next("warmth"),
                    _next("banter_license"),
                    _next("support_sensitivity"),
                    _next("boundary_sensitivity"),
                    _next("topic_alignment", -1.0, 1.0),
                    next_conf,
                    next_eff_weight,
                    1 if is_voice else 0,
                    0 if is_voice else 1,
                    json.dumps(risk_flags, ensure_ascii=True),
                    today,
                    next_budget_used,
                    int(message_id),
                    next_consistency,
                )

                for dim, delta in applied_deltas.items():
                    if abs(delta) < 0.0009:
                        continue
                    await conn.execute(
                        """
                        INSERT INTO persona_relationship_evidence (
                            persona_id, guild_id, user_id, message_id, dimension_key, delta,
                            signal_confidence, quality_weight, influence_weight, reason_text
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        """,
                        persona_key,
                        guild_key,
                        user_key,
                        int(message_id),
                        dim,
                        delta,
                        signal_conf,
                        quality_weight,
                        applied_global_weight,
                        reason_text,
                    )

                episode_id: int | None = None
                episode_created = False
                if episode and allow_callbacks:
                    dedupe_key = str(episode.get("dedupe_key") or "").strip()[:200]
                    title = str(episode.get("title") or "").strip()[:140]
                    summary = str(episode.get("summary") or "").strip()[:420]
                    if dedupe_key and title and summary:
                        ep_row = await conn.fetchrow(
                            """
                            INSERT INTO persona_episodes (
                                persona_id, guild_id, channel_id, source_dedupe_key, episode_type,
                                title, summary, callback_line, status, privacy_level, callback_safety,
                                valence, importance, vividness, confidence, source_message_count,
                                first_message_id, last_message_id, last_participant_user_id
                            )
                            VALUES (
                                $1, $2, $3, $4, $5,
                                $6, $7, $8, $9, $10, $11,
                                $12, $13, $14, $15, 1,
                                $16, $16, $17
                            )
                            ON CONFLICT(source_dedupe_key) DO UPDATE SET
                                importance = GREATEST(persona_episodes.importance, EXCLUDED.importance),
                                vividness = GREATEST(persona_episodes.vividness * 0.92, EXCLUDED.vividness),
                                confidence = GREATEST(persona_episodes.confidence, EXCLUDED.confidence),
                                source_message_count = persona_episodes.source_message_count + 1,
                                last_message_id = EXCLUDED.last_message_id,
                                updated_at = NOW()
                            RETURNING episode_id, (xmax = 0) AS inserted_fresh
                            """,
                            persona_key,
                            guild_key,
                            channel_key,
                            dedupe_key,
                            str(episode.get("episode_type") or "moment").strip().lower()[:40],
                            title,
                            summary,
                            str(episode.get("callback_line") or "")[:220],
                            str(episode.get("status") or "candidate").strip().lower()[:20],
                            str(episode.get("privacy_level") or "participants_only").strip().lower()[:24],
                            str(episode.get("callback_safety") or "safe").strip().lower()[:20],
                            _clamp(float(episode.get("valence", 0.0)), -1.0, 1.0),
                            _clamp(float(episode.get("importance", 0.4)), 0.0, 1.0),
                            _clamp(float(episode.get("vividness", 0.4)), 0.0, 1.0),
                            _clamp(float(episode.get("confidence", 0.4)), 0.0, 1.0),
                            int(message_id),
                            user_key,
                        )
                        if ep_row is not None:
                            episode_id = int(ep_row["episode_id"])
                            try:
                                episode_created = bool(ep_row["inserted_fresh"])
                            except Exception:
                                episode_created = False
                            await conn.execute(
                                """
                                INSERT INTO persona_episode_participants (
                                    episode_id, guild_id, user_id, participant_role, influence_weight, mention_count
                                )
                                VALUES ($1, $2, $3, 'speaker', $4, 1)
                                ON CONFLICT(episode_id, user_id) DO UPDATE SET
                                    mention_count = persona_episode_participants.mention_count + 1,
                                    influence_weight = GREATEST(persona_episode_participants.influence_weight, EXCLUDED.influence_weight)
                                """,
                                episode_id,
                                guild_key,
                                user_key,
                                _clamp(influence_weight_in, 0.0, 2.5),
                            )
                            await conn.execute(
                                """
                                INSERT INTO persona_episode_evidence (
                                    episode_id, message_id, user_id, extractor, confidence, snippet
                                )
                                VALUES ($1, $2, $3, 'persona_episode_heuristic', $4, $5)
                                """,
                                episode_id,
                                int(message_id),
                                user_key,
                                _clamp(float(episode.get("confidence", 0.4)), 0.0, 1.0),
                                str(episode.get("snippet") or text)[:220],
                            )
                            await conn.execute(
                                """
                                UPDATE persona_relationships
                                SET last_episode_candidate_at = NOW(), updated_at = NOW()
                                WHERE persona_id = $1 AND guild_id = $2 AND user_id = $3
                                """,
                                persona_key,
                                guild_key,
                                user_key,
                            )

                await conn.execute(
                    """
                    UPDATE persona_global_state
                    SET total_messages_seen = total_messages_seen + 1,
                        eligible_messages_seen = eligible_messages_seen + $2,
                        unique_users_seen = (SELECT COUNT(*) FROM persona_relationships pr WHERE pr.persona_id = $1),
                        updated_at = NOW()
                    WHERE persona_id = $1
                    """,
                    persona_key,
                    (1 if eligible_for_growth else 0),
                )

                return {
                    "applied": True,
                    "deduped": False,
                    "memory_mode": memory_mode,
                    "allow_callbacks": allow_callbacks,
                    "allow_personality_influence": allow_personality_influence,
                    "applied_global_influence_weight": float(applied_global_weight),
                    "daily_budget_used": float(next_budget_used),
                    "episode_id": episode_id,
                    "episode_created": episode_created,
                    "relationship_confidence": float(next_conf),
                    "toxicity_risk": float(toxicity_risk),
                }

    async def list_persona_episode_callbacks(
        self,
        persona_id: str,
        guild_id: str,
        channel_id: str,
        user_id: str | None,
        *,
        limit: int = 3,
        query_text: str = "",
    ) -> List[Dict[str, object]]:
        target_user = str(user_id or "").strip()
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    e.episode_id, e.channel_id, e.episode_type, e.title, e.summary, e.callback_line,
                    e.status, e.privacy_level, e.callback_safety, e.importance, e.vividness, e.confidence,
                    e.valence, e.source_message_count, e.recall_count, e.updated_at,
                    EXISTS (
                        SELECT 1 FROM persona_episode_participants p
                        WHERE p.episode_id = e.episode_id AND p.user_id = $3
                    ) AS has_target_user,
                    COALESCE((
                        SELECT COUNT(*) FROM persona_episode_participants p
                        WHERE p.episode_id = e.episode_id
                    ), 0) AS participant_count
                FROM persona_episodes e
                WHERE e.persona_id = $1
                  AND e.guild_id = $2
                  AND e.status IN ('candidate', 'confirmed')
                  AND e.callback_line <> ''
                  AND e.callback_safety <> 'avoid'
                ORDER BY e.updated_at DESC
                LIMIT 80
                """,
                persona_id,
                guild_id,
                target_user,
            )

        query_tokens = _tokenize_text(query_text)
        scored: list[tuple[float, Dict[str, object]]] = []
        for row in rows:
            privacy_level = str(row["privacy_level"] or "participants_only").strip().lower()
            has_target_user = bool(row["has_target_user"])
            if privacy_level == "private":
                continue
            if target_user in {"", "*"}:
                if privacy_level != "public":
                    continue
            elif privacy_level == "participants_only" and not has_target_user:
                continue

            callback_line = str(row["callback_line"] or "").strip()
            if not callback_line:
                continue
            score = (
                float(row["importance"]) * 2.0
                + float(row["confidence"]) * 1.7
                + float(row["vividness"]) * 1.5
                + (0.45 if str(row["channel_id"] or "") == str(channel_id) else 0.0)
                + (0.65 if has_target_user else 0.0)
                + (0.30 if str(row["status"]) == "confirmed" else 0.0)
                - (0.14 if int(row["recall_count"] or 0) >= 2 else 0.0)
            )
            if query_tokens:
                ep_tokens = _tokenize_text(f"{row['title']} {row['summary']} {callback_line}")
                overlap = len(query_tokens.intersection(ep_tokens))
                if overlap:
                    score += 0.18 + (overlap / max(1.0, float(len(query_tokens)))) * 1.45
            payload = {
                "episode_id": int(row["episode_id"]),
                "episode_type": str(row["episode_type"]),
                "title": str(row["title"]),
                "summary": str(row["summary"]),
                "callback_line": callback_line,
                "status": str(row["status"]),
                "privacy_level": privacy_level,
                "callback_safety": str(row["callback_safety"]),
                "importance": float(row["importance"]),
                "vividness": float(row["vividness"]),
                "confidence": float(row["confidence"]),
                "valence": float(row["valence"]),
                "source_message_count": int(row["source_message_count"] or 0),
                "participant_count": int(row["participant_count"] or 0),
                "has_target_user": has_target_user,
                "channel_id": str(row["channel_id"] or ""),
                "updated_at": str(row["updated_at"]),
            }
            scored.append((score, payload))

        scored.sort(key=lambda item: item[0], reverse=True)
        out: List[Dict[str, object]] = []
        seen: set[str] = set()
        for _, payload in scored:
            key = str(payload.get("callback_line") or "").casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(payload)
            if len(out) >= max(0, int(limit)):
                break
        return out

    async def touch_persona_episode_recalls(
        self,
        persona_id: str,
        episode_ids: list[int],
    ) -> Dict[str, object]:
        ids = sorted({int(item) for item in (episode_ids or []) if int(item) > 0})
        if not ids:
            return {"ok": True, "updated_count": 0, "episode_ids": []}
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                UPDATE persona_episodes
                SET recall_count = recall_count + 1,
                    last_recalled_at = NOW(),
                    vividness = LEAST(1.0, GREATEST(vividness, vividness * 0.97 + 0.05)),
                    confidence = LEAST(1.0, GREATEST(confidence, confidence * 0.985 + 0.015)),
                    updated_at = NOW()
                WHERE persona_id = $1
                  AND episode_id = ANY($2::bigint[])
                  AND status IN ('candidate', 'confirmed')
                RETURNING episode_id
                """,
                persona_id,
                ids,
            )
        touched = [int(row["episode_id"]) for row in rows]
        return {
            "ok": True,
            "updated_count": len(touched),
            "episode_ids": touched,
        }

    async def list_persona_episodes(
        self,
        persona_id: str,
        guild_id: str,
        *,
        user_id: str | None = None,
        channel_id: str | None = None,
        limit: int = 12,
    ) -> List[Dict[str, object]]:
        params: list[object] = [persona_id, guild_id]
        clauses = ["e.persona_id = $1", "e.guild_id = $2"]
        if channel_id:
            params.append(channel_id)
            clauses.append(f"e.channel_id = ${len(params)}")
        if user_id:
            params.append(user_id)
            clauses.append(
                f"EXISTS (SELECT 1 FROM persona_episode_participants p WHERE p.episode_id = e.episode_id AND p.user_id = ${len(params)})"
            )
        params.append(max(1, int(limit)))
        sql = f"""
            SELECT
                e.episode_id, e.guild_id, e.channel_id, e.episode_type, e.title, e.summary, e.callback_line,
                e.status, e.privacy_level, e.callback_safety, e.valence, e.importance, e.vividness, e.confidence,
                e.source_message_count, e.first_message_id, e.last_message_id, e.recall_count,
                e.last_participant_user_id, e.created_at, e.updated_at,
                COALESCE((SELECT COUNT(*) FROM persona_episode_participants p WHERE p.episode_id = e.episode_id), 0) AS participant_count,
                COALESCE((
                    SELECT ee.snippet
                    FROM persona_episode_evidence ee
                    WHERE ee.episode_id = e.episode_id
                    ORDER BY ee.episode_evidence_id DESC
                    LIMIT 1
                ), '') AS latest_snippet
            FROM persona_episodes e
            WHERE {' AND '.join(clauses)}
            ORDER BY
                CASE e.status WHEN 'confirmed' THEN 3 WHEN 'candidate' THEN 2 ELSE 1 END DESC,
                e.updated_at DESC
            LIMIT ${len(params)}
        """
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [
            {
                "episode_id": int(row["episode_id"]),
                "guild_id": str(row["guild_id"]),
                "channel_id": str(row["channel_id"]),
                "episode_type": str(row["episode_type"]),
                "title": str(row["title"]),
                "summary": str(row["summary"]),
                "callback_line": str(row["callback_line"] or ""),
                "status": str(row["status"]),
                "privacy_level": str(row["privacy_level"]),
                "callback_safety": str(row["callback_safety"]),
                "valence": float(row["valence"]),
                "importance": float(row["importance"]),
                "vividness": float(row["vividness"]),
                "confidence": float(row["confidence"]),
                "source_message_count": int(row["source_message_count"] or 0),
                "first_message_id": int(row["first_message_id"]) if row["first_message_id"] is not None else None,
                "last_message_id": int(row["last_message_id"]) if row["last_message_id"] is not None else None,
                "recall_count": int(row["recall_count"] or 0),
                "last_participant_user_id": str(row["last_participant_user_id"] or ""),
                "participant_count": int(row["participant_count"] or 0),
                "latest_snippet": str(row["latest_snippet"] or ""),
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
            }
            for row in rows
        ]

    async def get_persona_reflection_window(
        self,
        persona_id: str,
        *,
        after_message_id: int,
        message_limit: int = 160,
        evidence_limit: int = 240,
        episode_limit: int = 40,
        relationship_limit: int = 12,
    ) -> Dict[str, object]:
        cursor_id = max(0, int(after_message_id))
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            ingested_summary = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) AS ingested_count,
                    COALESCE(MAX(im.message_id), 0) AS max_message_id,
                    COUNT(DISTINCT im.user_id) AS unique_users,
                    COALESCE(SUM(CASE WHEN im.modality = 'voice' THEN 1 ELSE 0 END), 0) AS voice_messages,
                    COALESCE(SUM(CASE WHEN im.modality <> 'voice' THEN 1 ELSE 0 END), 0) AS text_messages
                FROM persona_ingested_messages im
                WHERE im.persona_id = $1
                  AND im.message_id > $2
                """,
                persona_id,
                cursor_id,
            )
            message_rows = await conn.fetch(
                """
                SELECT im.message_id, im.guild_id, im.channel_id, im.user_id, im.modality, im.source, im.quality, im.ingested_at,
                       m.author_label, m.content_clean, m.created_at
                FROM persona_ingested_messages im
                JOIN messages m ON m.message_id = im.message_id
                WHERE im.persona_id = $1
                  AND im.message_id > $2
                ORDER BY im.message_id ASC
                LIMIT $3
                """,
                persona_id,
                cursor_id,
                max(1, int(message_limit)),
            )
            evidence_rows = await conn.fetch(
                """
                SELECT
                    re.guild_id,
                    re.user_id,
                    re.dimension_key,
                    COUNT(*) AS sample_count,
                    COALESCE(SUM(re.delta), 0.0) AS delta_sum,
                    COALESCE(AVG(re.signal_confidence), 0.0) AS avg_signal_confidence,
                    COALESCE(AVG(re.quality_weight), 0.0) AS avg_quality_weight,
                    COALESCE(AVG(re.influence_weight), 0.0) AS avg_influence_weight,
                    MAX(re.message_id) AS max_message_id,
                    MAX(re.created_at) AS last_seen_at
                FROM persona_relationship_evidence re
                WHERE re.persona_id = $1
                  AND COALESCE(re.message_id, 0) > $2
                GROUP BY re.guild_id, re.user_id, re.dimension_key
                ORDER BY ABS(COALESCE(SUM(re.delta), 0.0)) DESC, COUNT(*) DESC
                LIMIT $3
                """,
                persona_id,
                cursor_id,
                max(1, int(evidence_limit)),
            )
            episode_rows = await conn.fetch(
                """
                SELECT
                    e.episode_id, e.guild_id, e.channel_id, e.episode_type, e.title, e.summary, e.callback_line,
                    e.status, e.privacy_level, e.callback_safety, e.valence, e.importance, e.vividness, e.confidence,
                    e.source_message_count, e.first_message_id, e.last_message_id, e.updated_at,
                    COALESCE((
                        SELECT COUNT(*) FROM persona_episode_participants p WHERE p.episode_id = e.episode_id
                    ), 0) AS participant_count
                FROM persona_episodes e
                WHERE e.persona_id = $1
                  AND COALESCE(e.last_message_id, e.first_message_id, 0) > $2
                ORDER BY e.updated_at DESC
                LIMIT $3
                """,
                persona_id,
                cursor_id,
                max(1, int(episode_limit)),
            )
            relationship_rows = await conn.fetch(
                """
                SELECT
                    guild_id, user_id, user_label_cache, status, consent_scope,
                    familiarity, trust, warmth, banter_license, support_sensitivity, boundary_sensitivity,
                    topic_alignment, confidence, interaction_count, voice_turn_count, text_turn_count,
                    effective_influence_weight, consistency_score, last_interaction_at, updated_at,
                    daily_influence_budget_used, daily_influence_budget_day
                FROM persona_relationships
                WHERE persona_id = $1
                ORDER BY updated_at DESC
                LIMIT $2
                """,
                persona_id,
                max(1, int(relationship_limit)),
            )

        summary_row = ingested_summary or {}
        messages = [
            {
                "message_id": int(row["message_id"]),
                "guild_id": str(row["guild_id"]),
                "channel_id": str(row["channel_id"]),
                "user_id": str(row["user_id"]),
                "modality": str(row["modality"]),
                "source": str(row["source"]),
                "quality": float(row["quality"]),
                "author_label": str(row["author_label"] or ""),
                "content": str(row["content_clean"] or ""),
                "created_at": str(row["created_at"]),
                "ingested_at": str(row["ingested_at"]),
            }
            for row in message_rows
        ]
        relationship_evidence = [
            {
                "guild_id": str(row["guild_id"]),
                "user_id": str(row["user_id"]),
                "dimension_key": str(row["dimension_key"]),
                "sample_count": int(row["sample_count"] or 0),
                "delta_sum": float(row["delta_sum"] or 0.0),
                "avg_signal_confidence": float(row["avg_signal_confidence"] or 0.0),
                "avg_quality_weight": float(row["avg_quality_weight"] or 0.0),
                "avg_influence_weight": float(row["avg_influence_weight"] or 0.0),
                "max_message_id": int(row["max_message_id"]) if row["max_message_id"] is not None else None,
                "last_seen_at": str(row["last_seen_at"]) if row["last_seen_at"] is not None else "",
            }
            for row in evidence_rows
        ]
        episodes = [
            {
                "episode_id": int(row["episode_id"]),
                "guild_id": str(row["guild_id"]),
                "channel_id": str(row["channel_id"]),
                "episode_type": str(row["episode_type"]),
                "title": str(row["title"]),
                "summary": str(row["summary"]),
                "callback_line": str(row["callback_line"] or ""),
                "status": str(row["status"]),
                "privacy_level": str(row["privacy_level"]),
                "callback_safety": str(row["callback_safety"]),
                "valence": float(row["valence"]),
                "importance": float(row["importance"]),
                "vividness": float(row["vividness"]),
                "confidence": float(row["confidence"]),
                "source_message_count": int(row["source_message_count"] or 0),
                "first_message_id": int(row["first_message_id"]) if row["first_message_id"] is not None else None,
                "last_message_id": int(row["last_message_id"]) if row["last_message_id"] is not None else None,
                "participant_count": int(row["participant_count"] or 0),
                "updated_at": str(row["updated_at"]),
            }
            for row in episode_rows
        ]
        relationships = [
            {
                "guild_id": str(row["guild_id"]),
                "user_id": str(row["user_id"]),
                "user_label_cache": str(row["user_label_cache"] or ""),
                "status": str(row["status"]),
                "consent_scope": str(row["consent_scope"]),
                "familiarity": float(row["familiarity"]),
                "trust": float(row["trust"]),
                "warmth": float(row["warmth"]),
                "banter_license": float(row["banter_license"]),
                "support_sensitivity": float(row["support_sensitivity"]),
                "boundary_sensitivity": float(row["boundary_sensitivity"]),
                "topic_alignment": float(row["topic_alignment"]),
                "confidence": float(row["confidence"]),
                "interaction_count": int(row["interaction_count"] or 0),
                "voice_turn_count": int(row["voice_turn_count"] or 0),
                "text_turn_count": int(row["text_turn_count"] or 0),
                "effective_influence_weight": float(row["effective_influence_weight"] or 0.0),
                "consistency_score": float(row["consistency_score"] or 0.0),
                "daily_influence_budget_used": float(row["daily_influence_budget_used"] or 0.0),
                "daily_influence_budget_day": str(row["daily_influence_budget_day"]) if row["daily_influence_budget_day"] is not None else "",
                "last_interaction_at": str(row["last_interaction_at"]) if row["last_interaction_at"] is not None else "",
                "updated_at": str(row["updated_at"]),
            }
            for row in relationship_rows
        ]
        return {
            "after_message_id": cursor_id,
            "ingested_count": int(summary_row["ingested_count"] or 0) if summary_row else 0,
            "max_message_id": int(summary_row["max_message_id"] or 0) if summary_row else 0,
            "unique_users": int(summary_row["unique_users"] or 0) if summary_row else 0,
            "voice_messages": int(summary_row["voice_messages"] or 0) if summary_row else 0,
            "text_messages": int(summary_row["text_messages"] or 0) if summary_row else 0,
            "messages": messages,
            "relationship_evidence": relationship_evidence,
            "episodes": episodes,
            "relationships": relationships,
        }

    async def mark_persona_reflection_started(self, reflection_id: int) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE persona_reflections
                SET status = 'running',
                    started_at = NOW()
                WHERE reflection_id = $1
                """,
                int(reflection_id),
            )

    async def update_persona_global_reflection_checkpoint(
        self,
        persona_id: str,
        *,
        reflection_cursor_message_id: int,
        overlay_summary: str | None = None,
    ) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            if overlay_summary is None:
                await conn.execute(
                    """
                    UPDATE persona_global_state
                    SET reflection_cursor_message_id = GREATEST(reflection_cursor_message_id, $2),
                        last_reflection_at = NOW(),
                        updated_at = NOW()
                    WHERE persona_id = $1
                    """,
                    persona_id,
                    max(0, int(reflection_cursor_message_id)),
                )
            else:
                await conn.execute(
                    """
                    UPDATE persona_global_state
                    SET reflection_cursor_message_id = GREATEST(reflection_cursor_message_id, $2),
                        last_reflection_at = NOW(),
                        overlay_summary = $3,
                        updated_at = NOW()
                    WHERE persona_id = $1
                    """,
                    persona_id,
                    max(0, int(reflection_cursor_message_id)),
                    str(overlay_summary).strip()[:800],
                )

    async def run_persona_decay_cycle(
        self,
        persona_id: str,
        *,
        min_interval_minutes: int = 180,
    ) -> Dict[str, object]:
        now_dt = datetime.now(timezone.utc)
        min_interval = max(0, int(min_interval_minutes))

        def _as_dt(value: object) -> datetime | None:
            if value is None:
                return None
            if isinstance(value, datetime):
                dt = value
            else:
                try:
                    text = str(value or "").strip()
                    if not text:
                        return None
                    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
                except Exception:
                    return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt

        def _days_since(value: object) -> float:
            dt = _as_dt(value)
            if dt is None:
                return 0.0
            return max(0.0, (now_dt - dt).total_seconds() / 86400.0)

        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                state_row = await conn.fetchrow(
                    """
                    SELECT last_decay_at
                    FROM persona_global_state
                    WHERE persona_id = $1
                    FOR UPDATE
                    """,
                    persona_id,
                )
                if state_row is None:
                    return {"ok": False, "error": "persona_state_missing"}

                last_decay_at = _as_dt(state_row["last_decay_at"])
                if min_interval > 0 and last_decay_at is not None:
                    elapsed_min = (now_dt - last_decay_at).total_seconds() / 60.0
                    if elapsed_min < float(min_interval):
                        return {
                            "ok": True,
                            "skipped": True,
                            "reason": "min_interval_not_reached",
                            "minutes_remaining": round(max(0.0, float(min_interval) - elapsed_min), 2),
                        }

                trait_rows = await conn.fetch(
                    """
                    SELECT
                        t.trait_key, t.anchor_value, t.current_value, t.confidence, t.drift_velocity,
                        t.evidence_count, t.support_score, t.contradiction_score, t.status,
                        t.last_changed_at, t.last_reconfirmed_at, t.updated_at, t.created_at,
                        c.min_value, c.max_value, c.max_abs_drift, c.plasticity, c.protected_mode
                    FROM persona_traits t
                    JOIN persona_trait_catalog c ON c.trait_key = t.trait_key
                    WHERE t.persona_id = $1
                    """,
                    persona_id,
                )
                rel_rows = await conn.fetch(
                    """
                    SELECT
                        guild_id, user_id, familiarity, trust, warmth, banter_license,
                        support_sensitivity, boundary_sensitivity, topic_alignment, confidence,
                        effective_influence_weight, status, last_interaction_at, updated_at, created_at
                    FROM persona_relationships
                    WHERE persona_id = $1
                    """,
                    persona_id,
                )
                ep_rows = await conn.fetch(
                    """
                    SELECT
                        episode_id, status, importance, vividness, confidence, recall_count,
                        source_message_count, updated_at, last_recalled_at, last_confirmed_at, created_at
                    FROM persona_episodes
                    WHERE persona_id = $1
                    """,
                    persona_id,
                )

                trait_updates = 0
                rel_updates = 0
                episode_updates = 0
                episode_archived = 0
                trait_examples: list[dict[str, object]] = []
                rel_examples: list[dict[str, object]] = []
                episode_examples: list[dict[str, object]] = []

                for row in trait_rows:
                    protected = str(row["protected_mode"] or "soft")
                    status = str(row["status"] or "emerging")
                    if protected == "locked" or status == "frozen":
                        continue
                    age_days = _days_since(row["last_reconfirmed_at"] or row["last_changed_at"] or row["created_at"])
                    if age_days < 0.75:
                        continue

                    anchor = float(row["anchor_value"] or 0.0)
                    current = float(row["current_value"] or 0.0)
                    conf = float(row["confidence"] or 0.0)
                    plasticity = float(row["plasticity"] or 0.0)
                    min_value = float(row["min_value"] or 0.0)
                    max_value = float(row["max_value"] or 1.0)
                    max_abs_drift = abs(float(row["max_abs_drift"] or 0.0))
                    drift = current - anchor

                    drift_pull_rate = 0.0016 + (plasticity * 0.0028)
                    conf_decay_rate = 0.0009 + (plasticity * 0.0015)
                    drift_pull = min(abs(drift), age_days * drift_pull_rate)
                    next_current = current
                    if abs(drift) > 1e-9 and drift_pull > 0.0:
                        next_current = current - (drift / abs(drift)) * drift_pull

                    bounded_drift = _clamp(next_current - anchor, -max_abs_drift, max_abs_drift)
                    next_current = _clamp(anchor + bounded_drift, min_value, max_value)
                    next_conf = _clamp(conf - min(0.06, age_days * conf_decay_rate), 0.05, 1.0)
                    next_support = max(0.0, float(row["support_score"] or 0.0) * max(0.90, 1.0 - (age_days * 0.01)))
                    next_contra = max(
                        0.0,
                        float(row["contradiction_score"] or 0.0) * max(0.88, 1.0 - (age_days * 0.012)),
                    )
                    next_status = status
                    if status == "stable" and next_conf < 0.56:
                        next_status = "emerging"
                    elif status == "contested" and next_contra < 0.04 and next_conf >= 0.45:
                        next_status = "emerging"

                    if (
                        abs(next_current - current) < 1e-6
                        and abs(next_conf - conf) < 1e-6
                        and abs(next_support - float(row["support_score"] or 0.0)) < 1e-6
                        and abs(next_contra - float(row["contradiction_score"] or 0.0)) < 1e-6
                        and next_status == status
                    ):
                        continue

                    await conn.execute(
                        """
                        UPDATE persona_traits
                        SET current_value = $3,
                            confidence = $4,
                            drift_velocity = $5,
                            support_score = $6,
                            contradiction_score = $7,
                            status = $8,
                            updated_at = NOW()
                        WHERE persona_id = $1 AND trait_key = $2
                        """,
                        persona_id,
                        str(row["trait_key"]),
                        float(next_current),
                        float(next_conf),
                        float(next_current - current),
                        float(next_support),
                        float(next_contra),
                        next_status,
                    )
                    trait_updates += 1
                    if len(trait_examples) < 4:
                        trait_examples.append(
                            {
                                "trait_key": str(row["trait_key"]),
                                "delta": round(next_current - current, 6),
                                "confidence_delta": round(next_conf - conf, 6),
                                "status": next_status,
                            }
                        )

                for row in rel_rows:
                    status = str(row["status"] or "active")
                    if status in {"blocked", "forgotten"}:
                        continue
                    inactivity_days = _days_since(row["last_interaction_at"] or row["created_at"])
                    if inactivity_days < 2.0:
                        continue
                    decay_days = min(45.0, inactivity_days - 1.0)
                    if decay_days <= 0.0:
                        continue

                    familiarity = float(row["familiarity"] or 0.0)
                    trust = float(row["trust"] or 0.0)
                    warmth = float(row["warmth"] or 0.0)
                    banter = float(row["banter_license"] or 0.0)
                    support = float(row["support_sensitivity"] or 0.0)
                    boundary = float(row["boundary_sensitivity"] or 0.0)
                    topic = float(row["topic_alignment"] or 0.0)
                    conf = float(row["confidence"] or 0.0)

                    next_familiarity = _clamp(familiarity - decay_days * 0.0048, 0.08, 1.0)
                    next_trust = _clamp(trust - decay_days * 0.0018, 0.25, 1.0)
                    next_warmth = _clamp(warmth - decay_days * 0.0032, 0.30, 1.0)
                    next_banter = _clamp(banter - decay_days * 0.0049, 0.10, 1.0)
                    next_support = _clamp(support - decay_days * 0.0016, 0.35, 1.0)
                    next_boundary = _clamp(boundary - decay_days * 0.0012, 0.35, 1.0)
                    if topic > 0:
                        next_topic = max(0.0, topic - decay_days * 0.0030)
                    elif topic < 0:
                        next_topic = min(0.0, topic + decay_days * 0.0030)
                    else:
                        next_topic = 0.0
                    next_conf = _clamp(conf - decay_days * 0.0021, 0.10, 1.0)

                    if (
                        abs(next_familiarity - familiarity) < 1e-6
                        and abs(next_trust - trust) < 1e-6
                        and abs(next_warmth - warmth) < 1e-6
                        and abs(next_banter - banter) < 1e-6
                        and abs(next_support - support) < 1e-6
                        and abs(next_boundary - boundary) < 1e-6
                        and abs(next_topic - topic) < 1e-6
                        and abs(next_conf - conf) < 1e-6
                    ):
                        continue

                    await conn.execute(
                        """
                        UPDATE persona_relationships
                        SET familiarity = $4,
                            trust = $5,
                            warmth = $6,
                            banter_license = $7,
                            support_sensitivity = $8,
                            boundary_sensitivity = $9,
                            topic_alignment = $10,
                            confidence = $11,
                            updated_at = NOW()
                        WHERE persona_id = $1 AND guild_id = $2 AND user_id = $3
                        """,
                        persona_id,
                        str(row["guild_id"]),
                        str(row["user_id"]),
                        float(next_familiarity),
                        float(next_trust),
                        float(next_warmth),
                        float(next_banter),
                        float(next_support),
                        float(next_boundary),
                        float(next_topic),
                        float(next_conf),
                    )
                    rel_updates += 1
                    if len(rel_examples) < 4:
                        rel_examples.append(
                            {
                                "guild_id": str(row["guild_id"]),
                                "user_id": str(row["user_id"]),
                                "warmth_delta": round(next_warmth - warmth, 6),
                                "trust_delta": round(next_trust - trust, 6),
                                "banter_delta": round(next_banter - banter, 6),
                            }
                        )

                for row in ep_rows:
                    status = str(row["status"] or "candidate")
                    if status in {"forgotten"}:
                        continue
                    age_days = _days_since(row["last_recalled_at"] or row["last_confirmed_at"] or row["created_at"])
                    if age_days < 1.0:
                        continue
                    vividness = float(row["vividness"] or 0.0)
                    confidence = float(row["confidence"] or 0.0)
                    importance = float(row["importance"] or 0.0)
                    recall_count = int(row["recall_count"] or 0)

                    vivid_decay = 0.0075 if status == "confirmed" else 0.0115
                    conf_decay = 0.0012 if status == "confirmed" else 0.0030
                    imp_decay = 0.0008 if status == "confirmed" else 0.0016
                    recall_buffer = min(0.6, recall_count * 0.06)

                    next_vividness = _clamp(vividness - max(0.0, age_days - recall_buffer) * vivid_decay, 0.04, 1.0)
                    next_confidence = _clamp(confidence - max(0.0, age_days - recall_buffer * 0.6) * conf_decay, 0.08, 1.0)
                    next_importance = _clamp(importance - max(0.0, age_days - recall_buffer * 0.4) * imp_decay, 0.05, 1.0)
                    next_status = status

                    if status == "candidate":
                        if age_days >= 21 and next_vividness < 0.16 and next_importance < 0.20 and next_confidence < 0.35:
                            next_status = "archived"
                    elif status == "confirmed":
                        if age_days >= 120 and recall_count <= 1 and next_vividness < 0.12 and next_importance < 0.18:
                            next_status = "archived"

                    if (
                        abs(next_vividness - vividness) < 1e-6
                        and abs(next_confidence - confidence) < 1e-6
                        and abs(next_importance - importance) < 1e-6
                        and next_status == status
                    ):
                        continue

                    await conn.execute(
                        """
                        UPDATE persona_episodes
                        SET vividness = $3,
                            confidence = $4,
                            importance = $5,
                            status = $6,
                            updated_at = NOW()
                        WHERE persona_id = $1 AND episode_id = $2
                        """,
                        persona_id,
                        int(row["episode_id"]),
                        float(next_vividness),
                        float(next_confidence),
                        float(next_importance),
                        next_status,
                    )
                    episode_updates += 1
                    if next_status == "archived" and status != "archived":
                        episode_archived += 1
                    if len(episode_examples) < 4:
                        episode_examples.append(
                            {
                                "episode_id": int(row["episode_id"]),
                                "status": next_status,
                                "vividness_delta": round(next_vividness - vividness, 6),
                                "importance_delta": round(next_importance - importance, 6),
                            }
                        )

                await conn.execute(
                    """
                    UPDATE persona_global_state
                    SET last_decay_at = NOW(),
                        updated_at = NOW()
                    WHERE persona_id = $1
                    """,
                    persona_id,
                )

        return {
            "ok": True,
            "skipped": False,
            "trait_updates": trait_updates,
            "relationship_updates": rel_updates,
            "episode_updates": episode_updates,
            "episode_archived": episode_archived,
            "trait_examples": trait_examples,
            "relationship_examples": rel_examples,
            "episode_examples": episode_examples,
            "ran_at": now_dt.isoformat(),
        }
