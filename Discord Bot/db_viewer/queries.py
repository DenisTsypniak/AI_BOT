from __future__ import annotations

import asyncpg
import json
import logging
import re
import time

logger = logging.getLogger(__name__)


def _coerce_json_list(value: object) -> list[dict]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            return []
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def _normalize_provenance_row(row: dict) -> dict:
    row["provenance_guild_breakdown"] = _coerce_json_list(row.get("provenance_guild_breakdown"))
    row["provenance_directness_breakdown"] = _coerce_json_list(row.get("provenance_directness_breakdown"))
    return row


def _normalize_limit_offset(limit: int, offset: int, *, default_limit: int, max_limit: int) -> tuple[int, int]:
    try:
        safe_limit = int(limit)
    except (TypeError, ValueError):
        safe_limit = default_limit
    try:
        safe_offset = int(offset)
    except (TypeError, ValueError):
        safe_offset = 0
    if safe_limit <= 0:
        safe_limit = default_limit
    safe_limit = min(safe_limit, max_limit)
    safe_offset = max(0, safe_offset)
    return safe_limit, safe_offset


async def get_users(pool: asyncpg.Pool, limit: int = 50, offset: int = 0) -> list[dict]:
    limit, offset = _normalize_limit_offset(limit, offset, default_limit=50, max_limit=500)
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(
                """
                SELECT 
                    u.guild_id, u.user_id, u.discord_username, u.combined_label, u.first_seen, u.updated_at,
                    (SELECT COUNT(*) FROM messages m WHERE m.user_id = u.user_id AND m.guild_id = u.guild_id) as message_count
                FROM users u
                ORDER BY u.updated_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset,
            )
            return [dict(row) for row in rows]
        except asyncpg.exceptions.UndefinedTableError:
            logger.debug("get_users: users/messages table missing")
            return []


async def get_recent_messages(
    pool: asyncpg.Pool, limit: int = 100, offset: int = 0, filters: dict = None
) -> list[dict]:
    limit, offset = _normalize_limit_offset(limit, offset, default_limit=100, max_limit=10000)
    if filters is None:
        filters = {}
        
    async with pool.acquire() as conn:
        try:
            base_query = """
                SELECT m.message_id, m.session_id, m.guild_id, m.channel_id, m.user_id, 
                       m.author_label, m.role, m.modality, m.content_clean, m.source, m.created_at,
                       s.transcript, s.confidence
                FROM messages m
                LEFT JOIN stt_turns s ON m.message_id = s.message_id
            """
            
            where_clauses = []
            args = []
            idx = 1
            
            # Simple Exact Matches
            for field in ["guild_id", "channel_id", "user_id", "role", "modality", "source"]:
                if filters.get(field):
                    where_clauses.append(f"m.{field} = ${idx}")
                    args.append(filters[field])
                    idx += 1
                    
            # Text Search
            if filters.get("q"):
                where_clauses.append(f"m.content_clean ILIKE ${idx}")
                args.append(f"%{filters['q']}%")
                idx += 1
                
            # Date Range
            if filters.get("from_date"):
                where_clauses.append(f"m.created_at >= ${idx}::timestamp")
                args.append(filters["from_date"])
                idx += 1
            if filters.get("to_date"):
                where_clauses.append(f"m.created_at <= ${idx}::timestamp")
                args.append(filters["to_date"])
                idx += 1
                    
            if where_clauses:
                base_query += " WHERE " + " AND ".join(where_clauses)
                
            base_query += f" ORDER BY m.message_id DESC LIMIT ${idx} OFFSET ${idx+1}"
            args.extend([limit, offset])
            
            rows = await conn.fetch(base_query, *args)
            return [dict(row) for row in rows]
        except asyncpg.exceptions.UndefinedTableError:
            logger.debug("get_recent_messages: messages/stt_turns table missing")
            return []


async def get_user_facts(pool: asyncpg.Pool, limit: int = 100) -> list[dict]:
    limit, _ = _normalize_limit_offset(limit, 0, default_limit=100, max_limit=500)
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(
                """
                SELECT f.fact_id, f.guild_id, f.user_id, u.discord_username, 
                       f.fact_key, f.fact_value, f.fact_type, f.about_target, f.directness, f.evidence_quote,
                       f.confidence, f.importance, 
                       f.status, f.pinned, f.evidence_count, f.updated_at
                FROM user_facts f
                LEFT JOIN users u ON f.user_id = u.user_id AND f.guild_id = u.guild_id
                ORDER BY f.importance DESC, f.updated_at DESC
                LIMIT $1
                """,
                limit,
            )
            return [dict(row) for row in rows]
        except asyncpg.exceptions.UndefinedTableError:
            logger.debug("get_user_facts: user_facts/users table missing")
            return []


async def get_persona_episodes(pool: asyncpg.Pool, limit: int = 50) -> list[dict]:
    limit, _ = _normalize_limit_offset(limit, 0, default_limit=50, max_limit=500)
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(
                """
                SELECT episode_id, persona_id, guild_id, channel_id, episode_type, 
                       title, summary, status, importance, vividness, confidence, 
                       source_message_count, updated_at
                FROM persona_episodes
                ORDER BY importance DESC, updated_at DESC
                LIMIT $1
                """,
                limit,
            )
            return [dict(row) for row in rows]
        except asyncpg.exceptions.UndefinedTableError:
            logger.debug("get_persona_episodes: persona_episodes table missing")
            return []


async def get_memory_provenance_snapshot(
    pool: asyncpg.Pool,
    *,
    limit: int = 80,
    offset: int = 0,
) -> dict[str, list[dict]]:
    limit, offset = _normalize_limit_offset(limit, offset, default_limit=80, max_limit=300)
    snapshot: dict[str, list[dict]] = {
        "global_facts": [],
        "persona_facts": [],
        "local_facts": [],
        "biographies": [],
    }
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(
                """
                SELECT
                    guf.global_fact_id,
                    guf.user_id,
                    CASE
                        WHEN guf.user_id LIKE 'persona::%' THEN guf.user_id
                        ELSE COALESCE(gp.primary_display_name, gp.discord_global_name, gp.discord_username, ul.combined_label, ul.discord_username, guf.user_id)
                    END AS display_name,
                    guf.fact_key,
                    guf.fact_value,
                    guf.fact_type,
                    guf.about_target,
                    guf.directness,
                    guf.evidence_quote,
                    guf.confidence,
                    guf.importance,
                    guf.status,
                    guf.pinned,
                    guf.evidence_count,
                    guf.last_source_guild_id,
                    guf.updated_at,
                    COALESCE(ev.total_evidence, 0) AS provenance_evidence_count,
                    COALESCE(ev.guild_breakdown, '[]'::jsonb) AS provenance_guild_breakdown,
                    COALESCE(ev.directness_breakdown, '[]'::jsonb) AS provenance_directness_breakdown
                FROM global_user_facts guf
                LEFT JOIN global_user_profiles gp ON gp.user_id = guf.user_id
                LEFT JOIN LATERAL (
                    SELECT u.combined_label, u.discord_username
                    FROM users u
                    WHERE u.user_id = guf.user_id
                    ORDER BY u.updated_at DESC
                    LIMIT 1
                ) ul ON TRUE
                LEFT JOIN LATERAL (
                    SELECT
                        COUNT(*)::int AS total_evidence,
                        COALESCE(
                            jsonb_agg(
                                jsonb_build_object('guild_id', e.guild_id, 'count', e.cnt)
                                ORDER BY e.cnt DESC, e.guild_id
                            ),
                            '[]'::jsonb
                        ) AS guild_breakdown,
                        COALESCE(
                            (
                                SELECT jsonb_agg(
                                    jsonb_build_object('directness', d.directness, 'count', d.cnt)
                                    ORDER BY d.cnt DESC, d.directness
                                )
                                FROM (
                                    SELECT COALESCE(directness, 'explicit') AS directness, COUNT(*)::int AS cnt
                                    FROM global_fact_evidence
                                    WHERE global_fact_id = guf.global_fact_id
                                    GROUP BY COALESCE(directness, 'explicit')
                                ) d
                            ),
                            '[]'::jsonb
                        ) AS directness_breakdown
                    FROM (
                        SELECT COALESCE(source_guild_id, '') AS guild_id, COUNT(*)::int AS cnt
                        FROM global_fact_evidence
                        WHERE global_fact_id = guf.global_fact_id
                        GROUP BY COALESCE(source_guild_id, '')
                    ) e
                ) ev ON TRUE
                ORDER BY guf.updated_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset,
            )
            for row in rows:
                item = _normalize_provenance_row(dict(row))
                is_persona = str(item.get("user_id") or "").startswith("persona::")
                item["is_persona_memory"] = is_persona
                if is_persona:
                    snapshot["persona_facts"].append(item)
                else:
                    snapshot["global_facts"].append(item)
        except asyncpg.exceptions.UndefinedTableError:
            logger.debug("get_memory_provenance_snapshot: global fact tables missing")
        except Exception:
            logger.exception("get_memory_provenance_snapshot failed while loading global facts")

        try:
            rows = await conn.fetch(
                """
                SELECT
                    uf.fact_id,
                    uf.guild_id,
                    uf.user_id,
                    COALESCE(u.combined_label, u.discord_username, uf.user_id) AS display_name,
                    uf.fact_key,
                    uf.fact_value,
                    uf.fact_type,
                    uf.about_target,
                    uf.directness,
                    uf.evidence_quote,
                    uf.confidence,
                    uf.importance,
                    uf.status,
                    uf.pinned,
                    uf.evidence_count,
                    uf.updated_at
                FROM user_facts uf
                LEFT JOIN users u ON u.guild_id = uf.guild_id AND u.user_id = uf.user_id
                ORDER BY uf.updated_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset,
            )
            snapshot["local_facts"] = [dict(row) for row in rows]
        except asyncpg.exceptions.UndefinedTableError:
            logger.debug("get_memory_provenance_snapshot: local fact tables missing")
        except Exception:
            logger.exception("get_memory_provenance_snapshot failed while loading local facts")

        try:
            rows = await conn.fetch(
                """
                SELECT subject_kind, subject_id, summary_text, source_fact_count, source_summary_count,
                       source_updated_at, last_source_guild_id, updated_at
                FROM global_biography_summaries
                ORDER BY updated_at DESC
                LIMIT $1
                """,
                max(20, min(limit, 200)),
            )
            snapshot["biographies"] = [dict(row) for row in rows]
        except asyncpg.exceptions.UndefinedTableError:
            logger.debug("get_memory_provenance_snapshot: global_biography_summaries table missing")
        except Exception:
            logger.exception("get_memory_provenance_snapshot failed while loading biographies")

    return snapshot


async def get_persona_memory_snapshot(
    pool: asyncpg.Pool,
    *,
    fact_limit: int = 120,
    fact_offset: int = 0,
    run_limit: int = 40,
    candidate_limit: int = 80,
) -> dict[str, object]:
    fact_limit, fact_offset = _normalize_limit_offset(fact_limit, fact_offset, default_limit=120, max_limit=500)
    run_limit, _ = _normalize_limit_offset(run_limit, 0, default_limit=40, max_limit=200)
    candidate_limit, _ = _normalize_limit_offset(candidate_limit, 0, default_limit=80, max_limit=300)

    snapshot: dict[str, object] = {
        "summary": {},
        "biographies": [],
        "facts": [],
        "recent_runs": [],
        "recent_candidates": [],
        "rejected_candidates": [],
    }

    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*)::int AS total_facts,
                    COUNT(*) FILTER (WHERE status = 'confirmed')::int AS confirmed_facts,
                    COUNT(*) FILTER (WHERE status = 'candidate')::int AS candidate_facts,
                    COUNT(*) FILTER (WHERE pinned = TRUE)::int AS pinned_facts,
                    COUNT(DISTINCT user_id)::int AS persona_subjects,
                    COALESCE(ROUND(AVG(confidence)::numeric, 3), 0)::float8 AS avg_confidence,
                    MAX(updated_at) AS last_fact_update
                FROM global_user_facts
                WHERE user_id LIKE 'persona::%'
                """
            )
            snapshot["summary"] = dict(row) if row is not None else {}
        except asyncpg.exceptions.UndefinedTableError:
            logger.debug("get_persona_memory_snapshot: global_user_facts table missing")
        except Exception:
            logger.exception("get_persona_memory_snapshot failed while loading summary")

        try:
            rows = await conn.fetch(
                """
                SELECT
                    subject_kind,
                    subject_id,
                    summary_text,
                    source_fact_count,
                    source_summary_count,
                    source_updated_at,
                    last_source_guild_id,
                    updated_at
                FROM global_biography_summaries
                WHERE subject_kind = 'persona'
                ORDER BY updated_at DESC
                LIMIT $1
                """,
                max(10, min(fact_limit, 100)),
            )
            snapshot["biographies"] = [dict(row) for row in rows]
        except asyncpg.exceptions.UndefinedTableError:
            logger.debug("get_persona_memory_snapshot: global_biography_summaries table missing")
        except Exception:
            logger.exception("get_persona_memory_snapshot failed while loading biographies")

        try:
            rows = await conn.fetch(
                """
                SELECT
                    guf.global_fact_id,
                    guf.user_id,
                    regexp_replace(guf.user_id, '^persona::', '') AS persona_id,
                    guf.fact_key,
                    guf.fact_value,
                    guf.fact_type,
                    guf.about_target,
                    guf.directness,
                    guf.evidence_quote,
                    guf.confidence,
                    guf.importance,
                    guf.status,
                    guf.pinned,
                    guf.evidence_count,
                    guf.last_source_guild_id,
                    guf.updated_at,
                    COALESCE(ev.total_evidence, 0) AS provenance_evidence_count,
                    COALESCE(ev.guild_breakdown, '[]'::jsonb) AS provenance_guild_breakdown,
                    COALESCE(ev.directness_breakdown, '[]'::jsonb) AS provenance_directness_breakdown
                FROM global_user_facts guf
                LEFT JOIN LATERAL (
                    SELECT
                        COUNT(*)::int AS total_evidence,
                        COALESCE(
                            jsonb_agg(
                                jsonb_build_object('guild_id', e.guild_id, 'count', e.cnt)
                                ORDER BY e.cnt DESC, e.guild_id
                            ),
                            '[]'::jsonb
                        ) AS guild_breakdown,
                        COALESCE(
                            (
                                SELECT jsonb_agg(
                                    jsonb_build_object('directness', d.directness, 'count', d.cnt)
                                    ORDER BY d.cnt DESC, d.directness
                                )
                                FROM (
                                    SELECT COALESCE(directness, 'explicit') AS directness, COUNT(*)::int AS cnt
                                    FROM global_fact_evidence
                                    WHERE global_fact_id = guf.global_fact_id
                                    GROUP BY COALESCE(directness, 'explicit')
                                ) d
                            ),
                            '[]'::jsonb
                        ) AS directness_breakdown
                    FROM (
                        SELECT COALESCE(source_guild_id, '') AS guild_id, COUNT(*)::int AS cnt
                        FROM global_fact_evidence
                        WHERE global_fact_id = guf.global_fact_id
                        GROUP BY COALESCE(source_guild_id, '')
                    ) e
                ) ev ON TRUE
                WHERE guf.user_id LIKE 'persona::%'
                ORDER BY guf.updated_at DESC, guf.global_fact_id DESC
                LIMIT $1 OFFSET $2
                """,
                fact_limit,
                fact_offset,
            )
            snapshot["facts"] = [_normalize_provenance_row(dict(row)) for row in rows]
        except asyncpg.exceptions.UndefinedTableError:
            logger.debug("get_persona_memory_snapshot: persona/global fact tables missing")
        except Exception:
            logger.exception("get_persona_memory_snapshot failed while loading facts")

        try:
            rows = await conn.fetch(
                """
                SELECT
                    run_id, guild_id, channel_id, speaker_user_id, fact_owner_kind, fact_owner_id, speaker_role,
                    modality, source, backend_name, model_name, dry_run, llm_attempted, llm_ok, json_valid, fallback_used,
                    latency_ms, candidate_count, accepted_count, saved_count, filtered_count, error_text, created_at
                FROM memory_extractor_runs
                WHERE fact_owner_kind = 'persona'
                ORDER BY created_at DESC, run_id DESC
                LIMIT $1
                """,
                run_limit,
            )
            snapshot["recent_runs"] = [dict(row) for row in rows]
        except asyncpg.exceptions.UndefinedTableError:
            logger.debug("get_persona_memory_snapshot: memory_extractor_runs table missing")
        except Exception:
            logger.exception("get_persona_memory_snapshot failed while loading recent runs")

        try:
            rows = await conn.fetch(
                """
                SELECT
                    c.candidate_row_id, c.run_id, c.fact_key, c.fact_value, c.fact_type,
                    c.about_target, c.directness, c.evidence_quote, c.confidence, c.importance,
                    c.moderation_action, c.moderation_reason, c.selected_for_apply, c.saved_to_memory, c.created_at,
                    r.guild_id, r.channel_id, r.speaker_user_id, r.fact_owner_kind, r.fact_owner_id,
                    r.backend_name, r.model_name, r.dry_run
                FROM memory_extractor_candidates c
                JOIN memory_extractor_runs r ON r.run_id = c.run_id
                WHERE r.fact_owner_kind = 'persona'
                ORDER BY c.created_at DESC, c.candidate_row_id DESC
                LIMIT $1
                """,
                candidate_limit,
            )
            all_candidates = [dict(row) for row in rows]
            snapshot["recent_candidates"] = all_candidates
            snapshot["rejected_candidates"] = [
                row for row in all_candidates if str(row.get("moderation_action") or "accept") != "accept"
            ]
        except asyncpg.exceptions.UndefinedTableError:
            logger.debug("get_persona_memory_snapshot: memory_extractor_candidates table missing")
        except Exception:
            logger.exception("get_persona_memory_snapshot failed while loading candidates")

    return snapshot


async def get_memory_extractor_diagnostics_snapshot(
    pool: asyncpg.Pool,
    *,
    run_limit: int = 80,
    candidate_limit: int = 120,
) -> dict[str, object]:
    run_limit, _ = _normalize_limit_offset(run_limit, 0, default_limit=80, max_limit=300)
    candidate_limit, _ = _normalize_limit_offset(candidate_limit, 0, default_limit=120, max_limit=500)
    snapshot: dict[str, object] = {
        "summary": {},
        "by_backend": [],
        "recent_runs": [],
        "dry_run_candidates": [],
        "rejected_candidates": [],
    }
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*)::int AS total_runs,
                    COUNT(*) FILTER (WHERE dry_run)::int AS dry_run_runs,
                    COUNT(*) FILTER (WHERE llm_attempted)::int AS llm_attempted_runs,
                    COUNT(*) FILTER (WHERE llm_attempted AND json_valid)::int AS json_valid_runs,
                    COUNT(*) FILTER (WHERE llm_attempted AND NOT json_valid)::int AS json_invalid_runs,
                    COUNT(*) FILTER (WHERE error_text <> '')::int AS error_runs,
                    COALESCE(ROUND(AVG(NULLIF(latency_ms, 0))::numeric, 1), 0)::float8 AS avg_latency_ms,
                    COALESCE(ROUND(AVG(CASE WHEN llm_attempted THEN (CASE WHEN json_valid THEN 1.0 ELSE 0.0 END) END)::numeric, 4), 0)::float8 AS json_valid_rate,
                    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '24 hours')::int AS runs_24h,
                    COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '24 hours' AND dry_run)::int AS dry_run_runs_24h,
                    COALESCE(
                        ROUND(
                            AVG(
                                CASE
                                    WHEN llm_attempted AND created_at >= NOW() - INTERVAL '24 hours'
                                    THEN (CASE WHEN json_valid THEN 1.0 ELSE 0.0 END)
                                END
                            )::numeric,
                            4
                        ),
                        0
                    )::float8 AS json_valid_rate_24h
                FROM memory_extractor_runs
                """
            )
            snapshot["summary"] = dict(row) if row is not None else {}
        except asyncpg.exceptions.UndefinedTableError:
            logger.debug("get_memory_extractor_diagnostics_snapshot: audit tables missing")
            return snapshot
        except Exception:
            logger.exception("get_memory_extractor_diagnostics_snapshot failed while loading summary")
            return snapshot

        try:
            rows = await conn.fetch(
                """
                SELECT
                    backend_name,
                    model_name,
                    COUNT(*)::int AS runs,
                    COUNT(*) FILTER (WHERE dry_run)::int AS dry_run_runs,
                    COUNT(*) FILTER (WHERE llm_attempted AND json_valid)::int AS json_valid_runs,
                    COUNT(*) FILTER (WHERE llm_attempted AND NOT json_valid)::int AS json_invalid_runs,
                    COALESCE(ROUND(AVG(NULLIF(latency_ms, 0))::numeric, 1), 0)::float8 AS avg_latency_ms,
                    COALESCE(ROUND(AVG(CASE WHEN llm_attempted THEN (CASE WHEN json_valid THEN 1.0 ELSE 0.0 END) END)::numeric, 4), 0)::float8 AS json_valid_rate
                FROM memory_extractor_runs
                GROUP BY backend_name, model_name
                ORDER BY runs DESC, backend_name, model_name
                """
            )
            snapshot["by_backend"] = [dict(row) for row in rows]
        except Exception:
            logger.exception("get_memory_extractor_diagnostics_snapshot failed while loading backend breakdown")

        try:
            rows = await conn.fetch(
                """
                SELECT
                    run_id, guild_id, channel_id, speaker_user_id, fact_owner_kind, fact_owner_id, speaker_role,
                    modality, source, backend_name, model_name, dry_run, llm_attempted, llm_ok, json_valid, fallback_used,
                    latency_ms, candidate_count, accepted_count, saved_count, filtered_count, error_text, created_at
                FROM memory_extractor_runs
                ORDER BY created_at DESC, run_id DESC
                LIMIT $1
                """,
                run_limit,
            )
            snapshot["recent_runs"] = [dict(row) for row in rows]
        except Exception:
            logger.exception("get_memory_extractor_diagnostics_snapshot failed while loading recent runs")

        try:
            rows = await conn.fetch(
                """
                SELECT
                    c.candidate_row_id, c.run_id, c.fact_key, c.fact_value, c.fact_type,
                    c.about_target, c.directness, c.evidence_quote, c.confidence, c.importance,
                    c.moderation_action, c.moderation_reason, c.selected_for_apply, c.saved_to_memory, c.created_at,
                    r.guild_id, r.channel_id, r.speaker_user_id, r.fact_owner_kind, r.fact_owner_id,
                    r.backend_name, r.model_name, r.dry_run
                FROM memory_extractor_candidates c
                JOIN memory_extractor_runs r ON r.run_id = c.run_id
                WHERE r.dry_run = TRUE
                ORDER BY c.created_at DESC, c.candidate_row_id DESC
                LIMIT $1
                """,
                candidate_limit,
            )
            snapshot["dry_run_candidates"] = [dict(row) for row in rows]
        except Exception:
            logger.exception("get_memory_extractor_diagnostics_snapshot failed while loading dry-run candidates")

        try:
            rows = await conn.fetch(
                """
                SELECT
                    c.candidate_row_id, c.run_id, c.fact_key, c.fact_value, c.fact_type,
                    c.about_target, c.directness, c.evidence_quote, c.confidence, c.importance,
                    c.moderation_action, c.moderation_reason, c.selected_for_apply, c.saved_to_memory, c.created_at,
                    r.guild_id, r.channel_id, r.speaker_user_id, r.fact_owner_kind, r.fact_owner_id,
                    r.backend_name, r.model_name, r.dry_run
                FROM memory_extractor_candidates c
                JOIN memory_extractor_runs r ON r.run_id = c.run_id
                WHERE c.moderation_action <> 'accept'
                ORDER BY c.created_at DESC, c.candidate_row_id DESC
                LIMIT $1
                """,
                candidate_limit,
            )
            snapshot["rejected_candidates"] = [dict(row) for row in rows]
        except Exception:
            logger.exception("get_memory_extractor_diagnostics_snapshot failed while loading rejected candidates")

    return snapshot


async def get_persona_traits(pool: asyncpg.Pool) -> list[dict]:
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(
                """
                SELECT t.persona_id, t.trait_key, c.label, c.description,
                       t.anchor_value, t.current_value, t.confidence, t.drift_velocity,
                       t.evidence_count, t.status, t.updated_at
                FROM persona_traits t
                JOIN persona_trait_catalog c ON t.trait_key = c.trait_key
                ORDER BY t.current_value DESC
                """
            )
            return [dict(row) for row in rows]
        except asyncpg.exceptions.UndefinedTableError:
            logger.debug("get_persona_traits: persona trait tables missing")
            return []

async def get_message_by_id(pool: asyncpg.Pool, message_id: int) -> dict | None:
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(
                """
                SELECT m.*, s.transcript, s.confidence
                FROM messages m
                LEFT JOIN stt_turns s ON m.message_id = s.message_id
                WHERE m.message_id = $1
                """,
                message_id
            )
            return dict(row) if row else None
        except Exception:
            logger.exception("get_message_by_id failed for message_id=%s", message_id)
            return None

async def get_user_details(pool: asyncpg.Pool, guild_id: str, user_id: str) -> dict | None:
    async with pool.acquire() as conn:
        try:
            user = await conn.fetchrow(
                "SELECT * FROM users WHERE guild_id = $1 AND user_id = $2",
                guild_id, user_id
            )
            if not user: return None
            
            messages = await conn.fetch(
                "SELECT message_id, content_clean, role, created_at FROM messages WHERE guild_id = $1 AND user_id = $2 ORDER BY created_at DESC LIMIT 50",
                guild_id, user_id
            )
            
            facts = await conn.fetch(
                "SELECT fact_id, fact_key, fact_value, status, importance FROM user_facts WHERE guild_id = $1 AND user_id = $2 ORDER BY updated_at DESC",
                guild_id, user_id
            )
            
            return {
                "user": dict(user),
                "messages": [dict(m) for m in messages],
                "facts": [dict(f) for f in facts]
            }
        except Exception:
            logger.exception("get_user_details failed for guild_id=%s user_id=%s", guild_id, user_id)
            return None

async def get_row_by_pk(pool: asyncpg.Pool, table: str, pk_col: str, pk_val: int | str) -> dict | None:
    # Basic protection
    valid_tables = ["persona_episodes", "persona_reflections"]
    if table not in valid_tables: return None
    
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(f"SELECT * FROM {table} WHERE {pk_col} = $1", pk_val)
            return dict(row) if row else None
        except Exception:
            logger.exception("get_row_by_pk failed for table=%s pk_col=%s pk_val=%s", table, pk_col, pk_val)
            return None

async def get_tables(pool: asyncpg.Pool) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.relname as table_name,
                   c.reltuples::bigint AS estimated_count
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'r'
              AND n.nspname = 'public'
            ORDER BY c.relname;
            """
        )
        tables = []
        for row in rows:
            t_name = row['table_name']
            count = row['estimated_count'] if row['estimated_count'] >= 0 else 0
            tables.append({"name": t_name, "count": count})
        return tables

async def get_table_columns(pool: asyncpg.Pool, table_name: str) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = $1
            ORDER BY ordinal_position;
            """,
            table_name
        )
        return [dict(row) for row in rows]

async def get_table_rows(pool: asyncpg.Pool, table_name: str, limit: int = 50, offset: int = 0) -> list[dict]:
    # Security: table_name must be validated before calling this
    limit, offset = _normalize_limit_offset(limit, offset, default_limit=50, max_limit=10000)
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(
                f"SELECT * FROM {table_name} LIMIT $1 OFFSET $2",
                limit, offset
            )
            return [dict(row) for row in rows]
        except Exception:
            logger.exception("get_table_rows failed for table=%s limit=%s offset=%s", table_name, limit, offset)
            return []

async def execute_readonly_query(pool: asyncpg.Pool, query: str) -> dict:
    clean_query = query.strip()
    
    # Strip basic block/line comments for regex checks
    comment_stripped = re.sub(r'/\*.*?\*/', '', clean_query, flags=re.DOTALL)
    comment_stripped = re.sub(r'--.*?\n', '', f"{comment_stripped}\n")
    upper_query = comment_stripped.upper()

    first_word = upper_query.strip().split()[0] if upper_query.strip() else ""
    
    if first_word not in ["SELECT", "WITH", "EXPLAIN"]:
        return {
            "success": False,
            "error": f"Security Policy blocks this statement. Must start with SELECT, WITH, or EXPLAIN.",
            "execution_time_ms": 0, "columns": [], "rows": []
        }

    # Explicit blocked keywords that could mutate state implicitly or bypass readonly
    blocked_keywords = [
        "ANALYZE", "INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "TRUNCATE ", "COMMIT", "ROLLBACK", "GRANT", "REVOKE", "CREATE ", "EXEC ", "EXECUTE "
    ]
    for b_kw in blocked_keywords:
        if b_kw in upper_query:
            return {
                "success": False,
                "error": f"Security Policy blocks the keyword '{b_kw}' to prevent mutations.",
                "execution_time_ms": 0, "columns": [], "rows": []
            }
            
    # Block multiple statements exactly
    if ";" in comment_stripped.strip().rstrip(";"):
         return {
            "success": False,
            "error": "Single statements only. Chained queries (;) are not allowed.",
            "execution_time_ms": 0, "columns": [], "rows": []
        }
    
    # Enforce a hard cap of 500 rows even if the user explicitly provided a large limit.
    # EXPLAIN cannot be wrapped as a subquery, so we skip wrapping it.
    if first_word in ["SELECT", "WITH"]:
        clean_query = f"SELECT * FROM ({clean_query.rstrip(';')}) AS __wrapped_query__ LIMIT 500"

    start_time = time.perf_counter()
    async with pool.acquire() as conn:
        try:
            async with conn.transaction():
                await conn.execute("SET TRANSACTION READ ONLY;")
                # Enforce server-side 5s timeout
                await conn.execute("SET LOCAL statement_timeout = '5000';")
                
                stmt = await conn.prepare(clean_query)
                records = await stmt.fetch(timeout=5.0)
                
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                
                if not records:
                    return {
                        "success": True, "execution_time_ms": execution_time_ms,
                        "columns": [], "rows": []
                    }
                    
                columns = list(records[0].keys())
                rows = [dict(record) for record in records]
                
                return {
                    "success": True,
                    "execution_time_ms": execution_time_ms,
                    "columns": columns,
                    "rows": rows,
                    "truncated": False,
                    "total_fetched": len(records)
                }
                
        except asyncpg.exceptions.QueryCanceledError:
             logger.warning("execute_readonly_query timed out")
             return {
                "success": False, "error": "Query timed out after 5 seconds.",
                "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                "columns": [], "rows": []
            }
        except Exception as e:
            logger.exception("execute_readonly_query failed")
            return {
                "success": False, "error": str(e),
                "execution_time_ms": (time.perf_counter() - start_time) * 1000,
                "columns": [], "rows": []
            }


async def delete_user(pool: asyncpg.Pool, guild_id: str, user_id: str) -> bool:
    """Deletes a specific user and all their associated data."""
    tables_guild_user = [
        "stt_turns",
        "user_facts", 
        "dialogue_summaries",
        "persona_relationships",
        "persona_ingested_messages",
        "persona_relationship_evidence",
        "persona_user_memory_prefs",
        "persona_episode_participants",
        "messages"
    ]
    async with pool.acquire() as conn:
        try:
            async with conn.transaction():
                for table in tables_guild_user:
                    try:
                        await conn.execute(f"DELETE FROM {table} WHERE guild_id = $1 AND user_id = $2", guild_id, user_id)
                    except asyncpg.exceptions.UndefinedTableError:
                        logger.debug("delete_user: table missing and skipped (%s)", table)
                        pass
                
                try:
                    await conn.execute("DELETE FROM persona_episode_evidence WHERE user_id = $1", user_id)
                except asyncpg.exceptions.UndefinedTableError:
                    logger.debug("delete_user: persona_episode_evidence table missing")
                    pass

                result = await conn.execute("""
                    DELETE FROM users
                    WHERE guild_id = $1 AND user_id = $2
                """, guild_id, user_id)
            return result.startswith("DELETE") and not result.endswith(" 0")
        except Exception:
            logger.exception("Error executing complete user delete for guild_id=%s user_id=%s", guild_id, user_id)
            return False

async def delete_fact(pool: asyncpg.Pool, guild_id: str, user_id: str, fact_id: int) -> bool:
    """Deletes a specific user fact."""
    async with pool.acquire() as conn:
        try:
            result = await conn.execute("""
                DELETE FROM user_facts
                WHERE guild_id = $1 AND user_id = $2 AND fact_id = $3
            """, guild_id, user_id, fact_id)
            return result.startswith("DELETE") and not result.endswith(" 0")
        except Exception:
            logger.exception(
                "delete_fact failed for guild_id=%s user_id=%s fact_id=%s",
                guild_id,
                user_id,
                fact_id,
            )
            return False
