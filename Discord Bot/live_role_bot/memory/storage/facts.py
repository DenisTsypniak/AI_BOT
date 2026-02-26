from __future__ import annotations

import logging
from typing import Dict, List

import aiosqlite

from .utils import (
    _clamp,
    _sqlite_memory_connection,
    apply_memory_fact_promotion_policy,
    merge_memory_fact_metadata_state,
    merge_memory_fact_state,
    normalize_memory_fact_about_target,
    normalize_memory_fact_directness,
    sanitize_memory_fact_evidence_quote,
)


logger = logging.getLogger("live_role_bot")


class MemoryFactsMixin:
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
        async with _sqlite_memory_connection(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO memory_extractor_runs (
                    guild_id, channel_id, speaker_user_id, fact_owner_kind, fact_owner_id, speaker_role,
                    modality, source, backend_name, model_name, dry_run, llm_attempted, llm_ok, json_valid,
                    fallback_used, latency_ms, candidate_count, accepted_count, saved_count, filtered_count, error_text,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
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
                    1 if dry_run else 0,
                    1 if llm_attempted else 0,
                    1 if llm_ok else 0,
                    1 if json_valid else 0,
                    1 if fallback_used else 0,
                    max(0, int(latency_ms)),
                    max(0, int(candidate_count)),
                    max(0, int(accepted_count)),
                    max(0, int(saved_count)),
                    max(0, int(filtered_count)),
                    str(error_text or "")[:400],
                ),
            )
            run_id = int(cursor.lastrowid)

            for row in list(candidates or []):
                if not isinstance(row, dict):
                    continue
                await db.execute(
                    """
                    INSERT INTO memory_extractor_candidates (
                        run_id, fact_key, fact_value, fact_type, about_target, directness, evidence_quote,
                        confidence, importance, moderation_action, moderation_reason, selected_for_apply, saved_to_memory,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
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
                        1 if bool(row.get("selected_for_apply")) else 0,
                        1 if bool(row.get("saved_to_memory")) else 0,
                    ),
                )

            await db.commit()
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

        async with _sqlite_memory_connection(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT fact_id, fact_value, fact_type, confidence, importance, evidence_count, status, pinned,
                       COALESCE(about_target, 'self') AS about_target,
                       COALESCE(directness, 'explicit') AS directness,
                       COALESCE(evidence_quote, '') AS evidence_quote
                FROM user_facts
                WHERE guild_id = ? AND user_id = ? AND fact_key = ?
                """,
                (guild_id, user_id, key),
            ) as cursor:
                row = await cursor.fetchone()

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
                cursor = await db.execute(
                    """
                    INSERT INTO user_facts (
                        guild_id, user_id, fact_key, fact_value, fact_type,
                        about_target, directness, evidence_quote,
                        confidence, importance, status, pinned, evidence_count,
                        created_at, updated_at, last_seen_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (
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
                    ),
                )
                fact_id = int(cursor.lastrowid)
            else:
                fact_id = int(row["fact_id"])
                merge = merge_memory_fact_state(
                    prior_value=str(row["fact_value"] or ""),
                    prior_fact_type=str(row["fact_type"] or "fact"),
                    prior_confidence=float(row["confidence"]),
                    prior_importance=float(row["importance"]),
                    prior_evidence_count=int(row["evidence_count"]),
                    prior_status=str(row["status"]),
                    pinned=bool(int(row["pinned"])),
                    incoming_value=value,
                    incoming_fact_type=fact_type_clean,
                    incoming_confidence=conf,
                    incoming_importance=imp,
                )
                if merge.value_conflict:
                    logger.debug(
                        "Memory fact conflict (sqlite local): guild=%s user=%s key=%s replaced=%s prior_conf=%.3f incoming_conf=%.3f",
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
                    pinned=bool(int(row["pinned"])),
                    confidence=merge.confidence,
                    importance=merge.importance,
                    evidence_count=merge.evidence_count,
                    directness=meta.directness,
                    value_conflict=merge.value_conflict,
                    value_replaced=merge.value_replaced,
                )

                await db.execute(
                    """
                    UPDATE user_facts
                    SET fact_value = ?,
                        fact_type = ?,
                        about_target = ?,
                        directness = ?,
                        evidence_quote = ?,
                        confidence = ?,
                        importance = ?,
                        status = ?,
                        evidence_count = ?,
                        updated_at = CURRENT_TIMESTAMP,
                        last_seen_at = CURRENT_TIMESTAMP
                    WHERE fact_id = ?
                    """,
                    (
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
                    ),
                )

            await db.execute(
                """
                INSERT INTO fact_evidence (
                    fact_id, message_id, extractor, about_target, directness, evidence_quote, confidence, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (fact_id, message_id, extractor, about_target_clean, directness_clean, evidence_quote_clean, conf),
            )

            async with db.execute(
                """
                SELECT global_fact_id, fact_value, fact_type, confidence, importance, evidence_count, status, pinned,
                       COALESCE(about_target, 'self') AS about_target,
                       COALESCE(directness, 'explicit') AS directness,
                       COALESCE(evidence_quote, '') AS evidence_quote
                FROM global_user_facts
                WHERE user_id = ? AND fact_key = ?
                """,
                (user_id, key),
            ) as cursor:
                global_row = await cursor.fetchone()

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
                cursor = await db.execute(
                    """
                    INSERT INTO global_user_facts (
                        user_id, fact_key, fact_value, fact_type,
                        about_target, directness, evidence_quote,
                        confidence, importance, status, pinned, evidence_count,
                        first_seen_at, updated_at, last_seen_at,
                        last_source_guild_id, last_source_message_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)
                    """,
                    (
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
                    ),
                )
                global_fact_id = int(cursor.lastrowid)
            else:
                global_fact_id = int(global_row["global_fact_id"])
                g_merge = merge_memory_fact_state(
                    prior_value=str(global_row["fact_value"] or ""),
                    prior_fact_type=str(global_row["fact_type"] or "fact"),
                    prior_confidence=float(global_row["confidence"]),
                    prior_importance=float(global_row["importance"]),
                    prior_evidence_count=int(global_row["evidence_count"]),
                    prior_status=str(global_row["status"]),
                    pinned=bool(int(global_row["pinned"])),
                    incoming_value=value,
                    incoming_fact_type=fact_type_clean,
                    incoming_confidence=conf,
                    incoming_importance=imp,
                )
                if g_merge.value_conflict:
                    logger.debug(
                        "Memory fact conflict (sqlite global): user=%s key=%s source_guild=%s replaced=%s prior_conf=%.3f incoming_conf=%.3f",
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
                    pinned=bool(int(global_row["pinned"])),
                    confidence=g_merge.confidence,
                    importance=g_merge.importance,
                    evidence_count=g_merge.evidence_count,
                    directness=g_meta.directness,
                    value_conflict=g_merge.value_conflict,
                    value_replaced=g_merge.value_replaced,
                )

                await db.execute(
                    """
                    UPDATE global_user_facts
                    SET fact_value = ?,
                        fact_type = ?,
                        about_target = ?,
                        directness = ?,
                        evidence_quote = ?,
                        confidence = ?,
                        importance = ?,
                        status = ?,
                        evidence_count = ?,
                        updated_at = CURRENT_TIMESTAMP,
                        last_seen_at = CURRENT_TIMESTAMP,
                        last_source_guild_id = ?,
                        last_source_message_id = ?
                    WHERE global_fact_id = ?
                    """,
                    (
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
                    ),
                )

            await db.execute(
                """
                INSERT INTO global_fact_evidence (
                    global_fact_id, source_guild_id, source_message_id, extractor,
                    about_target, directness, evidence_quote, confidence, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    global_fact_id,
                    guild_id,
                    message_id,
                    extractor,
                    about_target_clean,
                    directness_clean,
                    evidence_quote_clean,
                    conf,
                ),
            )

            await db.commit()
            return fact_id

    async def get_user_facts(self, guild_id: str, user_id: str, limit: int) -> List[Dict[str, object]]:
        async with _sqlite_memory_connection(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT fact_id, fact_key, fact_value, fact_type, confidence, importance,
                       COALESCE(about_target, 'self') AS about_target,
                       COALESCE(directness, 'explicit') AS directness,
                       COALESCE(evidence_quote, '') AS evidence_quote,
                       status, pinned, evidence_count, updated_at, last_seen_at
                FROM user_facts
                WHERE guild_id = ? AND user_id = ?
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
                LIMIT ?
                """,
                (guild_id, user_id, max(1, int(limit))),
            ) as cursor:
                rows = await cursor.fetchall()

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
                "pinned": bool(int(row["pinned"])),
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
        async with _sqlite_memory_connection(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if exclude_guild_id:
                query = """
                    SELECT guf.global_fact_id, guf.fact_key, guf.fact_value, guf.fact_type, guf.confidence, guf.importance,
                           COALESCE(guf.about_target, 'self') AS about_target,
                           COALESCE(guf.directness, 'explicit') AS directness,
                           COALESCE(guf.evidence_quote, '') AS evidence_quote,
                           guf.status, guf.pinned, guf.evidence_count, guf.updated_at, guf.last_seen_at, guf.last_source_guild_id
                    FROM global_user_facts AS guf
                    WHERE guf.user_id = ?
                      AND (
                            guf.last_source_guild_id IS NULL
                         OR guf.last_source_guild_id <> ?
                         OR EXISTS (
                                SELECT 1
                                FROM global_fact_evidence AS gfe
                                WHERE gfe.global_fact_id = guf.global_fact_id
                                  AND gfe.source_guild_id IS NOT NULL
                                  AND gfe.source_guild_id <> ?
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
                    LIMIT ?
                """
                params = (user_id, exclude_guild_id, exclude_guild_id, max(1, int(limit)))
            else:
                query = """
                    SELECT global_fact_id, fact_key, fact_value, fact_type, confidence, importance,
                           COALESCE(about_target, 'self') AS about_target,
                           COALESCE(directness, 'explicit') AS directness,
                           COALESCE(evidence_quote, '') AS evidence_quote,
                           status, pinned, evidence_count, updated_at, last_seen_at, last_source_guild_id
                    FROM global_user_facts
                    WHERE user_id = ?
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
                    LIMIT ?
                """
                params = (user_id, max(1, int(limit)))
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

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
                    "pinned": bool(int(row["pinned"])),
                    "evidence_count": int(row["evidence_count"]),
                    "updated_at": str(row["updated_at"]),
                    "last_seen_at": str(row["last_seen_at"]),
                    "global": True,
                }
                for row in rows
            ]

        # Legacy fallback if global tables are still empty
        async with _sqlite_memory_connection(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if exclude_guild_id:
                query = """
                    SELECT fact_id, guild_id, fact_key, fact_value, fact_type, confidence, importance,
                           COALESCE(about_target, 'self') AS about_target,
                           COALESCE(directness, 'explicit') AS directness,
                           COALESCE(evidence_quote, '') AS evidence_quote,
                           status, pinned, evidence_count, updated_at, last_seen_at
                    FROM user_facts
                    WHERE user_id = ? AND guild_id <> ?
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
                    LIMIT ?
                """
                params = (user_id, exclude_guild_id, max(1, int(limit) * 4))
            else:
                query = """
                    SELECT fact_id, guild_id, fact_key, fact_value, fact_type, confidence, importance,
                           COALESCE(about_target, 'self') AS about_target,
                           COALESCE(directness, 'explicit') AS directness,
                           COALESCE(evidence_quote, '') AS evidence_quote,
                           status, pinned, evidence_count, updated_at, last_seen_at
                    FROM user_facts
                    WHERE user_id = ?
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
                    LIMIT ?
                """
                params = (user_id, max(1, int(limit) * 4))
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

        deduped: List[Dict[str, object]] = []
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
                    "pinned": bool(int(row["pinned"])),
                    "evidence_count": int(row["evidence_count"]),
                    "updated_at": str(row["updated_at"]),
                    "last_seen_at": str(row["last_seen_at"]),
                }
            )
            if len(deduped) >= max(1, int(limit)):
                break
        return deduped
