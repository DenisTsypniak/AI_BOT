from __future__ import annotations

from typing import Dict, List

import aiosqlite

from .utils import _clamp


class MemoryFactsMixin:
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

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT fact_id, confidence, importance, evidence_count, status, pinned
                FROM user_facts
                WHERE guild_id = ? AND user_id = ? AND fact_key = ?
                """,
                (guild_id, user_id, key),
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                status = "confirmed" if conf >= 0.78 else "candidate"
                cursor = await db.execute(
                    """
                    INSERT INTO user_facts (
                        guild_id, user_id, fact_key, fact_value, fact_type,
                        confidence, importance, status, pinned, evidence_count,
                        created_at, updated_at, last_seen_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (
                        guild_id,
                        user_id,
                        key,
                        value[:280],
                        fact_type_clean,
                        conf,
                        imp,
                        status,
                    ),
                )
                fact_id = int(cursor.lastrowid)
            else:
                fact_id = int(row["fact_id"])
                prior_conf = float(row["confidence"])
                prior_imp = float(row["importance"])
                prior_count = int(row["evidence_count"])
                prior_status = str(row["status"])
                pinned = int(row["pinned"])

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

                await db.execute(
                    """
                    UPDATE user_facts
                    SET fact_value = ?,
                        fact_type = ?,
                        confidence = ?,
                        importance = ?,
                        status = ?,
                        evidence_count = ?,
                        updated_at = CURRENT_TIMESTAMP,
                        last_seen_at = CURRENT_TIMESTAMP
                    WHERE fact_id = ?
                    """,
                    (
                        value[:280],
                        fact_type_clean,
                        merged_conf,
                        merged_imp,
                        merged_status,
                        merged_count,
                        fact_id,
                    ),
                )

            await db.execute(
                """
                INSERT INTO fact_evidence (fact_id, message_id, extractor, confidence, created_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (fact_id, message_id, extractor, conf),
            )

            async with db.execute(
                """
                SELECT global_fact_id, confidence, importance, evidence_count, status, pinned
                FROM global_user_facts
                WHERE user_id = ? AND fact_key = ?
                """,
                (user_id, key),
            ) as cursor:
                global_row = await cursor.fetchone()

            if global_row is None:
                global_status = "confirmed" if conf >= 0.78 else "candidate"
                cursor = await db.execute(
                    """
                    INSERT INTO global_user_facts (
                        user_id, fact_key, fact_value, fact_type,
                        confidence, importance, status, pinned, evidence_count,
                        first_seen_at, updated_at, last_seen_at,
                        last_source_guild_id, last_source_message_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)
                    """,
                    (
                        user_id,
                        key,
                        value[:280],
                        fact_type_clean,
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
                g_prior_conf = float(global_row["confidence"])
                g_prior_imp = float(global_row["importance"])
                g_prior_count = int(global_row["evidence_count"])
                g_prior_status = str(global_row["status"])
                g_pinned = int(global_row["pinned"])

                g_merged_conf = _clamp(max(g_prior_conf * 0.86, conf), 0.0, 1.0)
                g_merged_imp = _clamp(max(g_prior_imp, imp), 0.0, 1.0)
                g_merged_count = g_prior_count + 1

                if g_pinned:
                    g_merged_status = "pinned"
                elif g_merged_count >= 2 or g_merged_conf >= 0.78:
                    g_merged_status = "confirmed"
                elif g_prior_status == "confirmed":
                    g_merged_status = "confirmed"
                else:
                    g_merged_status = "candidate"

                await db.execute(
                    """
                    UPDATE global_user_facts
                    SET fact_value = ?,
                        fact_type = ?,
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
                        value[:280],
                        fact_type_clean,
                        g_merged_conf,
                        g_merged_imp,
                        g_merged_status,
                        g_merged_count,
                        guild_id,
                        message_id,
                        global_fact_id,
                    ),
                )

            await db.execute(
                """
                INSERT INTO global_fact_evidence (
                    global_fact_id, source_guild_id, source_message_id, extractor, confidence, created_at
                )
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (global_fact_id, guild_id, message_id, extractor, conf),
            )

            await db.commit()
            return fact_id

    async def get_user_facts(self, guild_id: str, user_id: str, limit: int) -> List[Dict[str, object]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT fact_id, fact_key, fact_value, fact_type, confidence, importance,
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
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT global_fact_id, fact_key, fact_value, fact_type, confidence, importance,
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
                """,
                (user_id, max(1, int(limit))),
            ) as cursor:
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
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if exclude_guild_id:
                query = """
                    SELECT fact_id, guild_id, fact_key, fact_value, fact_type, confidence, importance,
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
