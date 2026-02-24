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
