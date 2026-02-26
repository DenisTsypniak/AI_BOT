from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

try:
    import aiosqlite
except Exception:  # pragma: no cover - optional in Postgres-only deployments
    aiosqlite = None  # type: ignore[assignment]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sqlite_busy_timeout_ms() -> int:
    raw = os.getenv("MEMORY_SQLITE_BUSY_TIMEOUT_MS", "5000").strip()
    try:
        timeout = int(raw)
    except ValueError:
        timeout = 5000
    return max(0, min(timeout, 60000))


@asynccontextmanager
async def _sqlite_memory_connection(db_path: str | Path) -> AsyncIterator["aiosqlite.Connection"]:
    if aiosqlite is None:
        raise RuntimeError("SQLite memory backend requires aiosqlite")
    async with aiosqlite.connect(db_path) as db:  # type: ignore[union-attr]
        await db.execute("PRAGMA foreign_keys=ON")
        timeout_ms = _sqlite_busy_timeout_ms()
        if timeout_ms > 0:
            await db.execute(f"PRAGMA busy_timeout={timeout_ms}")
        yield db


def _normalize_fact_value(value: str) -> str:
    return " ".join((value or "").strip().split()).casefold()


_DIRECTNESS_RANK = {
    "inferred": 1,
    "implicit": 2,
    "explicit": 3,
}


def normalize_memory_fact_directness(value: str, *, default: str = "explicit") -> str:
    raw = str(value or "").strip().casefold()
    if not raw:
        raw = default
    aliases = {
        "direct": "explicit",
        "stated": "explicit",
        "explicitly_stated": "explicit",
        "implied": "implicit",
        "hinted": "implicit",
        "soft_inference": "implicit",
        "deduced": "inferred",
        "guess": "inferred",
    }
    normalized = aliases.get(raw, raw)
    return normalized if normalized in _DIRECTNESS_RANK else default


def memory_fact_directness_rank(value: str) -> int:
    return _DIRECTNESS_RANK.get(normalize_memory_fact_directness(value), _DIRECTNESS_RANK["explicit"])


def normalize_memory_fact_about_target(value: str, *, default: str = "self") -> str:
    raw = str(value or "").strip().casefold()
    if not raw:
        raw = default
    aliases = {
        "speaker": "self",
        "user": "self",
        "user_self": "self",
        "self_user": "self",
        "me": "self",
        "myself": "self",
        "assistant": "assistant_self",
        "persona": "assistant_self",
        "assistant-persona": "assistant_self",
        "assistant_self_claim": "assistant_self",
        "bot": "assistant_self",
        "another_person": "other",
    }
    normalized = aliases.get(raw, raw)
    allowed = {"self", "assistant_self", "other", "unknown"}
    return normalized if normalized in allowed else default


def sanitize_memory_fact_evidence_quote(value: str, *, max_chars: int = 220) -> str:
    cleaned = " ".join(str(value or "").strip().split())
    return cleaned[: max(1, int(max_chars))]


@dataclass(slots=True)
class MemoryFactMergeResult:
    fact_value: str
    fact_type: str
    confidence: float
    importance: float
    status: str
    evidence_count: int
    value_conflict: bool
    value_replaced: bool
    same_value: bool


@dataclass(slots=True)
class MemoryFactMetadataMergeResult:
    about_target: str
    directness: str
    evidence_quote: str


def merge_memory_fact_state(
    *,
    prior_value: str,
    prior_fact_type: str,
    prior_confidence: float,
    prior_importance: float,
    prior_evidence_count: int,
    prior_status: str,
    pinned: bool,
    incoming_value: str,
    incoming_fact_type: str,
    incoming_confidence: float,
    incoming_importance: float,
    value_max_chars: int = 280,
) -> MemoryFactMergeResult:
    prior_value_clean = str(prior_value or "").strip()
    incoming_value_clean = str(incoming_value or "").strip()
    prior_type_clean = str(prior_fact_type or "").strip().lower() or "fact"
    incoming_type_clean = str(incoming_fact_type or "").strip().lower() or "fact"

    prior_conf = _clamp(float(prior_confidence), 0.0, 1.0)
    incoming_conf = _clamp(float(incoming_confidence), 0.0, 1.0)
    prior_imp = _clamp(float(prior_importance), 0.0, 1.0)
    incoming_imp = _clamp(float(incoming_importance), 0.0, 1.0)
    prior_count = max(0, int(prior_evidence_count))
    prior_status_clean = str(prior_status or "candidate").strip().lower() or "candidate"

    prior_norm = _normalize_fact_value(prior_value_clean)
    incoming_norm = _normalize_fact_value(incoming_value_clean)
    same_value = bool(prior_norm and incoming_norm and prior_norm == incoming_norm)
    value_conflict = bool(prior_norm and incoming_norm and prior_norm != incoming_norm)

    replace_value = same_value
    if value_conflict and not pinned:
        prior_is_strong = prior_status_clean in {"confirmed", "pinned"} or prior_conf >= 0.78
        if incoming_conf >= prior_conf + 0.18:
            replace_value = True
        elif (not prior_is_strong) and incoming_conf >= max(0.55, prior_conf + 0.05):
            replace_value = True
        elif prior_conf < 0.45 and incoming_conf >= 0.55:
            replace_value = True

    next_value = incoming_value_clean if replace_value else (prior_value_clean or incoming_value_clean)
    next_type = incoming_type_clean if replace_value else (prior_type_clean or incoming_type_clean)
    next_value = next_value[: max(1, int(value_max_chars))]

    if same_value:
        next_conf = _clamp(max(prior_conf * 0.86, incoming_conf), 0.0, 1.0)
    elif replace_value:
        next_conf = _clamp(max(prior_conf * 0.72, incoming_conf), 0.0, 1.0)
    elif value_conflict:
        next_conf = _clamp(max(prior_conf * 0.94, incoming_conf * 0.65), 0.0, 1.0)
    else:
        next_conf = _clamp(max(prior_conf * 0.86, incoming_conf), 0.0, 1.0)

    next_imp = _clamp(max(prior_imp, incoming_imp), 0.0, 1.0)
    next_count = prior_count + 1

    if pinned:
        next_status = "pinned"
    elif prior_status_clean == "confirmed" and value_conflict and not replace_value and next_conf >= 0.55:
        next_status = "confirmed"
    elif next_conf >= 0.78:
        next_status = "confirmed"
    elif next_count >= 2 and next_conf >= 0.70:
        next_status = "confirmed"
    else:
        next_status = "candidate"

    return MemoryFactMergeResult(
        fact_value=next_value,
        fact_type=next_type,
        confidence=next_conf,
        importance=next_imp,
        status=next_status,
        evidence_count=next_count,
        value_conflict=value_conflict,
        value_replaced=replace_value and value_conflict,
        same_value=same_value,
    )


def merge_memory_fact_metadata_state(
    *,
    prior_value: str,
    incoming_value: str,
    prior_about_target: str,
    incoming_about_target: str,
    prior_directness: str,
    incoming_directness: str,
    prior_evidence_quote: str,
    incoming_evidence_quote: str,
    value_conflict: bool,
    value_replaced: bool,
) -> MemoryFactMetadataMergeResult:
    prior_target = normalize_memory_fact_about_target(prior_about_target, default="self")
    incoming_target = normalize_memory_fact_about_target(incoming_about_target, default="self")
    prior_dir = normalize_memory_fact_directness(prior_directness, default="explicit")
    incoming_dir = normalize_memory_fact_directness(incoming_directness, default="explicit")
    prior_quote = sanitize_memory_fact_evidence_quote(prior_evidence_quote)
    incoming_quote = sanitize_memory_fact_evidence_quote(incoming_evidence_quote)

    prior_norm = _normalize_fact_value(prior_value)
    incoming_norm = _normalize_fact_value(incoming_value)
    same_value = bool(prior_norm and incoming_norm and prior_norm == incoming_norm)

    if not prior_norm:
        return MemoryFactMetadataMergeResult(
            about_target=incoming_target,
            directness=incoming_dir,
            evidence_quote=incoming_quote,
        )

    if value_conflict and not value_replaced:
        # Keep the currently selected fact metadata when we reject a conflicting value.
        return MemoryFactMetadataMergeResult(
            about_target=prior_target,
            directness=prior_dir,
            evidence_quote=prior_quote,
        )

    if value_replaced:
        return MemoryFactMetadataMergeResult(
            about_target=incoming_target,
            directness=incoming_dir,
            evidence_quote=incoming_quote or prior_quote,
        )

    if same_value:
        target = prior_target
        if prior_target == "unknown" and incoming_target != "unknown":
            target = incoming_target
        directness = prior_dir
        if memory_fact_directness_rank(incoming_dir) > memory_fact_directness_rank(prior_dir):
            directness = incoming_dir
        quote = incoming_quote or prior_quote
        return MemoryFactMetadataMergeResult(
            about_target=target,
            directness=directness,
            evidence_quote=quote,
        )

    # Non-conflicting updates with empty prior or unusual normalization edge cases.
    return MemoryFactMetadataMergeResult(
        about_target=incoming_target if incoming_target != "unknown" else prior_target,
        directness=incoming_dir,
        evidence_quote=incoming_quote or prior_quote,
    )


def apply_memory_fact_promotion_policy(
    *,
    current_status: str,
    prior_status: str,
    pinned: bool,
    confidence: float,
    importance: float,
    evidence_count: int,
    directness: str,
    value_conflict: bool,
    value_replaced: bool,
) -> str:
    if pinned:
        return "pinned"

    cur = str(current_status or "candidate").strip().lower() or "candidate"
    prev = str(prior_status or "candidate").strip().lower() or "candidate"
    conf = _clamp(float(confidence), 0.0, 1.0)
    imp = _clamp(float(importance), 0.0, 1.0)
    count = max(0, int(evidence_count))
    direct = normalize_memory_fact_directness(directness, default="explicit")

    # Preserve confirmed facts when we reject a conflicting weaker alternative.
    if prev in {"confirmed", "pinned"} and value_conflict and not value_replaced and conf >= 0.50:
        return "confirmed"

    if direct == "explicit":
        if conf >= 0.78:
            return "confirmed"
        if count >= 2 and conf >= 0.68:
            return "confirmed"
        if count >= 3 and conf >= 0.62 and imp >= 0.55:
            return "confirmed"
        return "candidate"

    if direct == "implicit":
        if count >= 2 and conf >= 0.72 and imp >= 0.50:
            return "confirmed"
        if count >= 3 and conf >= 0.66:
            return "confirmed"
        return "candidate"

    # inferred
    if count >= 3 and conf >= 0.78 and imp >= 0.50:
        return "confirmed"
    if count >= 4 and conf >= 0.70 and imp >= 0.45:
        return "confirmed"

    # Fall back to the current status only if it is stronger and no unresolved conflict occurred.
    if cur == "confirmed" and not value_conflict:
        return "confirmed"
    return "candidate"
