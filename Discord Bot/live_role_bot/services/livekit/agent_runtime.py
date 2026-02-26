from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import re
import site
import sys
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


def _bootstrap_local_venv() -> None:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    py_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    paths = [
        os.path.join(root, ".venv", "Lib", "site-packages"),
        os.path.join(root, ".venv", "lib", py_dir, "site-packages"),
    ]
    for path in paths:
        if os.path.isdir(path) and path not in sys.path:
            site.addsitedir(path)


_bootstrap_local_venv()

from ...config import Settings
from ...discord.common import collapse_spaces, load_rp_canon
from ...memory.factory import build_memory_store
from ...memory.extractor import MemoryExtractor
from ...memory.fact_moderation import CandidateModerationInput, FactModerationPolicyV2
from ...memory.storage.utils import (
    normalize_memory_fact_about_target,
    normalize_memory_fact_directness,
    sanitize_memory_fact_evidence_quote,
)
from ...prompts.dialogue import (
    build_persona_biography_summary_line,
    build_persona_relevant_facts_section,
    build_relevant_facts_section,
    build_user_biography_summary_line,
    build_user_dialogue_summary_line,
)
from ...prompts.voice import build_native_audio_system_instruction
from ...services.gemini_client import GeminiClient
from .bridge import DiscordLiveKitBridge, LiveKitBridgeConfig
from .config import LiveKitAgentSettings
from .observability import HealthHeartbeat, LiveKitRuntimeHealth

logger = logging.getLogger("live_role_bot.livekit")


class _DropNoisyRootMessages(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple log filter
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if "ignoring text stream with topic" in msg and "no callback attached" in msg:
            return False
        return True


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    root_logger = logging.getLogger()
    has_noise_filter = any(isinstance(f, _DropNoisyRootMessages) for f in getattr(root_logger, "filters", []))
    if not has_noise_filter:
        root_logger.addFilter(_DropNoisyRootMessages())
    logging.getLogger("livekit").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


@dataclass(slots=True)
class _PromptContext:
    system_core_prompt: str
    preferred_language: str
    role_name: str
    role_goal: str
    role_style: str
    role_constraints: str
    rp_canon_prompt: str


@dataclass(slots=True)
class _RuntimeDeps:
    base_settings: Settings
    lk_settings: "LiveKitAgentSettings"
    instructions: str
    google: Any
    silero: Any | None
    health: "LiveKitRuntimeHealth"
    backfill_extractor: MemoryExtractor | None
    backfill_llm: GeminiClient | None


@dataclass(slots=True)
class _BridgeRuntimeContextState:
    latest_packet: dict[str, Any] | None = None
    last_seq: int = 0
    last_received_at: float = 0.0
    last_applied_hash: str = ""
    last_applied_chars: int = 0
    updates_received: int = 0
    updates_applied: int = 0
    updates_ignored: int = 0
    updates_errors: int = 0
    ack_packets_seen: int = 0
    last_apply_error: str = ""
    last_applied_at: float = 0.0
    last_received_reason: str = ""
    last_applied_reason: str = ""
    memory_updates_applied: int = 0
    memory_updates_ignored: int = 0
    memory_updates_errors: int = 0
    last_memory_error: str = ""
    last_memory_focus_user_id: str = ""
    last_memory_focus_label: str = ""
    last_memory_hash: str = ""
    memory_backfill_attempts: int = 0
    memory_backfill_saved: int = 0
    memory_backfill_skipped: int = 0


def _persona_memory_subject_user_id(base_settings: Settings) -> str:
    persona_id = str(getattr(base_settings, "persona_id", "") or "persona").strip() or "persona"
    return f"persona::{persona_id}"


def _parse_bridge_room_guild_id(room_name: str, room_prefix: str) -> str:
    room = str(room_name or "").strip()
    prefix = str(room_prefix or "").strip()
    if not room or not prefix:
        return ""
    match = re.match(rf"^{re.escape(prefix)}-g(\d+)-v\d+$", room)
    if match is None:
        return ""
    return str(match.group(1) or "")


def _build_prompt_context(base_settings: Settings, lk_settings: LiveKitAgentSettings) -> _PromptContext:
    rp_canon = load_rp_canon(lk_settings.bot_history_json_path)
    if not rp_canon.strip():
        raise RuntimeError(
            f"LiveKit agent requires a valid character file at {lk_settings.bot_history_json_path}. "
            "Ensure bot_history.json is present, enabled=true, and contains prompt text."
        )
    return _PromptContext(
        system_core_prompt=base_settings.system_core_prompt,
        preferred_language=base_settings.preferred_response_language,
        role_name=base_settings.role_name,
        role_goal=base_settings.role_goal,
        role_style=base_settings.role_style,
        role_constraints=base_settings.role_constraints,
        rp_canon_prompt=rp_canon,
    )


def _build_livekit_instructions(ctx: _PromptContext) -> str:
    lines = [
        "RP CANON (highest priority, character source):\n" + ctx.rp_canon_prompt.strip(),
        f"Primary language: {ctx.preferred_language}.",
        "Context: You are speaking in a realtime voice room. Keep replies speech-friendly and natural.",
        "Voice turn policy: default to 1-3 short sentences unless user explicitly asks for details.",
        "Voice turn policy: react first, then give one useful point, then optionally one short question.",
        "Voice turn policy: vary openers and avoid repeating the same phrasing across consecutive turns.",
        "Voice turn policy: if audio is unclear, ask for a repeat in a natural human way.",
        "Human continuity: remember the last few turns in this room and callback naturally when relevant.",
        "Human continuity: do not sound like each reply starts a brand-new conversation.",
        "Group voice behavior: if multiple people are talking, handle it casually and keep replies short/reactive.",
        "Group voice behavior: you may address the group briefly (for example 'народ' or 'чуваки') but do not spam it.",
        "Emotion matching: if someone sounds upset/serious, reduce memes and answer calmly/supportively.",
    ]

    base = "\n".join(part for part in lines if part).strip()
    return build_native_audio_system_instruction(base, ctx.preferred_language)


def _trim_runtime_text(value: object, limit: int = 80) -> str:
    text = collapse_spaces(str(value or ""))
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."


def _build_runtime_context_block(
    packet: dict[str, Any] | None,
    *,
    max_chars: int,
) -> str:
    if not isinstance(packet, dict):
        return ""
    payload = packet.get("payload", {}) if isinstance(packet.get("payload"), dict) else {}
    try:
        packet_seq = int(packet.get("seq", 0) or 0)
    except Exception:
        packet_seq = 0
    packet_reason = str(packet.get("reason") or "") if isinstance(packet, dict) else ""
    guild_obj = payload.get("guild", {}) if isinstance(payload.get("guild"), dict) else {}
    voice_obj = payload.get("voice_channel", {}) if isinstance(payload.get("voice_channel"), dict) else {}
    text_obj = payload.get("text_channel", {}) if isinstance(payload.get("text_channel"), dict) else {}
    participants_obj = payload.get("participants", {}) if isinstance(payload.get("participants"), dict) else {}
    bridge_obj = payload.get("bridge_runtime", {}) if isinstance(payload.get("bridge_runtime"), dict) else {}
    lk_obj = payload.get("livekit_room", {}) if isinstance(payload.get("livekit_room"), dict) else {}
    members = participants_obj.get("members", []) if isinstance(participants_obj.get("members"), list) else []
    active_speakers = (
        participants_obj.get("active_speaker_hints", [])
        if isinstance(participants_obj.get("active_speaker_hints"), list)
        else []
    )

    participant_names: list[str] = []
    for item in members[:8]:
        if not isinstance(item, dict):
            continue
        name = _trim_runtime_text(item.get("display_name"), 28)
        if not name:
            continue
        if bool(item.get("bot")):
            name = f"{name}[bot]"
        participant_names.append(name)
    participant_line = ", ".join(participant_names)
    if len(members) > len(participant_names):
        participant_line = (participant_line + (", " if participant_line else "")) + f"+{len(members) - len(participant_names)} more"

    lk_remote = lk_obj.get("remote_participants", []) if isinstance(lk_obj.get("remote_participants"), list) else []
    remote_identities: list[str] = []
    for item in lk_remote[:5]:
        if not isinstance(item, dict):
            continue
        ident = _trim_runtime_text(item.get("identity"), 32)
        if ident:
            remote_identities.append(ident)

    lines = [
        "RUNTIME DISCORD CONTEXT (bridge snapshot, lower priority than RP CANON):",
        f"- guild: {_trim_runtime_text(guild_obj.get('name'), 64)} ({guild_obj.get('id') or '?'})",
        f"- voice_channel: {_trim_runtime_text(voice_obj.get('name'), 64)} ({voice_obj.get('id') or '?'})",
        f"- text_channel: {_trim_runtime_text(text_obj.get('name'), 64)} ({text_obj.get('id') or '?'})",
        (
            f"- participants_now: {int(participants_obj.get('count', 0) or 0)} "
            f"(humans={int(participants_obj.get('humans', 0) or 0)}, bots={int(participants_obj.get('bots', 0) or 0)})"
        ),
    ]
    if participant_line:
        lines.append(f"- members: {participant_line}")
    if active_speakers:
        speaker_names = ", ".join(_trim_runtime_text(x, 26) for x in active_speakers[:6] if str(x or "").strip())
        if speaker_names:
            lines.append(f"- active_speakers: {speaker_names}")
    lines.append(
        (
            f"- bridge_runtime: ingress_active={'yes' if bridge_obj.get('ingress_active') else 'no'} "
            f"remote_streams={int(bridge_obj.get('remote_streams', 0) or 0)}"
        )
    )
    last_ingress_label = _trim_runtime_text(bridge_obj.get("last_ingress_user_label"), 28)
    last_ingress_user_id = _trim_runtime_text(bridge_obj.get("last_ingress_user_id"), 24)
    if last_ingress_label or last_ingress_user_id:
        lines.append(
            f"- recent_discord_speaker: {last_ingress_label or '?'} ({last_ingress_user_id or '?'})"
        )
    if remote_identities:
        lines.append(f"- livekit_remote: {', '.join(remote_identities)}")

    text = "\n".join(line for line in lines if line).strip()
    max_chars_safe = max(120, int(max_chars))
    if len(text) > max_chars_safe:
        text = text[: max_chars_safe - 3].rstrip() + "..."
    return text


def _context_payload(packet: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(packet, dict):
        return {}
    payload = packet.get("payload")
    return payload if isinstance(payload, dict) else {}


def _resolve_voice_memory_focuses(packet: dict[str, Any] | None, *, max_users: int = 3) -> list[dict[str, str]]:
    payload = _context_payload(packet)
    guild_obj = payload.get("guild", {}) if isinstance(payload.get("guild"), dict) else {}
    text_obj = payload.get("text_channel", {}) if isinstance(payload.get("text_channel"), dict) else {}
    participants_obj = payload.get("participants", {}) if isinstance(payload.get("participants"), dict) else {}
    bridge_obj = payload.get("bridge_runtime", {}) if isinstance(payload.get("bridge_runtime"), dict) else {}
    members = participants_obj.get("members", []) if isinstance(participants_obj.get("members"), list) else []

    guild_id = str(guild_obj.get("id") or packet.get("guild_id") or "").strip() if isinstance(packet, dict) else ""
    channel_id = str(text_obj.get("id") or packet.get("text_channel_id") or "").strip() if isinstance(packet, dict) else ""
    preferred_user_id = str(bridge_obj.get("last_ingress_user_id") or "").strip()
    preferred_label = _trim_runtime_text(bridge_obj.get("last_ingress_user_label") or "", 40)

    ordered_members: list[dict[str, Any]] = []
    seen_user_ids: set[str] = set()

    def _push_member(item: dict[str, Any] | None) -> None:
        if not isinstance(item, dict):
            return
        if bool(item.get("bot")):
            return
        user_id = str(item.get("user_id") or "").strip()
        if not user_id or user_id in seen_user_ids:
            return
        ordered_members.append(item)
        seen_user_ids.add(user_id)

    if preferred_user_id:
        for item in members:
            if not isinstance(item, dict):
                continue
            if str(item.get("user_id") or "").strip() == preferred_user_id:
                _push_member(item)
                break

    for item in members:
        _push_member(item if isinstance(item, dict) else None)
        if len(ordered_members) >= max(1, int(max_users)):
            break

    focuses: list[dict[str, str]] = []
    for item in ordered_members[: max(1, int(max_users))]:
        user_id = str(item.get("user_id") or "").strip()
        label = _trim_runtime_text(item.get("display_name") or "", 40)
        focuses.append(
            {
                "guild_id": guild_id,
                "channel_id": channel_id,
                "user_id": user_id,
                "label": label,
            }
        )

    # Fallback when participant snapshot is sparse but bridge knows the last ingress speaker.
    if not focuses and preferred_user_id:
        focuses.append(
            {
                "guild_id": guild_id,
                "channel_id": channel_id,
                "user_id": preferred_user_id,
                "label": preferred_label,
            }
        )
    return focuses


def _select_voice_memory_facts(base_settings: Settings, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not facts:
        return []
    def _priority(fact: dict[str, Any]) -> tuple[int, float, float]:
        try:
            fact_type = str(fact.get("fact_type") or "")
            status = str(fact.get("status") or "")
            confidence = float(fact.get("confidence") or 0.0)
            importance = float(fact.get("importance") or 0.0)
        except Exception:
            return (0, 0.0, 0.0)
        tier = 0
        if fact_type == "identity":
            tier = 3
        elif fact_type == "preference":
            tier = 2
        elif status == "confirmed":
            tier = 1
        return (tier, confidence, importance)

    keep: list[dict[str, Any]] = []
    limit = max(2, min(6, int(getattr(base_settings, "memory_fact_top_k", 4) or 4)))
    for fact in sorted(facts, key=_priority, reverse=True):
        try:
            fact_type = str(fact.get("fact_type") or "")
            confidence = float(fact.get("confidence") or 0.0)
            status = str(fact.get("status") or "")
            value = str(fact.get("fact_value") or "").strip()
        except Exception:
            continue
        if not value:
            continue
        # Keep identity/preference signals aggressively; others require a little confidence.
        if fact_type not in {"identity", "preference"} and confidence < 0.45 and status != "confirmed":
            continue
        keep.append(fact)
        if len(keep) >= limit:
            break
    return keep


def _merge_voice_memory_fact_rows(
    primary: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fact in [*(primary or []), *(fallback or [])]:
        if not isinstance(fact, dict):
            continue
        fact_key = str(fact.get("fact_key") or "").strip().casefold()
        fact_value = str(fact.get("fact_value") or "").strip()
        dedupe_key = fact_key or fact_value.casefold()
        if not dedupe_key or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        merged.append(fact)
        if len(merged) >= max(1, int(limit)):
            break
    return merged


def _voice_has_identity_fact(facts: list[dict[str, Any]]) -> bool:
    for fact in facts or []:
        if not isinstance(fact, dict):
            continue
        fact_key = str(fact.get("fact_key") or "").strip().casefold()
        fact_type = str(fact.get("fact_type") or "").strip().casefold()
        if fact_type == "identity" or fact_key.startswith("identity:"):
            return True
    return False


async def _get_voice_identity_fallback_fact(
    memory_store: Any,
    *,
    guild_id: str,
    user_id: str,
) -> dict[str, Any] | None:
    getter = getattr(memory_store, "get_latest_user_identity_by_user_id", None)
    if not callable(getter):
        return None
    identity = None
    with contextlib.suppress(Exception):
        identity = await getter(user_id, exclude_guild_id=guild_id)
    if not identity:
        return None
    primary_name = str(identity.get("discord_global_name") or "").strip() or str(identity.get("discord_username") or "").strip()
    username = str(identity.get("discord_username") or "").strip()
    if not (primary_name or username):
        return None
    value = primary_name or username
    return {
        "fact_key": "identity:discord_primary_name",
        "fact_value": value,
        "fact_type": "identity",
        "confidence": 0.98,
        "importance": 0.9,
        "status": "confirmed",
        "pinned": False,
        "evidence_count": 1,
    }


async def _get_voice_summary_with_global_fallback(
    memory_store: Any,
    *,
    guild_id: str,
    user_id: str,
    channel_id: str,
    allow_cross_server_summary_fallback: bool = False,
) -> dict[str, Any] | None:
    summary_row = await memory_store.get_dialogue_summary(guild_id, user_id, channel_id)
    if summary_row is not None and str(summary_row.get("summary_text") or "").strip():
        return summary_row
    if not allow_cross_server_summary_fallback:
        return summary_row
    getter = getattr(memory_store, "get_latest_dialogue_summary_by_user_id", None)
    if not callable(getter):
        return summary_row
    with contextlib.suppress(Exception):
        global_row = await getter(user_id, exclude_guild_id=guild_id)
        if global_row is not None and str(global_row.get("summary_text") or "").strip():
            return global_row
    return summary_row


async def _get_voice_user_biography_summary(memory_store: Any, *, user_id: str) -> dict[str, Any] | None:
    getter = getattr(memory_store, "get_global_user_biography_summary", None)
    if callable(getter):
        with contextlib.suppress(Exception):
            row = await getter(user_id)
            if row is not None and str(row.get("summary_text") or "").strip():
                return row
        return None
    generic_getter = getattr(memory_store, "get_global_biography_summary", None)
    if callable(generic_getter):
        with contextlib.suppress(Exception):
            row = await generic_getter("user", user_id)
            if row is not None and str(row.get("summary_text") or "").strip():
                return row
    return None


async def _get_voice_persona_biography_summary(memory_store: Any, *, persona_id: str) -> dict[str, Any] | None:
    getter = getattr(memory_store, "get_persona_biography_summary", None)
    if callable(getter):
        with contextlib.suppress(Exception):
            row = await getter(persona_id)
            if row is not None and str(row.get("summary_text") or "").strip():
                return row
        return None
    generic_getter = getattr(memory_store, "get_global_biography_summary", None)
    if callable(generic_getter):
        with contextlib.suppress(Exception):
            row = await generic_getter("persona", persona_id)
            if row is not None and str(row.get("summary_text") or "").strip():
                return row
    return None


async def _get_voice_persona_self_facts(
    memory_store: Any,
    *,
    persona_subject_user_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    getter = getattr(memory_store, "get_user_facts_global_by_user_id", None)
    if callable(getter):
        with contextlib.suppress(Exception):
            rows = await getter(persona_subject_user_id, limit=max(1, int(limit)))
            if isinstance(rows, list):
                filtered = [
                    row for row in rows
                    if str((row or {}).get("about_target") or "assistant_self").strip().lower() in {"assistant_self", "unknown"}
                ]
                return filtered
    return []


async def _get_voice_facts_with_global_fallback(
    memory_store: Any,
    *,
    guild_id: str,
    user_id: str,
    limit: int,
    target_count: int,
) -> list[dict[str, Any]]:
    local_facts = await memory_store.get_user_facts(guild_id, user_id, limit=max(1, int(limit)))
    getter = getattr(memory_store, "get_user_facts_global_by_user_id", None)
    if not callable(getter):
        merged = list(local_facts)
        if not _voice_has_identity_fact(merged):
            identity_fact = await _get_voice_identity_fallback_fact(
                memory_store,
                guild_id=guild_id,
                user_id=user_id,
            )
            if identity_fact:
                merged = _merge_voice_memory_fact_rows([identity_fact], merged, limit=max(limit, target_count))
        return merged
    global_facts: list[dict[str, Any]] = []
    with contextlib.suppress(Exception):
        global_facts = await getter(
            user_id,
            limit=max(1, int(max(limit, target_count))),
            exclude_guild_id=guild_id,
        )
    merged = _merge_voice_memory_fact_rows(
        local_facts,
        global_facts,
        limit=max(limit, target_count),
    )
    if not _voice_has_identity_fact(merged):
        identity_fact = await _get_voice_identity_fallback_fact(
            memory_store,
            guild_id=guild_id,
            user_id=user_id,
        )
        if identity_fact:
            merged = _merge_voice_memory_fact_rows([identity_fact], merged, limit=max(limit, target_count))
    return merged


def _recent_user_line_score(text: str) -> float:
    t = collapse_spaces(text).casefold()
    if not t:
        return -1.0
    score = 0.0
    # Explicit memory intent phrases should survive longer than generic chit-chat.
    for needle in ("запам'ятай", "запамятай", "remember that", "remember this"):
        if needle in t:
            score += 5.0
    # Preference/identity cues that are useful across restarts.
    for needle in (
        "я люблю",
        "мені подоба",
        "мене звати",
        "мені ",
        "my name",
        "i like",
        "i love",
        "i am ",
        "мене клич",
    ):
        if needle in t:
            score += 1.6
    # Questions are usually less useful as durable memory than declarative statements.
    if "?" in t or t.startswith("що ти") or t.startswith("ти тут"):
        score -= 0.8
    # Very short lines are often low-value for persistent memory.
    if len(t) < 10:
        score -= 0.5
    return score


def _select_recent_user_memory_candidates(rows: list[dict[str, Any]], *, limit: int = 2) -> list[dict[str, Any]]:
    seen: set[str] = set()
    candidates: list[tuple[float, int, int, str]] = []
    for idx, row in enumerate(rows):
        try:
            role = str(row.get("role") or "").strip().lower()
            text = collapse_spaces(str(row.get("content") or ""))
            message_id = int(row.get("message_id") or 0)
        except Exception:
            continue
        if role != "user" or not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        score = _recent_user_line_score(text)
        # Bias toward more recent rows while still allowing older "remember this" lines to win.
        score += idx * 0.05
        candidates.append((score, idx, message_id, text))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)
    top = candidates[: max(1, int(limit))]
    # Return in chronological order for readability in prompt.
    top.sort(key=lambda item: item[1])
    return [
        {"message_id": int(item[2]), "text": item[3], "score": float(item[0])}
        for item in top
    ]


def _memory_backfill_candidate_ok(candidate: dict[str, Any]) -> bool:
    try:
        text = collapse_spaces(str(candidate.get("text") or ""))
        score = float(candidate.get("score") or 0.0)
    except Exception:
        return False
    if not text:
        return False
    # Skip obvious ephemeral low-signal turns.
    if len(text) < 10 and "?" in text:
        return False
    return score >= 0.8


async def _maybe_backfill_facts_from_recent_candidates(
    *,
    memory_store: Any | None,
    extractor: MemoryExtractor | None,
    runtime_ctx: _BridgeRuntimeContextState,
    room_name: str,
    guild_id: str,
    user_id: str,
    candidates: list[dict[str, Any]],
    cache: dict[str, Any],
    preferred_language: str,
) -> None:
    if memory_store is None or extractor is None:
        return
    promoted_ids = cache.setdefault("promoted_message_ids", set())
    if not isinstance(promoted_ids, set):
        promoted_ids = set()
        cache["promoted_message_ids"] = promoted_ids
    for candidate in candidates[:2]:
        if not _memory_backfill_candidate_ok(candidate):
            runtime_ctx.memory_backfill_skipped += 1
            continue
        message_id = int(candidate.get("message_id") or 0)
        text = collapse_spaces(str(candidate.get("text") or ""))
        if not message_id or not text:
            runtime_ctx.memory_backfill_skipped += 1
            continue
        if message_id in promoted_ids:
            runtime_ctx.memory_backfill_skipped += 1
            continue
        runtime_ctx.memory_backfill_attempts += 1
        try:
            result = await extractor.extract(text, preferred_language)
            if result is None or not result.facts:
                promoted_ids.add(message_id)
                runtime_ctx.memory_backfill_skipped += 1
                continue
            saved = 0
            for fact in result.facts:
                if not str(fact.value or "").strip():
                    continue
                # Slightly stricter than profile worker because this path is opportunistic.
                if float(fact.confidence) < 0.55 and float(fact.importance) < 0.55:
                    continue
                await memory_store.upsert_user_fact(
                    guild_id=guild_id,
                    user_id=user_id,
                    fact_key=fact.key,
                    fact_value=fact.value,
                    fact_type=fact.fact_type,
                    confidence=float(fact.confidence),
                    importance=float(fact.importance),
                    message_id=message_id,
                    extractor="gemini_profile_extractor:voice_retrieval_backfill",
                )
                saved += 1
            promoted_ids.add(message_id)
            if saved:
                runtime_ctx.memory_backfill_saved += saved
                logger.info(
                    "[livekit.mem] backfill saved room=%s user=%s message_id=%s facts=%s",
                    room_name,
                    user_id,
                    message_id,
                    saved,
                )
            else:
                runtime_ctx.memory_backfill_skipped += 1
        except Exception as exc:
            runtime_ctx.memory_updates_errors += 1
            runtime_ctx.last_memory_error = f"backfill:{type(exc).__name__}"
            logger.debug(
                "[livekit.mem] backfill failed room=%s user=%s message_id=%s: %s",
                room_name,
                user_id,
                message_id,
                exc,
            )


async def _maybe_persist_persona_self_facts_from_agent_reply(
    *,
    memory_store: Any | None,
    extractor: MemoryExtractor | None,
    base_settings: Settings,
    room_name: str,
    guild_id: str,
    reply_text: str,
    dedupe_cache: dict[str, Any],
) -> None:
    if memory_store is None or extractor is None:
        return
    if not bool(getattr(base_settings, "memory_enabled", True)):
        return
    if not bool(getattr(base_settings, "memory_persona_self_facts_enabled", True)):
        return

    text = collapse_spaces(str(reply_text or ""))
    if len(text) < 8:
        return

    seen_hashes = dedupe_cache.setdefault("persona_reply_hashes", [])
    if not isinstance(seen_hashes, list):
        seen_hashes = []
        dedupe_cache["persona_reply_hashes"] = seen_hashes
    text_hash = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    if text_hash in seen_hashes:
        return

    dry_run = bool(getattr(base_settings, "memory_extractor_dry_run_enabled", False))
    moderation = FactModerationPolicyV2.from_settings(base_settings)
    persona_subject_id = str(getattr(base_settings, "persona_id", "") or "persona").strip() or "persona"
    persona_subject_user_id = _persona_memory_subject_user_id(base_settings)
    facts_saved = 0
    accepted = 0

    try:
        result = await extractor.extract_persona_self_facts(
            assistant_text=text,
            preferred_language=base_settings.preferred_response_language,
            dialogue_context=None,
        )
    except Exception as exc:
        logger.debug("[livekit.mem.persona] extractor failed room=%s: %s", room_name, exc)
        return

    facts = list(getattr(result, "facts", []) or [])
    if not facts:
        seen_hashes.append(text_hash)
        dedupe_cache["persona_reply_hashes"] = seen_hashes[-64:]
        return

    for fact in facts:
        value = str(getattr(fact, "value", "") or "").strip()
        if not value:
            continue
        about_target = normalize_memory_fact_about_target(
            str(getattr(fact, "about_target", "") or ""),
            default="assistant_self",
        )
        if about_target == "self":
            about_target = "assistant_self"
        if about_target not in {"assistant_self", "unknown"}:
            continue
        directness = normalize_memory_fact_directness(str(getattr(fact, "directness", "") or ""), default="explicit")
        evidence_quote = sanitize_memory_fact_evidence_quote(str(getattr(fact, "evidence_quote", "") or ""))
        decision = moderation.evaluate(
            CandidateModerationInput(
                fact_key=str(getattr(fact, "key", "") or ""),
                fact_value=value,
                fact_type=str(getattr(fact, "fact_type", "fact") or "fact"),
                about_target=about_target,
                directness=directness,
                confidence=float(getattr(fact, "confidence", 0.0) or 0.0),
                importance=float(getattr(fact, "importance", 0.0) or 0.0),
                evidence_quote=evidence_quote,
                owner_kind="persona",
                speaker_role="assistant",
            )
        )
        if not decision.accepted:
            continue
        accepted += 1
        if dry_run:
            continue
        with contextlib.suppress(Exception):
            await memory_store.upsert_user_fact(
                guild_id=guild_id or "livekit",
                user_id=persona_subject_user_id,
                fact_key=str(getattr(fact, "key", "") or ""),
                fact_value=value,
                fact_type=str(getattr(fact, "fact_type", "fact") or "fact"),
                confidence=float(getattr(fact, "confidence", 0.0) or 0.0),
                importance=float(getattr(fact, "importance", 0.0) or 0.0),
                message_id=None,
                extractor="livekit_persona_self_extractor:voice_agent",
                about_target=about_target,
                directness=directness,
                evidence_quote=evidence_quote,
            )
            facts_saved += 1

    seen_hashes.append(text_hash)
    dedupe_cache["persona_reply_hashes"] = seen_hashes[-64:]
    if facts_saved:
        logger.info(
            "[livekit.mem.persona] saved room=%s persona=%s facts=%s accepted=%s",
            room_name,
            persona_subject_id,
            facts_saved,
            accepted,
        )
    elif dry_run and accepted:
        logger.info(
            "[livekit.mem.persona.dry_run] room=%s persona=%s accepted=%s candidates=%s",
            room_name,
            persona_subject_id,
            accepted,
            len(facts),
        )


async def _build_voice_memory_overlay_block(
    *,
    memory_store: Any | None,
    extractor: MemoryExtractor | None,
    base_settings: Settings,
    packet: dict[str, Any] | None,
    room_name: str,
    runtime_ctx: _BridgeRuntimeContextState,
    cache: dict[str, Any],
    max_chars: int = 520,
) -> str:
    if memory_store is None or not bool(getattr(base_settings, "memory_enabled", True)):
        runtime_ctx.memory_updates_ignored += 1
        return ""
    focuses = _resolve_voice_memory_focuses(packet, max_users=3)
    if not focuses:
        runtime_ctx.memory_updates_ignored += 1
        return ""
    guild_id = str(focuses[0].get("guild_id") or "").strip()
    channel_id = str(focuses[0].get("channel_id") or "").strip()
    if not (guild_id and channel_id):
        runtime_ctx.memory_updates_ignored += 1
        return ""
    focus_user_ids = [str(item.get("user_id") or "").strip() for item in focuses if str(item.get("user_id") or "").strip()]
    if not focus_user_ids:
        runtime_ctx.memory_updates_ignored += 1
        return ""

    cache_key = f"{guild_id}:{channel_id}:{','.join(focus_user_ids[:3])}"
    now = time.monotonic()
    if cache.get("key") == cache_key and (now - float(cache.get("fetched_at", 0.0) or 0.0)) < 8.0:
        block = str(cache.get("block") or "")
        if block:
            runtime_ctx.last_memory_focus_user_id = focus_user_ids[0]
            runtime_ctx.last_memory_focus_label = str(focuses[0].get("label") or "")
        else:
            runtime_ctx.memory_updates_ignored += 1
        return block

    try:
        lines = ["PERSISTED MEMORY (voice retrieval, lower priority than RP CANON):"]
        persona_subject_id = str(getattr(base_settings, "persona_id", "") or "persona").strip() or "persona"
        persona_subject_user_id = _persona_memory_subject_user_id(base_settings)
        persona_bio_row = await _get_voice_persona_biography_summary(memory_store, persona_id=persona_subject_id)
        persona_bio_text = str((persona_bio_row or {}).get("summary_text") or "").strip()
        persona_facts_rows = await _get_voice_persona_self_facts(
            memory_store,
            persona_subject_user_id=persona_subject_user_id,
            limit=8,
        )
        persona_facts = _select_voice_memory_facts(base_settings, persona_facts_rows)[:4]
        if persona_bio_text or persona_facts:
            lines.append("PERSISTED PERSONA MEMORY:")
            if persona_bio_text:
                lines.append(build_persona_biography_summary_line(_trim_runtime_text(persona_bio_text, 220)))
            if persona_facts:
                lines.append(build_persona_relevant_facts_section(persona_facts))

        lines.append("PERSISTED USER MEMORY:")
        get_recent_dialogue_messages = getattr(memory_store, "get_recent_dialogue_messages", None)
        user_sections_added = 0
        for idx, focus in enumerate(focuses[:3]):
            user_id = str(focus.get("user_id") or "").strip()
            label = str(focus.get("label") or "").strip()
            if not user_id:
                continue
            summary_row = await _get_voice_summary_with_global_fallback(
                memory_store,
                guild_id=guild_id,
                user_id=user_id,
                channel_id=channel_id,
                allow_cross_server_summary_fallback=bool(
                    getattr(base_settings, "memory_cross_server_dialogue_summary_fallback_enabled", False)
                ),
            )
            user_bio_row = await _get_voice_user_biography_summary(memory_store, user_id=user_id)
            raw_facts = await _get_voice_facts_with_global_fallback(
                memory_store,
                guild_id=guild_id,
                user_id=user_id,
                limit=8,
                target_count=4,
            )
            recent_rows: list[dict[str, Any]] = []
            if callable(get_recent_dialogue_messages):
                with contextlib.suppress(Exception):
                    # Fetch deeper history so durable user preferences can be recalled across restarts.
                    recent_rows = await get_recent_dialogue_messages(guild_id, channel_id, user_id, 48)

            facts = _select_voice_memory_facts(base_settings, raw_facts)
            recent_candidates = _select_recent_user_memory_candidates(recent_rows, limit=2 if idx == 0 else 1)
            recent_user_lines = [
                str(item.get("text") or "")
                for item in recent_candidates
                if str(item.get("text") or "").strip()
            ]

            # Opportunistic backfill only for the primary focus user to avoid excess extractor calls.
            if idx == 0:
                await _maybe_backfill_facts_from_recent_candidates(
                    memory_store=memory_store,
                    extractor=extractor,
                    runtime_ctx=runtime_ctx,
                    room_name=room_name,
                    guild_id=guild_id,
                    user_id=user_id,
                    candidates=recent_candidates,
                    cache=cache,
                    preferred_language=base_settings.preferred_response_language,
                )

            summary_text = str((summary_row or {}).get("summary_text") or "").strip()
            user_bio_text = str((user_bio_row or {}).get("summary_text") or "").strip()
            if not (user_bio_text or summary_text or facts or recent_user_lines):
                continue
            user_sections_added += 1
            lines.append(
                f"- user[{user_sections_added}]: {(_trim_runtime_text(label, 32) or '?')} ({_trim_runtime_text(user_id, 24)})"
            )
            if user_bio_text:
                lines.append(build_user_biography_summary_line(_trim_runtime_text(user_bio_text, 220)))
            if summary_text:
                lines.append(build_user_dialogue_summary_line(_trim_runtime_text(summary_text, 180)))
            if facts:
                # Keep per-user facts concise so multiple people fit in one runtime block.
                lines.append(build_relevant_facts_section(facts[:3]))
            if recent_user_lines:
                lines.append("Recent user statements (fallback from messages):")
                for line in recent_user_lines[: (2 if idx == 0 else 1)]:
                    lines.append(f"- {_trim_runtime_text(line, 120)}")

        block = "\n".join(line for line in lines if line).strip()
        if user_sections_added <= 0 and not (persona_bio_text or persona_facts):
            block = ""
        if len(block) > max_chars:
            block = block[: max_chars - 3].rstrip() + "..."
        cache["key"] = cache_key
        cache["fetched_at"] = now
        cache["block"] = block
        if block:
            runtime_ctx.last_memory_focus_user_id = focus_user_ids[0]
            runtime_ctx.last_memory_focus_label = str(focuses[0].get("label") or "")
            block_hash = hashlib.sha1(block.encode("utf-8", errors="ignore")).hexdigest()
            if block_hash != runtime_ctx.last_memory_hash:
                runtime_ctx.memory_updates_applied += 1
                runtime_ctx.last_memory_hash = block_hash
            else:
                runtime_ctx.memory_updates_ignored += 1
            logger.debug(
                "[livekit.mem] room=%s users=%s chars=%s",
                room_name,
                ",".join(focus_user_ids[:3]),
                len(block),
            )
        else:
            runtime_ctx.memory_updates_ignored += 1
        runtime_ctx.last_memory_error = ""
        return block
    except Exception as exc:
        runtime_ctx.memory_updates_errors += 1
        runtime_ctx.last_memory_error = f"{type(exc).__name__}"
        logger.debug("[livekit.mem] retrieval failed room=%s: %s", room_name, exc)
        return ""


def _compose_livekit_instructions_with_runtime_context(base_instructions: str, runtime_block: str) -> str:
    base = (base_instructions or "").strip()
    block = (runtime_block or "").strip()
    if not block:
        return base
    if not base:
        return block
    return f"{base}\n\n{block}".strip()


def _bridge_snapshot_assistant_audio_busy(
    packet: dict[str, Any] | None,
    *,
    idle_grace_sec: float = 0.9,
) -> bool:
    if not isinstance(packet, dict):
        return False
    payload = packet.get("payload", {}) if isinstance(packet.get("payload"), dict) else {}
    bridge_obj = payload.get("bridge_runtime", {}) if isinstance(payload.get("bridge_runtime"), dict) else {}
    if bool(bridge_obj.get("assistant_audio_active")):
        return True
    try:
        last_unix = float(bridge_obj.get("assistant_audio_last_unix") or 0.0)
    except Exception:
        last_unix = 0.0
    if last_unix <= 0:
        return False
    return (time.time() - last_unix) < max(0.0, float(idle_grace_sec))


def _packet_topic(packet: object) -> str:
    return str(getattr(packet, "topic", "") or "").strip()


async def _publish_context_apply_ack(
    *,
    room: Any,
    lk_settings: LiveKitAgentSettings,
    room_name: str,
    runtime_ctx: _BridgeRuntimeContextState,
) -> None:
    try:
        payload = {
            "event": "agent_context_applied",
            "source": "livekit-agent-runtime",
            "version": 1,
            "room_name": room_name,
            "seq": int(runtime_ctx.last_seq or 0),
            "reason": str(runtime_ctx.last_applied_reason or runtime_ctx.last_received_reason or ""),
            "chars": int(runtime_ctx.last_applied_chars or 0),
            "received": int(runtime_ctx.updates_received or 0),
            "applied": int(runtime_ctx.updates_applied or 0),
            "ignored": int(runtime_ctx.updates_ignored or 0),
            "errors": int(runtime_ctx.updates_errors or 0),
            "last_error": str(runtime_ctx.last_apply_error or ""),
            "applied_at_unix": round(time.time(), 3),
        }
        await room.local_participant.publish_data(
            json.dumps(payload, ensure_ascii=False),
            reliable=True,
            topic=str(lk_settings.bridge_context_topic or "bridge-context"),
        )
    except Exception:
        # Best-effort observability path; never fail the session on ACK publish.
        return


def _import_livekit_modules() -> tuple[Any, Any, Any]:
    try:
        from livekit import agents  # type: ignore
        from livekit.agents import Agent, AgentServer, AgentSession, room_io  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "LiveKit Agents dependencies are missing. Install with: pip install -r requirements-livekit.txt"
        ) from exc

    return agents, (Agent, AgentServer, AgentSession, room_io), None


def _import_livekit_plugins(use_silero_vad: bool) -> tuple[Any, Any | None]:
    try:
        from livekit.plugins import google  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency check
        text = str(exc)
        if "Plugins must be registered on the main thread" in text:
            raise RuntimeError(
                "LiveKit Google plugin import happened outside the job process main thread. "
                "Use AgentServer.setup_fnc prewarm (patched in this project) or run the agent in start mode."
            ) from exc
        raise RuntimeError(
            "LiveKit Google plugin is unavailable. Install with: pip install -r requirements-livekit.txt"
        ) from exc

    silero = None
    if use_silero_vad:
        with contextlib.suppress(Exception):
            from livekit.plugins import silero as _silero  # type: ignore

            silero = _silero
        if silero is None:
            logger.warning("LIVEKIT_USE_SILERO_VAD=true, but silero plugin is unavailable. Using Gemini built-in VAD.")

    return google, silero


def _build_google_realtime_model(google: Any, lk_settings: LiveKitAgentSettings, instructions: str) -> Any:
    model_kwargs = {
        "model": lk_settings.google_realtime_model,
        "voice": lk_settings.voice,
        "temperature": lk_settings.temperature,
        "instructions": instructions,
    }
    if lk_settings.google_api_key:
        # Pass explicitly to avoid child-process env propagation quirks in LiveKit job runners.
        model_kwargs["api_key"] = lk_settings.google_api_key
    # LiveKit plugin versions expose Gemini realtime model under different namespaces.
    beta = getattr(google, "beta", None)
    if beta is not None and hasattr(beta, "realtime"):
        realtime_ns = getattr(beta, "realtime")
        if hasattr(realtime_ns, "RealtimeModel"):
            return realtime_ns.RealtimeModel(**model_kwargs)
    realtime_ns = getattr(google, "realtime", None)
    if realtime_ns is not None and hasattr(realtime_ns, "RealtimeModel"):
        return realtime_ns.RealtimeModel(**model_kwargs)
    raise RuntimeError("Unsupported livekit.plugins.google API: RealtimeModel not found")


def _validate_paths(lk_settings: LiveKitAgentSettings) -> None:
    path = lk_settings.bot_history_json_path
    if not path.exists():
        logger.warning("LiveKit role prompt file not found: %s (will use env role prompts only)", path)


@lru_cache(maxsize=1)
def _load_runtime_deps() -> _RuntimeDeps:
    # In LiveKit `dev` mode on Windows, worker jobs run in a spawned process.
    # Reconfigure logging here so child-process session logs are visible.
    configure_logging()
    base_settings = Settings.from_env()  # Discord token is not required for the LiveKit worker
    lk_settings = LiveKitAgentSettings.from_env(base_settings)
    lk_settings.validate()
    lk_settings.export_env()
    _validate_paths(lk_settings)

    prompt_ctx = _build_prompt_context(base_settings, lk_settings)
    instructions = _build_livekit_instructions(prompt_ctx)

    google, silero = _import_livekit_plugins(lk_settings.use_silero_vad)
    health = LiveKitRuntimeHealth(worker_name=lk_settings.worker_name, agent_name=lk_settings.agent_name)
    backfill_llm: GeminiClient | None = None
    backfill_extractor: MemoryExtractor | None = None
    try:
        if base_settings.memory_enabled and base_settings.gemini_api_key:
            backfill_llm = GeminiClient(
                api_key=base_settings.gemini_api_key,
                model=base_settings.gemini_model,
                timeout_seconds=max(20, int(base_settings.gemini_timeout_seconds)),
                temperature=0.1,
                max_output_tokens=800,
                base_url=base_settings.gemini_base_url,
            )
            backfill_extractor = MemoryExtractor(
                enabled=True,
                llm=backfill_llm,
                candidate_limit=max(1, int(base_settings.memory_candidate_fact_limit or 6)),
            )
            with contextlib.suppress(Exception):
                awaitable_start = getattr(backfill_llm, "start", None)
                if callable(awaitable_start):
                    # `_load_runtime_deps` is sync; defer explicit startup to first use if needed.
                    pass
    except Exception:
        backfill_llm = None
        backfill_extractor = None
    return _RuntimeDeps(
        base_settings=base_settings,
        lk_settings=lk_settings,
        instructions=instructions,
        google=google,
        silero=silero,
        health=health,
        backfill_extractor=backfill_extractor,
        backfill_llm=backfill_llm,
    )


class _LizaVoiceAgent:  # runtime base class will be created dynamically per process
    pass


def _prewarm_job_process(_job_proc: Any) -> None:
    """Runs in the LiveKit job process main thread before job tasks start.

    We preload runtime deps here so plugin registration happens on the main thread
    (required by livekit.plugins.google on Windows / threaded job executors).
    """
    try:
        deps = _load_runtime_deps()
        logger.debug(
            "[livekit.prewarm] deps loaded worker=%s agent=%s model=%s",
            deps.lk_settings.worker_name,
            deps.lk_settings.agent_name,
            deps.lk_settings.google_realtime_model,
        )
    except Exception:
        logger.exception("[livekit.prewarm] failed to preload runtime deps in job process")
        raise


async def _handle_session(ctx: Any) -> None:
    configure_logging()
    deps = _load_runtime_deps()
    base_settings = deps.base_settings
    lk_settings = deps.lk_settings
    health = deps.health
    backfill_extractor = deps.backfill_extractor
    backfill_llm = deps.backfill_llm

    _, agent_types, _ = _import_livekit_modules()
    Agent, _, AgentSession, room_io = agent_types

    # Build a small concrete Agent subclass at runtime with current instructions.
    class LizaVoiceAgent(Agent):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__(instructions=deps.instructions)

    room_name = "-"
    session: Any = None
    hb: HealthHeartbeat | None = None
    runtime_ctx = _BridgeRuntimeContextState()
    runtime_ctx_event = asyncio.Event()
    runtime_ctx_task: asyncio.Task[None] | None = None
    runtime_ctx_debounce_sec = max(0.25, min(2.0, float(lk_settings.bridge_context_min_interval_ms) / 2000.0))
    runtime_ctx_min_apply_sec = max(0.75, min(5.0, float(lk_settings.bridge_context_min_interval_ms) / 1000.0))
    runtime_ctx_busy_retry_sec = max(0.35, min(1.25, runtime_ctx_debounce_sec))
    runtime_ctx_idle_grace_sec = 0.9
    agent_instance: Any | None = None
    local_participant_identity = ""
    memory_store: Any | None = None
    memory_cache: dict[str, Any] = {"key": "", "fetched_at": 0.0, "block": ""}
    persona_self_cache: dict[str, Any] = {}
    persona_self_tasks: set[asyncio.Task[Any]] = set()

    try:
        try:
            memory_store = build_memory_store(base_settings.sqlite_path)
            init_memory = getattr(memory_store, "init", None)
            if callable(init_memory):
                await init_memory()
        except Exception as exc:
            memory_store = None
            logger.warning("[livekit.mem] disabled room=%s reason=%s", room_name, exc)

        with contextlib.suppress(Exception):
            room_name = str(getattr(getattr(ctx, "room", None), "name", "-") or "-")
        logger.info("[livekit.room] job received room=%s", room_name)

        await ctx.connect()
        with contextlib.suppress(Exception):
            room_name = str(getattr(ctx.room, "name", "-") or "-")
        with contextlib.suppress(Exception):
            local_participant_identity = str(getattr(getattr(ctx, "room", None), "local_participant", None).identity or "")
        health.room_name = room_name
        health.rooms_started += 1
        health.mark_activity()
        logger.info("[livekit.room] connected room=%s", room_name)

        hb = HealthHeartbeat(health, lk_settings.health_log_interval_seconds)
        hb.start()

        session_kwargs: dict[str, Any] = {
            "llm": _build_google_realtime_model(deps.google, lk_settings, deps.instructions)
        }
        if deps.silero is not None:
            with contextlib.suppress(Exception):
                session_kwargs["vad"] = deps.silero.VAD.load()

        session = AgentSession(**session_kwargs)
        room_guild_id = _parse_bridge_room_guild_id(room_name, lk_settings.room_prefix)
        room_input_kwargs: dict[str, Any] = {}
        if lk_settings.auto_subscribe_audio_only:
            with contextlib.suppress(Exception):
                room_input_kwargs["audio_enabled"] = True
                room_input_kwargs["video_enabled"] = False

        session_closed = asyncio.Event()
        job_shutdown = asyncio.Event()

        def _on_session_close(*_: Any) -> None:
            session_closed.set()

        async def _on_job_shutdown(*_: Any) -> None:
            job_shutdown.set()

        with contextlib.suppress(Exception):
            session.on("close", _on_session_close)
        with contextlib.suppress(Exception):
            ctx.add_shutdown_callback(_on_job_shutdown)

        def _on_conversation_item_added(event: Any) -> None:
            try:
                item = getattr(event, "item", None)
                role = str(getattr(item, "role", "") or "").strip().lower()
                if role != "assistant":
                    return
                text_content = getattr(item, "text_content", None)
                text = collapse_spaces(str(text_content or ""))
                if not text:
                    return
                task = asyncio.create_task(
                    _maybe_persist_persona_self_facts_from_agent_reply(
                        memory_store=memory_store,
                        extractor=backfill_extractor,
                        base_settings=base_settings,
                        room_name=room_name,
                        guild_id=room_guild_id or "livekit",
                        reply_text=text,
                        dedupe_cache=persona_self_cache,
                    ),
                    name=f"lk-persona-self-facts-{room_name}",
                )
                persona_self_tasks.add(task)

                def _drop_task(done: asyncio.Task[Any]) -> None:
                    persona_self_tasks.discard(done)
                    with contextlib.suppress(Exception):
                        done.result()

                task.add_done_callback(_drop_task)
            except Exception as exc:
                logger.debug("[livekit.mem.persona] conversation item hook failed room=%s: %s", room_name, exc)

        with contextlib.suppress(Exception):
            session.on("conversation_item_added", _on_conversation_item_added)

        def _on_room_data(packet: Any) -> None:
            if not lk_settings.agent_runtime_context_injection_enabled:
                return
            if _packet_topic(packet) != str(lk_settings.bridge_context_topic or "bridge-context"):
                return
            try:
                packet_participant = getattr(packet, "participant", None)
                packet_participant_identity = str(getattr(packet_participant, "identity", "") or "")
                raw = getattr(packet, "data", b"")
                raw_bytes_len = 0
                if isinstance(raw, bytes):
                    raw_bytes_len = len(raw)
                    payload_text = raw.decode("utf-8", errors="ignore")
                else:
                    payload_text = str(raw or "")
                    raw_bytes_len = len(payload_text.encode("utf-8", errors="ignore"))
                if raw_bytes_len <= 0 or raw_bytes_len > 65536:
                    runtime_ctx.updates_ignored += 1
                    runtime_ctx.last_apply_error = "packet_size_invalid"
                    return
                payload_obj = json.loads(payload_text)
                if not isinstance(payload_obj, dict):
                    runtime_ctx.updates_ignored += 1
                    return
                event_name = str(payload_obj.get("event") or "")
                if event_name == "agent_context_applied":
                    runtime_ctx.ack_packets_seen += 1
                    # This is our own observability ACK path on the same topic. Ignore silently so agent_ignored
                    # reflects bridge snapshot filtering/coalescing, not self-ACK traffic.
                    return
                if event_name != "bridge_context_snapshot":
                    runtime_ctx.updates_ignored += 1
                    return
                if local_participant_identity and packet_participant_identity == local_participant_identity:
                    runtime_ctx.updates_ignored += 1
                    runtime_ctx.last_apply_error = "unexpected_self_context_packet"
                    return
                source_name = str(payload_obj.get("source") or "")
                if source_name and source_name != "discord-livekit-bridge":
                    runtime_ctx.updates_ignored += 1
                    return
                packet_room_name = str(payload_obj.get("room_name") or "")
                if packet_room_name and packet_room_name != room_name:
                    runtime_ctx.updates_ignored += 1
                    return
                seq_raw = payload_obj.get("seq", 0)
                try:
                    seq = int(seq_raw or 0)
                except Exception:
                    seq = 0
                if seq > 0 and seq <= runtime_ctx.last_seq:
                    runtime_ctx.updates_ignored += 1
                    return
                runtime_ctx.last_seq = max(int(runtime_ctx.last_seq or 0), int(seq or 0))
                runtime_ctx.last_received_reason = str(payload_obj.get("reason") or "")
                runtime_ctx.latest_packet = payload_obj
                runtime_ctx.last_received_at = time.monotonic()
                runtime_ctx.updates_received += 1
                runtime_ctx_event.set()
                health.mark_activity()
            except Exception as exc:
                runtime_ctx.updates_errors += 1
                runtime_ctx.last_apply_error = f"packet_parse:{type(exc).__name__}"
                logger.debug("[livekit.ctx] failed to parse bridge context packet room=%s: %s", room_name, exc)

        if lk_settings.agent_runtime_context_injection_enabled:
            with contextlib.suppress(Exception):
                ctx.room.on("data_received", _on_room_data)

        async def _runtime_context_apply_loop() -> None:
            if not lk_settings.agent_runtime_context_injection_enabled:
                return
            while True:
                await runtime_ctx_event.wait()
                runtime_ctx_event.clear()
                if runtime_ctx_debounce_sec > 0:
                    await asyncio.sleep(runtime_ctx_debounce_sec)
                if runtime_ctx_event.is_set():
                    # Newer packet arrived while debouncing; coalesce and let next loop iteration handle it.
                    continue
                packet_obj = runtime_ctx.latest_packet
                if not isinstance(packet_obj, dict):
                    continue
                if _bridge_snapshot_assistant_audio_busy(
                    packet_obj,
                    idle_grace_sec=runtime_ctx_idle_grace_sec,
                ):
                    # Context updates during active assistant speech can interrupt or truncate realtime TTS output.
                    # Defer and retry after the current turn finishes (or a newer packet arrives).
                    await asyncio.sleep(runtime_ctx_busy_retry_sec)
                    if not runtime_ctx_event.is_set():
                        runtime_ctx_event.set()
                    continue
                runtime_ctx.last_applied_reason = str(packet_obj.get("reason") or runtime_ctx.last_received_reason or "")
                runtime_block = _build_runtime_context_block(
                    packet_obj,
                    max_chars=lk_settings.agent_runtime_context_max_chars,
                )
                memory_block = await _build_voice_memory_overlay_block(
                    memory_store=memory_store,
                    extractor=backfill_extractor,
                    base_settings=base_settings,
                    packet=packet_obj,
                    room_name=room_name,
                    runtime_ctx=runtime_ctx,
                    cache=memory_cache,
                    max_chars=max(220, min(720, lk_settings.agent_runtime_context_max_chars + 180)),
                )
                merged_instructions = _compose_livekit_instructions_with_runtime_context(
                    deps.instructions,
                    "\n\n".join(part for part in (runtime_block, memory_block) if part).strip(),
                )
                merged_hash = hashlib.sha1(merged_instructions.encode("utf-8", errors="ignore")).hexdigest()
                if merged_hash == runtime_ctx.last_applied_hash:
                    runtime_ctx.updates_ignored += 1
                    continue
                now = time.monotonic()
                wait_more = runtime_ctx_min_apply_sec - (now - float(runtime_ctx.last_applied_at or 0.0))
                if wait_more > 0:
                    await asyncio.sleep(wait_more)
                    if runtime_ctx_event.is_set():
                        continue
                if agent_instance is None or session is None:
                    runtime_ctx.updates_ignored += 1
                    continue
                try:
                    applied = False
                    update_instructions = getattr(agent_instance, "update_instructions", None)
                    if callable(update_instructions):
                        await update_instructions(merged_instructions)
                        applied = True
                    else:
                        setattr(agent_instance, "instructions", merged_instructions)
                        update_agent = getattr(session, "update_agent", None)
                        if callable(update_agent):
                            await update_agent(agent_instance)
                            applied = True
                    if not applied:
                        runtime_ctx.updates_ignored += 1
                        runtime_ctx.last_apply_error = "no_update_api"
                        logger.debug("[livekit.ctx] no runtime instruction update API room=%s", room_name)
                        continue
                    runtime_ctx.last_applied_hash = merged_hash
                    runtime_ctx.last_applied_chars = len(runtime_block)
                    runtime_ctx.last_applied_at = time.monotonic()
                    runtime_ctx.updates_applied += 1
                    runtime_ctx.last_apply_error = ""
                    health.mark_activity()
                    await _publish_context_apply_ack(
                        room=ctx.room,
                        lk_settings=lk_settings,
                        room_name=room_name,
                        runtime_ctx=runtime_ctx,
                    )
                    if runtime_ctx.updates_applied == 1 or (runtime_ctx.updates_applied % 10) == 0:
                        logger.info(
                            "[livekit.ctx] applied bridge context room=%s seq=%s reason=%s block_chars=%s applied=%s received=%s ignored=%s errors=%s",
                            room_name,
                            runtime_ctx.last_seq,
                            runtime_ctx.last_applied_reason or "-",
                            runtime_ctx.last_applied_chars,
                            runtime_ctx.updates_applied,
                            runtime_ctx.updates_received,
                            runtime_ctx.updates_ignored,
                            runtime_ctx.updates_errors,
                        )
                    else:
                        logger.debug(
                            "[livekit.ctx] applied bridge context room=%s seq=%s reason=%s block_chars=%s",
                            room_name,
                            runtime_ctx.last_seq,
                            runtime_ctx.last_applied_reason or "-",
                            runtime_ctx.last_applied_chars,
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    runtime_ctx.updates_errors += 1
                    runtime_ctx.last_apply_error = f"apply:{type(exc).__name__}"
                    await _publish_context_apply_ack(
                        room=ctx.room,
                        lk_settings=lk_settings,
                        room_name=room_name,
                        runtime_ctx=runtime_ctx,
                    )
                    logger.warning(
                        "[livekit.ctx] runtime context apply failed room=%s seq=%s: %s",
                        room_name,
                        runtime_ctx.last_seq,
                        exc,
                    )

        try:
            wait_job_task: asyncio.Task[bool] | None = None
            wait_close_task: asyncio.Task[bool] | None = None
            start_kwargs: dict[str, Any] = {}
            room_options_ctor = getattr(room_io, "RoomOptions", None)
            audio_input_ctor = getattr(room_io, "AudioInputOptions", None)
            if room_options_ctor is not None:
                ro_kwargs: dict[str, Any] = {}
                if lk_settings.auto_subscribe_audio_only:
                    # New API path (avoids deprecated RoomInputOptions warning).
                    if audio_input_ctor is not None:
                        ro_kwargs["audio_input"] = audio_input_ctor()
                    else:
                        ro_kwargs["audio_input"] = True
                    ro_kwargs["video_input"] = False
                start_kwargs["room_options"] = room_options_ctor(**ro_kwargs)
            else:
                # Compatibility fallback for older livekit-agents versions.
                start_kwargs["room_input_options"] = room_io.RoomInputOptions(**room_input_kwargs)

            agent_instance = LizaVoiceAgent()
            await session.start(
                agent=agent_instance,
                room=ctx.room,
                **start_kwargs,
            )
            health.mark_activity()
            logger.info("[livekit.room] session started room=%s", room_name)
            if lk_settings.agent_runtime_context_injection_enabled:
                runtime_ctx_task = asyncio.create_task(
                    _runtime_context_apply_loop(),
                    name=f"lk-agent-ctx-sync-{room_name}",
                )
            wait_job_task = asyncio.create_task(job_shutdown.wait(), name=f"lk-job-shutdown-{room_name}")
            wait_close_task = asyncio.create_task(session_closed.wait(), name=f"lk-session-close-{room_name}")
            try:
                done, pending = await asyncio.wait(
                    {wait_job_task, wait_close_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                reason = "job_shutdown" if wait_job_task in done else "session_closed"
                logger.info("[livekit.room] session wait completed room=%s reason=%s", room_name, reason)
            finally:
                if wait_job_task is not None:
                    with contextlib.suppress(Exception):
                        wait_job_task.cancel()
                if wait_close_task is not None:
                    with contextlib.suppress(Exception):
                        wait_close_task.cancel()
                if runtime_ctx_task is not None:
                    runtime_ctx_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await runtime_ctx_task
                if persona_self_tasks:
                    for task in list(persona_self_tasks):
                        task.cancel()
                    for task in list(persona_self_tasks):
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await task
                    persona_self_tasks.clear()
        finally:
            with contextlib.suppress(Exception):
                if session is not None:
                    await session.aclose()
            if hb is not None:
                await hb.stop()
            with contextlib.suppress(Exception):
                close_memory = getattr(memory_store, "close", None)
                if callable(close_memory):
                    await close_memory()
            with contextlib.suppress(Exception):
                if backfill_llm is not None:
                    await backfill_llm.close()
            health.rooms_closed += 1
            health.mark_activity()
            logger.info("[livekit.room] session closed room=%s", room_name)
    except Exception:
        logger.exception("[livekit.room] session error room=%s", room_name)
        raise


def main() -> None:
    configure_logging()
    deps = _load_runtime_deps()
    lk_settings = deps.lk_settings
    instructions = deps.instructions

    logger.info(
        "[livekit.start] worker=%s agent=%s url=%s model=%s voice=%s bridge_enabled=%s",
        lk_settings.worker_name,
        lk_settings.agent_name,
        lk_settings.url,
        lk_settings.google_realtime_model,
        lk_settings.voice,
        lk_settings.bridge_enabled,
    )
    logger.info("[livekit.start] instructions chars=%s", len(instructions))

    if lk_settings.bridge_enabled:
        bridge = DiscordLiveKitBridge(
            LiveKitBridgeConfig(
                enabled=True,
                room_prefix=lk_settings.room_prefix,
                control_channel_name=lk_settings.bridge_control_channel,
            )
        )
        logger.info(
            "[bridge.plan] enabled scaffold control_channel=%s room_prefix=%s",
            bridge.config.control_channel_name,
            bridge.config.room_prefix,
        )

    agents, agent_types, _ = _import_livekit_modules()
    _, AgentServer, _, _ = agent_types

    server = AgentServer(setup_fnc=_prewarm_job_process)

    server.rtc_session(agent_name=lk_settings.agent_name)(_handle_session)  # type: ignore[misc]

    agents.cli.run_app(server)
