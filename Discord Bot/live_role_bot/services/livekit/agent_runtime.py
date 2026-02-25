from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
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
from ...prompts.dialogue import build_relevant_facts_section, build_user_dialogue_summary_line
from ...prompts.voice import build_native_audio_system_instruction
from .bridge import DiscordLiveKitBridge, LiveKitBridgeConfig
from .config import LiveKitAgentSettings
from .observability import HealthHeartbeat, LiveKitRuntimeHealth

logger = logging.getLogger("live_role_bot.livekit")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("livekit").setLevel(logging.INFO)


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


def _resolve_voice_memory_focus(packet: dict[str, Any] | None) -> dict[str, str]:
    payload = _context_payload(packet)
    guild_obj = payload.get("guild", {}) if isinstance(payload.get("guild"), dict) else {}
    text_obj = payload.get("text_channel", {}) if isinstance(payload.get("text_channel"), dict) else {}
    participants_obj = payload.get("participants", {}) if isinstance(payload.get("participants"), dict) else {}
    bridge_obj = payload.get("bridge_runtime", {}) if isinstance(payload.get("bridge_runtime"), dict) else {}
    members = participants_obj.get("members", []) if isinstance(participants_obj.get("members"), list) else []

    guild_id = str(guild_obj.get("id") or packet.get("guild_id") or "").strip() if isinstance(packet, dict) else ""
    channel_id = str(text_obj.get("id") or packet.get("text_channel_id") or "").strip() if isinstance(packet, dict) else ""
    preferred_user_id = str(bridge_obj.get("last_ingress_user_id") or "").strip()

    focus_member: dict[str, Any] | None = None
    if preferred_user_id:
        for item in members:
            if not isinstance(item, dict):
                continue
            if str(item.get("user_id") or "").strip() == preferred_user_id:
                focus_member = item
                break
    if focus_member is None:
        for item in members:
            if not isinstance(item, dict):
                continue
            if bool(item.get("bot")):
                continue
            user_id = str(item.get("user_id") or "").strip()
            if user_id:
                focus_member = item
                break

    user_id = str((focus_member or {}).get("user_id") or preferred_user_id or "").strip()
    label = _trim_runtime_text((focus_member or {}).get("display_name") or bridge_obj.get("last_ingress_user_label") or "", 40)
    return {
        "guild_id": guild_id,
        "channel_id": channel_id,
        "user_id": user_id,
        "label": label,
    }


def _select_voice_memory_facts(base_settings: Settings, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not facts:
        return []
    keep: list[dict[str, Any]] = []
    limit = max(2, min(6, int(getattr(base_settings, "memory_fact_top_k", 4) or 4)))
    for fact in facts:
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


def _select_recent_user_memory_lines(rows: list[dict[str, Any]], *, limit: int = 2) -> list[str]:
    seen: set[str] = set()
    candidates: list[tuple[float, int, str]] = []
    for idx, row in enumerate(rows):
        try:
            role = str(row.get("role") or "").strip().lower()
            text = collapse_spaces(str(row.get("content") or ""))
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
        candidates.append((score, idx, text))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)
    top = candidates[: max(1, int(limit))]
    # Return in chronological order for readability in prompt.
    top.sort(key=lambda item: item[1])
    return [item[2] for item in top]


async def _build_voice_memory_overlay_block(
    *,
    memory_store: Any | None,
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
    focus = _resolve_voice_memory_focus(packet)
    guild_id = focus.get("guild_id", "")
    channel_id = focus.get("channel_id", "")
    user_id = focus.get("user_id", "")
    if not (guild_id and channel_id and user_id):
        runtime_ctx.memory_updates_ignored += 1
        return ""

    cache_key = f"{guild_id}:{channel_id}:{user_id}"
    now = time.monotonic()
    if cache.get("key") == cache_key and (now - float(cache.get("fetched_at", 0.0) or 0.0)) < 8.0:
        block = str(cache.get("block") or "")
        if block:
            runtime_ctx.last_memory_focus_user_id = user_id
            runtime_ctx.last_memory_focus_label = focus.get("label", "")
        else:
            runtime_ctx.memory_updates_ignored += 1
        return block

    try:
        summary_row = await memory_store.get_dialogue_summary(guild_id, user_id, channel_id)
        raw_facts = await memory_store.get_user_facts(guild_id, user_id, limit=8)
        recent_rows: list[dict[str, Any]] = []
        get_recent_dialogue_messages = getattr(memory_store, "get_recent_dialogue_messages", None)
        if callable(get_recent_dialogue_messages):
            with contextlib.suppress(Exception):
                # Fetch deeper history so explicit "remember this" user statements survive beyond a few turns.
                recent_rows = await get_recent_dialogue_messages(guild_id, channel_id, user_id, 48)
        facts = _select_voice_memory_facts(base_settings, raw_facts)
        recent_user_lines = _select_recent_user_memory_lines(recent_rows, limit=3)
        summary_text = str((summary_row or {}).get("summary_text") or "").strip()
        lines = ["PERSISTED USER MEMORY (voice retrieval, lower priority than RP CANON):"]
        label = focus.get("label", "")
        if label or user_id:
            lines.append(f"- likely_active_user: {label or '?'} ({_trim_runtime_text(user_id, 24)})")
        if summary_text:
            lines.append(build_user_dialogue_summary_line(_trim_runtime_text(summary_text, 260)))
        if facts:
            lines.append(build_relevant_facts_section(facts))
        if recent_user_lines:
            lines.append("Recent user statements (fresh, lower confidence than extracted facts):")
            for line in recent_user_lines:
                lines.append(f"- {_trim_runtime_text(line, 140)}")
        block = "\n".join(line for line in lines if line).strip()
        if len(lines) <= 1:
            block = ""
        if len(block) > max_chars:
            block = block[: max_chars - 3].rstrip() + "..."
        cache["key"] = cache_key
        cache["fetched_at"] = now
        cache["block"] = block
        if block:
            runtime_ctx.last_memory_focus_user_id = user_id
            runtime_ctx.last_memory_focus_label = label
            block_hash = hashlib.sha1(block.encode("utf-8", errors="ignore")).hexdigest()
            if block_hash != runtime_ctx.last_memory_hash:
                runtime_ctx.memory_updates_applied += 1
                runtime_ctx.last_memory_hash = block_hash
            else:
                runtime_ctx.memory_updates_ignored += 1
            logger.debug(
                "[livekit.mem] room=%s user=%s summary=%s facts=%s chars=%s",
                room_name,
                user_id,
                "yes" if bool(summary_text) else "no",
                len(facts),
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
    return _RuntimeDeps(
        base_settings=base_settings,
        lk_settings=lk_settings,
        instructions=instructions,
        google=google,
        silero=silero,
        health=health,
    )


class _LizaVoiceAgent:  # runtime base class will be created dynamically per process
    pass


async def _handle_session(ctx: Any) -> None:
    configure_logging()
    deps = _load_runtime_deps()
    base_settings = deps.base_settings
    lk_settings = deps.lk_settings
    health = deps.health

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
    agent_instance: Any | None = None
    local_participant_identity = ""
    memory_store: Any | None = None
    memory_cache: dict[str, Any] = {"key": "", "fetched_at": 0.0, "block": ""}

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
                runtime_ctx.last_applied_reason = str(packet_obj.get("reason") or runtime_ctx.last_received_reason or "")
                runtime_block = _build_runtime_context_block(
                    packet_obj,
                    max_chars=lk_settings.agent_runtime_context_max_chars,
                )
                memory_block = await _build_voice_memory_overlay_block(
                    memory_store=memory_store,
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

    server = AgentServer()

    server.rtc_session(agent_name=lk_settings.agent_name)(_handle_session)  # type: ignore[misc]

    agents.cli.run_app(server)
