from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]

    window = text[:limit]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "), window.rfind("; "))
    if cut >= int(limit * 0.62):
        return window[: cut + 1].strip()

    cut = window.rfind(" ")
    if cut >= int(limit * 0.7):
        return window[:cut].strip()

    return (window[: limit - 3].rstrip() + "...").strip()


def chunk_text(text: str, limit: int = 1900) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) <= limit:
            current += line
            continue
        if current:
            parts.append(current)
            current = ""
        if len(line) <= limit:
            current = line
        else:
            for i in range(0, len(line), limit):
                parts.append(line[i : i + limit])
    if current:
        parts.append(current)
    return parts


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[\w']{2,}", text.casefold(), flags=re.UNICODE)
    stop = {
        "you",
        "me",
        "my",
        "the",
        "and",
        "for",
        "are",
        "with",
        "that",
        "this",
        "have",
    }
    return {word for word in words if word not in stop}


def as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        with contextlib.suppress(ValueError):
            return float(value.strip())
    return default


def as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        with contextlib.suppress(ValueError):
            return int(value.strip())
    return default


def read_text_with_fallback(path: Path) -> str:
    last_exc: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return path.read_text(encoding=encoding)
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to read file: {path}")


def prompt_block_to_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        lines = [str(item).strip() for item in value if str(item).strip()]
        return "\n".join(lines).strip()
    return ""


def load_rp_canon(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        raw = read_text_with_fallback(path)
        payload = json.loads(raw)
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    if not bool(payload.get("enabled", True)):
        return ""

    blocks: list[str] = []
    for key in ("system_prompt", "style_persona_prompt"):
        block = prompt_block_to_text(payload.get(key))
        if block:
            blocks.append(block)
    return "\n\n".join(blocks).strip()


@dataclass(slots=True)
class PendingProfileUpdate:
    guild_id: str
    channel_id: str
    user_id: str
    message_id: int
    user_text: str
    user_label: str = ""
    modality: str = "text"
    source: str = "unknown"
    quality: float = 1.0
    speaker_role: str = "user"
    fact_owner_kind: str = "user"
    fact_owner_id: str = ""
    persona_ingest_enqueued: bool = False


@dataclass(slots=True)
class PendingSummaryUpdate:
    guild_id: str
    channel_id: str
    user_id: str


@dataclass(slots=True)
class VoiceTurnBuffer:
    data: bytearray = field(default_factory=bytearray)
    started_at: float = 0.0
    last_voice_at: float = 0.0
    user_label: str = "User"
    reply_enabled: bool = True
    transcript_source: str = "local_stt"


@dataclass(slots=True)
class PendingVoiceTurn:
    guild_id: int
    channel_id: int
    user_id: int
    user_label: str
    pcm_48k_stereo: bytes
    reply_enabled: bool = True
    transcript_source: str = "local_stt"


@dataclass(slots=True)
class ConversationSessionState:
    mood: str = "neutral"
    energy: str = "medium"
    topic_hint: str = ""
    open_loop: str = ""
    tease_level: str = "low"
    familiarity: str = "new"
    group_vibe: str = "solo"
    last_modality: str = "text"
    turn_count: int = 0
    last_user_text: str = ""
    last_bot_text: str = ""
    recent_speakers: list[str] = field(default_factory=list)
    recent_topics: list[str] = field(default_factory=list)
    callback_moments: list[str] = field(default_factory=list)
    repeated_openers: dict[str, int] = field(default_factory=dict)
