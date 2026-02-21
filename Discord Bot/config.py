from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Set

from dotenv import load_dotenv


load_dotenv()


def _as_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _as_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _as_id_set(name: str) -> Set[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return set()
    result: Set[int] = set()
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        try:
            result.add(int(value))
        except ValueError:
            continue
    return result


@dataclass(slots=True)
class Settings:
    discord_token: str
    command_prefix: str

    ollama_base_url: str
    ollama_model: str
    ollama_timeout_seconds: int
    ollama_temperature: float
    ollama_num_ctx: int

    sqlite_path: Path
    max_history_messages: int
    max_response_chars: int

    mention_only: bool
    auto_reply_channel_ids: Set[int]

    default_system_prompt: str

    voice_auto_speak_default: bool
    tts_voice: str
    tts_rate: str
    max_tts_chars: int

    @classmethod
    def from_env(cls) -> "Settings":
        sqlite_path = Path(os.getenv("SQLITE_PATH", "./data/memory.db")).expanduser()
        return cls(
            discord_token=os.getenv("DISCORD_TOKEN", "").strip(),
            command_prefix=os.getenv("DISCORD_COMMAND_PREFIX", "!").strip() or "!",
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip(),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b").strip(),
            ollama_timeout_seconds=_as_int("OLLAMA_TIMEOUT_SECONDS", 90),
            ollama_temperature=_as_float("OLLAMA_TEMPERATURE", 0.5),
            ollama_num_ctx=_as_int("OLLAMA_NUM_CTX", 8192),
            sqlite_path=sqlite_path,
            max_history_messages=_as_int("MAX_HISTORY_MESSAGES", 20),
            max_response_chars=_as_int("MAX_RESPONSE_CHARS", 4000),
            mention_only=_as_bool("BOT_MENTION_ONLY", True),
            auto_reply_channel_ids=_as_id_set("AUTO_REPLY_CHANNEL_IDS"),
            default_system_prompt=os.getenv(
                "DEFAULT_SYSTEM_PROMPT",
                "You are a helpful Discord AI assistant. Keep context, answer clearly, and adapt to user style.",
            ).strip(),
            voice_auto_speak_default=_as_bool("VOICE_AUTO_SPEAK_DEFAULT", True),
            tts_voice=os.getenv("TTS_VOICE", "en-US-AriaNeural").strip(),
            tts_rate=os.getenv("TTS_RATE", "+0%").strip(),
            max_tts_chars=_as_int("MAX_TTS_CHARS", 500),
        )

    def validate(self) -> None:
        if not self.discord_token:
            raise ValueError("DISCORD_TOKEN is required in .env")
        if self.max_history_messages < 1:
            raise ValueError("MAX_HISTORY_MESSAGES must be >= 1")
        if self.max_response_chars < 200:
            raise ValueError("MAX_RESPONSE_CHARS must be >= 200")
