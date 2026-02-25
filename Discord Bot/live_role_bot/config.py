from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Set

from dotenv import load_dotenv


load_dotenv()


def _env_lookup(name: str, aliases: tuple[str, ...] = ()) -> str | None:
    for key in (name, *aliases):
        # Be tolerant to UTF-8 BOM accidentally saved in .env key names.
        for candidate in (key, f"\ufeff{key}"):
            raw = os.getenv(candidate)
            if raw is not None:
                return raw
    return None


def _env_bool(name: str, default: bool, aliases: tuple[str, ...] = ()) -> bool:
    raw = _env_lookup(name, aliases)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, aliases: tuple[str, ...] = ()) -> int:
    raw = _env_lookup(name, aliases)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float, aliases: tuple[str, ...] = ()) -> float:
    raw = _env_lookup(name, aliases)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_id_set(name: str, aliases: tuple[str, ...] = ()) -> Set[int]:
    raw = (_env_lookup(name, aliases) or "").strip()
    if not raw:
        return set()
    result: Set[int] = set()
    for chunk in raw.split(","):
        value = chunk.strip()
        if not value:
            continue
        try:
            result.add(int(value))
        except ValueError:
            continue
    return result


def _clean_token(value: str) -> str:
    cleaned = value.strip()
    if cleaned.lower().startswith("bot "):
        cleaned = cleaned[4:].strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _env_str(name: str, default: str, aliases: tuple[str, ...] = ()) -> str:
    raw = _env_lookup(name, aliases)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


@dataclass(slots=True)
class Settings:
    discord_token: str
    command_prefix: str
    discord_message_content_intent: bool
    discord_members_intent: bool
    mention_only: bool
    auto_reply_channel_ids: Set[int]
    preferred_response_language: str

    gemini_api_key: str
    gemini_base_url: str
    gemini_model: str
    gemini_timeout_seconds: int
    gemini_temperature: float
    gemini_max_output_tokens: int
    gemini_native_audio_enabled: bool
    gemini_live_model: str
    gemini_live_voice: str
    gemini_live_temperature: float
    gemini_live_max_output_tokens: int
    gemini_live_input_sample_rate: int
    gemini_live_vad_silence_ms: int

    sqlite_path: Path
    max_recent_messages: int
    max_response_chars: int

    default_role_id: str
    role_name: str
    role_goal: str
    role_style: str
    role_constraints: str
    system_core_prompt: str
    bot_history_json_path: Path

    memory_enabled: bool
    memory_fact_top_k: int
    memory_candidate_fact_limit: int
    summary_enabled: bool
    summary_min_new_user_messages: int
    summary_window_messages: int
    summary_max_chars: int

    voice_enabled: bool
    voice_auto_join_on_mention: bool
    voice_auto_capture: bool
    voice_send_transcripts_to_text: bool
    voice_silence_rms: int
    voice_silence_ms: int
    voice_min_turn_ms: int
    voice_max_turn_seconds: int
    transcription_min_confidence: float

    local_stt_enabled: bool
    local_stt_model: str
    local_stt_fallback_model: str
    local_stt_device: str
    local_stt_compute_type: str
    local_stt_language: str
    local_stt_max_audio_seconds: int

    plugins_enabled: bool
    plugins_match_threshold: float

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            discord_token=_clean_token(_env_lookup("DISCORD_TOKEN") or ""),
            command_prefix=_env_str("DISCORD_COMMAND_PREFIX", "!"),
            discord_message_content_intent=_env_bool("DISCORD_MESSAGE_CONTENT_INTENT", True),
            discord_members_intent=_env_bool("DISCORD_MEMBERS_INTENT", True),
            mention_only=_env_bool("BOT_MENTION_ONLY", False),
            auto_reply_channel_ids=_env_id_set("AUTO_REPLY_CHANNEL_IDS"),
            preferred_response_language=_env_str("PREFERRED_RESPONSE_LANGUAGE", "Ukrainian"),
            gemini_api_key=_env_str("GEMINI_API_KEY", ""),
            gemini_base_url=_env_str("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"),
            gemini_model=_env_str("GEMINI_MODEL", "gemini-2.5-flash"),
            gemini_timeout_seconds=_env_int("GEMINI_TIMEOUT_SECONDS", 90),
            gemini_temperature=_env_float("GEMINI_TEMPERATURE", 0.6),
            gemini_max_output_tokens=_env_int("GEMINI_MAX_OUTPUT_TOKENS", 0),
            gemini_native_audio_enabled=_env_bool("GEMINI_NATIVE_AUDIO_ENABLED", True),
            gemini_live_model=_env_str(
                "GEMINI_LIVE_MODEL",
                "models/gemini-2.5-flash-native-audio-latest",
            ),
            gemini_live_voice=_env_str("GEMINI_LIVE_VOICE", "Aoede"),
            gemini_live_temperature=_env_float("GEMINI_LIVE_TEMPERATURE", 0.45),
            gemini_live_max_output_tokens=_env_int("GEMINI_LIVE_MAX_OUTPUT_TOKENS", 0),
            gemini_live_input_sample_rate=_env_int("GEMINI_LIVE_INPUT_SAMPLE_RATE", 24000),
            gemini_live_vad_silence_ms=_env_int("GEMINI_LIVE_VAD_SILENCE_MS", 420),
            sqlite_path=Path(_env_str("SQLITE_PATH", "./data/live_dialogue.db")).expanduser(),
            max_recent_messages=_env_int("MAX_RECENT_MESSAGES", 18, aliases=("MAX_HISTORY_MESSAGES",)),
            max_response_chars=_env_int("MAX_RESPONSE_CHARS", 0),
            default_role_id=_env_str("DEFAULT_ROLE_ID", "live-role-default"),
            role_name=_env_str("ROLE_NAME", ""),
            role_goal=_env_str("ROLE_GOAL", ""),
            role_style=_env_str("ROLE_STYLE", ""),
            role_constraints=_env_str("ROLE_CONSTRAINTS", ""),
            system_core_prompt=_env_str("SYSTEM_CORE_PROMPT", "", aliases=("DEFAULT_SYSTEM_PROMPT",)),
            bot_history_json_path=Path(_env_str("BOT_HISTORY_JSON_PATH", "./data/bot_history.json")).expanduser(),
            memory_enabled=_env_bool("MEMORY_ENABLED", True, aliases=("LONG_MEMORY_ENABLED",)),
            memory_fact_top_k=_env_int("MEMORY_FACT_TOP_K", 8, aliases=("PROFILE_FACT_LIMIT",)),
            memory_candidate_fact_limit=_env_int("MEMORY_CANDIDATE_FACT_LIMIT", 10),
            summary_enabled=_env_bool("SUMMARY_ENABLED", True),
            summary_min_new_user_messages=_env_int("SUMMARY_MIN_NEW_USER_MESSAGES", 5),
            summary_window_messages=_env_int("SUMMARY_WINDOW_MESSAGES", 24),
            summary_max_chars=_env_int("SUMMARY_MAX_CHARS", 1100),
            voice_enabled=_env_bool("VOICE_ENABLED", True),
            voice_auto_join_on_mention=_env_bool(
                "VOICE_AUTO_JOIN_ON_MENTION",
                True,
                aliases=("VOICE_AUTO_LISTEN_ON_JOIN",),
            ),
            voice_auto_capture=_env_bool("VOICE_AUTO_CAPTURE", True),
            voice_send_transcripts_to_text=_env_bool("VOICE_SEND_TRANSCRIPTS_TO_TEXT", False),
            voice_silence_rms=_env_int("VOICE_SILENCE_RMS", 95),
            voice_silence_ms=_env_int("VOICE_SILENCE_MS", 520),
            voice_min_turn_ms=_env_int("VOICE_MIN_TURN_MS", 420, aliases=("LOCAL_STT_MIN_AUDIO_MS",)),
            voice_max_turn_seconds=_env_int("VOICE_MAX_TURN_SECONDS", 18),
            transcription_min_confidence=_env_float("TRANSCRIPTION_MIN_CONFIDENCE", 0.46),
            local_stt_enabled=_env_bool("LOCAL_STT_ENABLED", True),
            local_stt_model=_env_str("LOCAL_STT_MODEL", "medium"),
            local_stt_fallback_model=_env_str("LOCAL_STT_FALLBACK_MODEL", "small"),
            local_stt_device=_env_str("LOCAL_STT_DEVICE", "auto"),
            local_stt_compute_type=_env_str("LOCAL_STT_COMPUTE_TYPE", "int8"),
            local_stt_language=_env_str("LOCAL_STT_LANGUAGE", "uk"),
            local_stt_max_audio_seconds=_env_int("LOCAL_STT_MAX_AUDIO_SECONDS", 24),
            plugins_enabled=_env_bool("PLUGINS_ENABLED", True),
            plugins_match_threshold=_env_float("PLUGINS_MATCH_THRESHOLD", 0.82),
        )

    def validate(self) -> None:
        if not self.discord_token:
            raise ValueError("DISCORD_TOKEN is required")
        if self.discord_token == "put_your_discord_bot_token_here":
            raise ValueError("DISCORD_TOKEN is still placeholder")
        if not self.command_prefix.strip():
            raise ValueError("DISCORD_COMMAND_PREFIX cannot be empty")

        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required")
        if self.gemini_api_key == "put_your_gemini_api_key_here":
            raise ValueError("GEMINI_API_KEY is still placeholder")

        if self.gemini_timeout_seconds < 20:
            raise ValueError("GEMINI_TIMEOUT_SECONDS must be >= 20")
        if self.gemini_max_output_tokens < 0:
            raise ValueError("GEMINI_MAX_OUTPUT_TOKENS must be >= 0 (0 disables explicit cap)")
        if self.gemini_max_output_tokens and self.gemini_max_output_tokens < 128:
            raise ValueError("GEMINI_MAX_OUTPUT_TOKENS must be 0 or >= 128")
        if not self.gemini_live_model:
            raise ValueError("GEMINI_LIVE_MODEL cannot be empty")
        if self.gemini_live_input_sample_rate < 8000:
            raise ValueError("GEMINI_LIVE_INPUT_SAMPLE_RATE must be >= 8000")
        if self.gemini_live_vad_silence_ms < 120:
            raise ValueError("GEMINI_LIVE_VAD_SILENCE_MS must be >= 120")
        if self.gemini_live_max_output_tokens < 0:
            raise ValueError("GEMINI_LIVE_MAX_OUTPUT_TOKENS must be >= 0 (0 disables explicit cap)")
        if self.gemini_live_max_output_tokens and self.gemini_live_max_output_tokens < 128:
            raise ValueError("GEMINI_LIVE_MAX_OUTPUT_TOKENS must be 0 or >= 128")

        if self.max_recent_messages < 4:
            raise ValueError("MAX_RECENT_MESSAGES must be >= 4")
        if self.max_response_chars < 0:
            raise ValueError("MAX_RESPONSE_CHARS must be >= 0 (0 disables explicit cap)")
        if self.max_response_chars and self.max_response_chars < 300:
            raise ValueError("MAX_RESPONSE_CHARS must be 0 or >= 300")

        if self.memory_fact_top_k < 1:
            raise ValueError("MEMORY_FACT_TOP_K must be >= 1")
        if self.memory_candidate_fact_limit < 1:
            raise ValueError("MEMORY_CANDIDATE_FACT_LIMIT must be >= 1")

        if self.summary_min_new_user_messages < 1:
            raise ValueError("SUMMARY_MIN_NEW_USER_MESSAGES must be >= 1")
        if self.summary_window_messages < 6:
            raise ValueError("SUMMARY_WINDOW_MESSAGES must be >= 6")
        if self.summary_max_chars < 200:
            raise ValueError("SUMMARY_MAX_CHARS must be >= 200")

        if self.voice_silence_rms < 20:
            raise ValueError("VOICE_SILENCE_RMS must be >= 20")
        if self.voice_silence_ms < 180:
            raise ValueError("VOICE_SILENCE_MS must be >= 180")
        if self.voice_min_turn_ms < 180:
            raise ValueError("VOICE_MIN_TURN_MS must be >= 180")
        if self.voice_max_turn_seconds < 4:
            raise ValueError("VOICE_MAX_TURN_SECONDS must be >= 4")

        if self.transcription_min_confidence < 0.0 or self.transcription_min_confidence > 1.0:
            raise ValueError("TRANSCRIPTION_MIN_CONFIDENCE must be in [0, 1]")
        if self.local_stt_max_audio_seconds < 4:
            raise ValueError("LOCAL_STT_MAX_AUDIO_SECONDS must be >= 4")
        if self.plugins_match_threshold < 0.0 or self.plugins_match_threshold > 1.0:
            raise ValueError("PLUGINS_MATCH_THRESHOLD must be in [0, 1]")
