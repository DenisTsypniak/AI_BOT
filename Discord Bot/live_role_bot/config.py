from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Set

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
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


def _env_str_tuple(name: str, aliases: tuple[str, ...] = ()) -> tuple[str, ...]:
    raw = (_env_lookup(name, aliases) or "").strip()
    if not raw:
        return ()
    items: list[str] = []
    for chunk in raw.split(","):
        value = chunk.strip()
        if value:
            items.append(value)
    return tuple(items)


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
    text_chat_dialogue_enabled: bool

    text_llm_backend: str
    text_ollama_base_url: str
    text_ollama_model: str
    text_ollama_timeout_seconds: int
    text_ollama_temperature: float
    text_ollama_max_output_tokens: int

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
    memory_extractor_backend: str
    memory_extractor_dialogue_window_messages: int
    memory_ollama_base_url: str
    memory_ollama_model: str
    memory_ollama_timeout_seconds: int
    memory_ollama_temperature: float
    memory_extractor_dry_run_enabled: bool
    memory_extractor_audit_enabled: bool
    memory_fact_moderation_v2_enabled: bool
    memory_fact_key_whitelist: tuple[str, ...]
    memory_fact_key_blacklist: tuple[str, ...]
    memory_persona_self_facts_enabled: bool
    memory_biography_summary_enabled: bool
    memory_biography_summary_max_chars: int
    memory_biography_summary_refresh_min_interval_seconds: int
    summary_enabled: bool
    memory_cross_server_dialogue_summary_fallback_enabled: bool
    summary_min_new_user_messages: int
    summary_window_messages: int
    summary_max_chars: int
    persona_growth_enabled: bool
    persona_growth_shadow_mode: bool
    persona_relationship_enabled: bool
    persona_episodic_enabled: bool
    persona_retrieval_enabled: bool
    persona_reflection_enabled: bool
    persona_reflection_apply_enabled: bool
    persona_trait_drift_enabled: bool
    persona_admin_commands_enabled: bool
    persona_id: str
    persona_policy_version: int
    persona_allowed_admin_user_ids: Set[int]
    persona_prompt_cache_ttl_seconds: int
    persona_text_prompt_budget_chars: int
    persona_voice_prompt_budget_chars: int
    persona_reflection_min_interval_minutes: int
    persona_reflection_min_new_messages: int
    persona_reflection_llm_proposer_enabled: bool
    persona_reflection_llm_temperature: float
    persona_reflection_llm_max_output_tokens: int
    persona_reflection_llm_message_sample_limit: int
    persona_decay_enabled: bool
    persona_decay_min_interval_minutes: int
    persona_episode_recall_reconfirm_enabled: bool
    persona_episode_recall_reconfirm_throttle_seconds: int
    persona_relationship_daily_influence_cap: float
    persona_episode_text_top_k: int
    persona_episode_voice_top_k: int
    persona_queue_isolation_enabled: bool
    persona_event_queue_maxsize: int

    voice_enabled: bool
    voice_auto_join_on_mention: bool
    voice_auto_capture: bool
    voice_send_transcripts_to_text: bool
    voice_bridge_memory_stt_enabled: bool
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
            text_chat_dialogue_enabled=_env_bool("TEXT_CHAT_DIALOGUE_ENABLED", True),
            text_llm_backend=_env_str("TEXT_LLM_BACKEND", "gemini").lower(),
            text_ollama_base_url=_env_str(
                "TEXT_OLLAMA_BASE_URL",
                "http://127.0.0.1:11434",
                aliases=("MEMORY_OLLAMA_BASE_URL",),
            ),
            text_ollama_model=_env_str(
                "TEXT_OLLAMA_MODEL",
                "",
                aliases=("MEMORY_OLLAMA_MODEL",),
            ),
            text_ollama_timeout_seconds=_env_int(
                "TEXT_OLLAMA_TIMEOUT_SECONDS",
                90,
                aliases=("MEMORY_OLLAMA_TIMEOUT_SECONDS",),
            ),
            text_ollama_temperature=_env_float(
                "TEXT_OLLAMA_TEMPERATURE",
                0.35,
            ),
            text_ollama_max_output_tokens=_env_int(
                "TEXT_OLLAMA_MAX_OUTPUT_TOKENS",
                0,
            ),
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
            memory_extractor_backend=_env_str("MEMORY_EXTRACTOR_BACKEND", "gemini").lower(),
            memory_extractor_dialogue_window_messages=_env_int("MEMORY_EXTRACTOR_DIALOGUE_WINDOW_MESSAGES", 6),
            memory_ollama_base_url=_env_str("MEMORY_OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            memory_ollama_model=_env_str("MEMORY_OLLAMA_MODEL", "qwen2.5:7b-instruct"),
            memory_ollama_timeout_seconds=_env_int("MEMORY_OLLAMA_TIMEOUT_SECONDS", 45),
            memory_ollama_temperature=_env_float("MEMORY_OLLAMA_TEMPERATURE", 0.1),
            memory_extractor_dry_run_enabled=_env_bool("MEMORY_EXTRACTOR_DRY_RUN_ENABLED", False),
            memory_extractor_audit_enabled=_env_bool("MEMORY_EXTRACTOR_AUDIT_ENABLED", True),
            memory_fact_moderation_v2_enabled=_env_bool("MEMORY_FACT_MODERATION_V2_ENABLED", True),
            memory_fact_key_whitelist=_env_str_tuple("MEMORY_FACT_KEY_WHITELIST"),
            memory_fact_key_blacklist=_env_str_tuple("MEMORY_FACT_KEY_BLACKLIST"),
            memory_persona_self_facts_enabled=_env_bool("MEMORY_PERSONA_SELF_FACTS_ENABLED", True),
            memory_biography_summary_enabled=_env_bool("MEMORY_BIOGRAPHY_SUMMARY_ENABLED", True),
            memory_biography_summary_max_chars=_env_int("MEMORY_BIOGRAPHY_SUMMARY_MAX_CHARS", 520),
            memory_biography_summary_refresh_min_interval_seconds=_env_int(
                "MEMORY_BIOGRAPHY_SUMMARY_REFRESH_MIN_INTERVAL_SECONDS",
                90,
            ),
            summary_enabled=_env_bool("SUMMARY_ENABLED", True),
            memory_cross_server_dialogue_summary_fallback_enabled=_env_bool(
                "MEMORY_CROSS_SERVER_DIALOGUE_SUMMARY_FALLBACK_ENABLED",
                False,
                aliases=("SUMMARY_CROSS_SERVER_FALLBACK_ENABLED",),
            ),
            summary_min_new_user_messages=_env_int("SUMMARY_MIN_NEW_USER_MESSAGES", 5),
            summary_window_messages=_env_int("SUMMARY_WINDOW_MESSAGES", 24),
            summary_max_chars=_env_int("SUMMARY_MAX_CHARS", 1100),
            persona_growth_enabled=_env_bool("PERSONA_GROWTH_ENABLED", False),
            persona_growth_shadow_mode=_env_bool("PERSONA_GROWTH_SHADOW_MODE", True),
            persona_relationship_enabled=_env_bool("PERSONA_RELATIONSHIP_ENABLED", True),
            persona_episodic_enabled=_env_bool("PERSONA_EPISODIC_ENABLED", False),
            persona_retrieval_enabled=_env_bool("PERSONA_RETRIEVAL_ENABLED", False),
            persona_reflection_enabled=_env_bool("PERSONA_REFLECTION_ENABLED", False),
            persona_reflection_apply_enabled=_env_bool("PERSONA_REFLECTION_APPLY_ENABLED", False),
            persona_trait_drift_enabled=_env_bool("PERSONA_TRAIT_DRIFT_ENABLED", False),
            persona_admin_commands_enabled=_env_bool("PERSONA_ADMIN_COMMANDS_ENABLED", False),
            persona_id=_env_str("PERSONA_ID", "liza"),
            persona_policy_version=_env_int("PERSONA_POLICY_VERSION", 1),
            persona_allowed_admin_user_ids=_env_id_set("PERSONA_ALLOWED_ADMIN_USER_IDS"),
            persona_prompt_cache_ttl_seconds=_env_int("PERSONA_PROMPT_CACHE_TTL_SECONDS", 18),
            persona_text_prompt_budget_chars=_env_int("PERSONA_TEXT_PROMPT_BUDGET_CHARS", 700),
            persona_voice_prompt_budget_chars=_env_int("PERSONA_VOICE_PROMPT_BUDGET_CHARS", 260),
            persona_reflection_min_interval_minutes=_env_int(
                "PERSONA_REFLECTION_MIN_INTERVAL_MINUTES",
                60,
                aliases=("PERSONA_REFLECTION_MIN_INTERVAL_MIN",),
            ),
            persona_reflection_min_new_messages=_env_int("PERSONA_REFLECTION_MIN_NEW_MESSAGES", 30),
            persona_reflection_llm_proposer_enabled=_env_bool("PERSONA_REFLECTION_LLM_PROPOSER_ENABLED", False),
            persona_reflection_llm_temperature=_env_float("PERSONA_REFLECTION_LLM_TEMPERATURE", 0.12),
            persona_reflection_llm_max_output_tokens=_env_int("PERSONA_REFLECTION_LLM_MAX_OUTPUT_TOKENS", 1200),
            persona_reflection_llm_message_sample_limit=_env_int("PERSONA_REFLECTION_LLM_MESSAGE_SAMPLE_LIMIT", 18),
            persona_decay_enabled=_env_bool("PERSONA_DECAY_ENABLED", False),
            persona_decay_min_interval_minutes=_env_int(
                "PERSONA_DECAY_MIN_INTERVAL_MINUTES",
                180,
                aliases=("PERSONA_DECAY_MIN_INTERVAL_MIN",),
            ),
            persona_episode_recall_reconfirm_enabled=_env_bool("PERSONA_EPISODE_RECALL_RECONFIRM_ENABLED", True),
            persona_episode_recall_reconfirm_throttle_seconds=_env_int(
                "PERSONA_EPISODE_RECALL_RECONFIRM_THROTTLE_SECONDS",
                45,
            ),
            persona_relationship_daily_influence_cap=_env_float("PERSONA_RELATIONSHIP_DAILY_INFLUENCE_CAP", 1.0),
            persona_episode_text_top_k=_env_int("PERSONA_EPISODE_TEXT_TOP_K", 2),
            persona_episode_voice_top_k=_env_int("PERSONA_EPISODE_VOICE_TOP_K", 1),
            persona_queue_isolation_enabled=_env_bool("PERSONA_QUEUE_ISOLATION_ENABLED", False),
            persona_event_queue_maxsize=_env_int("PERSONA_EVENT_QUEUE_MAXSIZE", 260),
            voice_enabled=_env_bool("VOICE_ENABLED", True),
            voice_auto_join_on_mention=_env_bool(
                "VOICE_AUTO_JOIN_ON_MENTION",
                True,
                aliases=("VOICE_AUTO_LISTEN_ON_JOIN",),
            ),
            voice_auto_capture=_env_bool("VOICE_AUTO_CAPTURE", True),
            voice_send_transcripts_to_text=_env_bool("VOICE_SEND_TRANSCRIPTS_TO_TEXT", False),
            voice_bridge_memory_stt_enabled=_env_bool("VOICE_BRIDGE_MEMORY_STT_ENABLED", True),
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

        if self.text_llm_backend not in {"gemini", "ollama"}:
            raise ValueError("TEXT_LLM_BACKEND must be 'gemini' or 'ollama'")
        if self.text_ollama_timeout_seconds < 5:
            raise ValueError("TEXT_OLLAMA_TIMEOUT_SECONDS must be >= 5")
        if self.text_ollama_temperature < 0.0 or self.text_ollama_temperature > 1.0:
            raise ValueError("TEXT_OLLAMA_TEMPERATURE must be in [0, 1]")
        if self.text_ollama_max_output_tokens < 0:
            raise ValueError("TEXT_OLLAMA_MAX_OUTPUT_TOKENS must be >= 0")
        if self.text_ollama_max_output_tokens and self.text_ollama_max_output_tokens < 64:
            raise ValueError("TEXT_OLLAMA_MAX_OUTPUT_TOKENS must be 0 or >= 64")
        if self.text_llm_backend == "ollama" and not self.text_ollama_model.strip():
            raise ValueError("TEXT_OLLAMA_MODEL cannot be empty when TEXT_LLM_BACKEND=ollama")

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
        if self.memory_extractor_backend not in {"gemini", "ollama"}:
            raise ValueError("MEMORY_EXTRACTOR_BACKEND must be 'gemini' or 'ollama'")
        if self.memory_extractor_dialogue_window_messages < 0 or self.memory_extractor_dialogue_window_messages > 24:
            raise ValueError("MEMORY_EXTRACTOR_DIALOGUE_WINDOW_MESSAGES must be in [0, 24]")
        if self.memory_ollama_timeout_seconds < 5:
            raise ValueError("MEMORY_OLLAMA_TIMEOUT_SECONDS must be >= 5")
        if self.memory_ollama_temperature < 0.0 or self.memory_ollama_temperature > 1.0:
            raise ValueError("MEMORY_OLLAMA_TEMPERATURE must be in [0, 1]")
        if self.memory_extractor_backend == "ollama" and not self.memory_ollama_model.strip():
            raise ValueError("MEMORY_OLLAMA_MODEL cannot be empty when MEMORY_EXTRACTOR_BACKEND=ollama")
        if len(self.memory_fact_key_whitelist) > 256:
            raise ValueError("MEMORY_FACT_KEY_WHITELIST has too many entries")
        if len(self.memory_fact_key_blacklist) > 256:
            raise ValueError("MEMORY_FACT_KEY_BLACKLIST has too many entries")
        if self.memory_biography_summary_max_chars < 160:
            raise ValueError("MEMORY_BIOGRAPHY_SUMMARY_MAX_CHARS must be >= 160")
        if self.memory_biography_summary_refresh_min_interval_seconds < 0:
            raise ValueError("MEMORY_BIOGRAPHY_SUMMARY_REFRESH_MIN_INTERVAL_SECONDS must be >= 0")

        if self.summary_min_new_user_messages < 1:
            raise ValueError("SUMMARY_MIN_NEW_USER_MESSAGES must be >= 1")
        if self.summary_window_messages < 6:
            raise ValueError("SUMMARY_WINDOW_MESSAGES must be >= 6")
        if self.summary_max_chars < 200:
            raise ValueError("SUMMARY_MAX_CHARS must be >= 200")
        if not self.persona_id.strip():
            raise ValueError("PERSONA_ID cannot be empty")
        if self.persona_policy_version < 1:
            raise ValueError("PERSONA_POLICY_VERSION must be >= 1")
        if self.persona_prompt_cache_ttl_seconds < 1:
            raise ValueError("PERSONA_PROMPT_CACHE_TTL_SECONDS must be >= 1")
        if self.persona_text_prompt_budget_chars < 120:
            raise ValueError("PERSONA_TEXT_PROMPT_BUDGET_CHARS must be >= 120")
        if self.persona_voice_prompt_budget_chars < 60:
            raise ValueError("PERSONA_VOICE_PROMPT_BUDGET_CHARS must be >= 60")
        if self.persona_reflection_min_interval_minutes < 1:
            raise ValueError("PERSONA_REFLECTION_MIN_INTERVAL_MINUTES must be >= 1")
        if self.persona_reflection_min_new_messages < 1:
            raise ValueError("PERSONA_REFLECTION_MIN_NEW_MESSAGES must be >= 1")
        if self.persona_reflection_llm_temperature < 0.0 or self.persona_reflection_llm_temperature > 1.0:
            raise ValueError("PERSONA_REFLECTION_LLM_TEMPERATURE must be in [0, 1]")
        if self.persona_reflection_llm_max_output_tokens < 256 or self.persona_reflection_llm_max_output_tokens > 4096:
            raise ValueError("PERSONA_REFLECTION_LLM_MAX_OUTPUT_TOKENS must be in [256, 4096]")
        if (
            self.persona_reflection_llm_message_sample_limit < 4
            or self.persona_reflection_llm_message_sample_limit > 40
        ):
            raise ValueError("PERSONA_REFLECTION_LLM_MESSAGE_SAMPLE_LIMIT must be in [4, 40]")
        if self.persona_decay_min_interval_minutes < 1 or self.persona_decay_min_interval_minutes > 24 * 60:
            raise ValueError("PERSONA_DECAY_MIN_INTERVAL_MINUTES must be in [1, 1440]")
        if (
            self.persona_episode_recall_reconfirm_throttle_seconds < 5
            or self.persona_episode_recall_reconfirm_throttle_seconds > 600
        ):
            raise ValueError("PERSONA_EPISODE_RECALL_RECONFIRM_THROTTLE_SECONDS must be in [5, 600]")
        if self.persona_relationship_daily_influence_cap < 0.1 or self.persona_relationship_daily_influence_cap > 5.0:
            raise ValueError("PERSONA_RELATIONSHIP_DAILY_INFLUENCE_CAP must be in [0.1, 5.0]")
        if self.persona_episode_text_top_k < 0 or self.persona_episode_text_top_k > 6:
            raise ValueError("PERSONA_EPISODE_TEXT_TOP_K must be in [0, 6]")
        if self.persona_episode_voice_top_k < 0 or self.persona_episode_voice_top_k > 3:
            raise ValueError("PERSONA_EPISODE_VOICE_TOP_K must be in [0, 3]")
        if self.persona_event_queue_maxsize < 20 or self.persona_event_queue_maxsize > 2000:
            raise ValueError("PERSONA_EVENT_QUEUE_MAXSIZE must be in [20, 2000]")

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
