from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

from ...config import Settings, _env_bool, _env_float, _env_int, _env_lookup, _env_str


DEFAULT_GEMINI_API_REALTIME_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"


def _clean_url(value: str) -> str:
    return value.strip().rstrip("/")


def _normalize_google_realtime_model(value: str) -> str:
    model = value.strip()
    if not model:
        return DEFAULT_GEMINI_API_REALTIME_MODEL
    # Existing Discord native-audio config uses Google REST-style names that don't always match LiveKit plugin examples.
    if "native-audio" in model and model.startswith("models/"):
        return DEFAULT_GEMINI_API_REALTIME_MODEL
    # LiveKit Google plugin uses Gemini API by default (`vertexai=False`).
    # If a Vertex-style model sneaks in, remap to a Gemini API live model to avoid 1008 policy errors.
    if model.startswith("gemini-live-"):
        return DEFAULT_GEMINI_API_REALTIME_MODEL
    return model


@dataclass(slots=True)
class LiveKitAgentSettings:
    enabled: bool
    url: str
    api_key: str
    api_secret: str
    worker_name: str
    agent_name: str
    room_prefix: str
    google_api_key: str
    google_realtime_model: str
    voice: str
    temperature: float
    use_silero_vad: bool
    auto_subscribe_audio_only: bool
    health_log_interval_seconds: int
    bridge_enabled: bool
    bridge_control_channel: str
    bridge_context_sync_enabled: bool
    bridge_context_topic: str
    bridge_context_min_interval_ms: int
    bridge_context_force_interval_seconds: int
    agent_runtime_context_injection_enabled: bool
    agent_runtime_context_max_chars: int
    bot_history_json_path: Path

    @classmethod
    def from_env(cls, base_settings: Settings) -> "LiveKitAgentSettings":
        google_api_key = (_env_lookup("GOOGLE_API_KEY") or "").strip() or base_settings.gemini_api_key
        # LiveKit's Gemini plugin expects a Gemini API key via GOOGLE_API_KEY.
        model = _normalize_google_realtime_model(
            _env_str("LIVEKIT_GOOGLE_REALTIME_MODEL", "").strip() or base_settings.gemini_live_model
        )
        return cls(
            enabled=_env_bool("LIVEKIT_ENABLED", False),
            url=_clean_url(_env_str("LIVEKIT_URL", "")),
            api_key=_env_str("LIVEKIT_API_KEY", ""),
            api_secret=_env_str("LIVEKIT_API_SECRET", ""),
            worker_name=_env_str("LIVEKIT_WORKER_NAME", "liza-livekit-worker"),
            agent_name=_env_str("LIVEKIT_AGENT_NAME", "liza-livekit-agent"),
            room_prefix=_env_str("LIVEKIT_ROOM_PREFIX", "liza-voice"),
            google_api_key=google_api_key,
            google_realtime_model=model,
            voice=_env_str("LIVEKIT_GOOGLE_VOICE", base_settings.gemini_live_voice or "Puck"),
            temperature=_env_float("LIVEKIT_GOOGLE_TEMPERATURE", base_settings.gemini_live_temperature),
            use_silero_vad=_env_bool("LIVEKIT_USE_SILERO_VAD", False),
            auto_subscribe_audio_only=_env_bool("LIVEKIT_AUTO_SUBSCRIBE_AUDIO_ONLY", True),
            health_log_interval_seconds=_env_int("LIVEKIT_HEALTH_LOG_INTERVAL_SECONDS", 30),
            bridge_enabled=_env_bool("LIVEKIT_BRIDGE_ENABLED", False),
            bridge_control_channel=_env_str("LIVEKIT_BRIDGE_CONTROL_CHANNEL", "bridge-control"),
            bridge_context_sync_enabled=_env_bool("LIVEKIT_BRIDGE_CONTEXT_SYNC_ENABLED", True),
            bridge_context_topic=_env_str("LIVEKIT_BRIDGE_CONTEXT_TOPIC", "bridge-context"),
            bridge_context_min_interval_ms=_env_int("LIVEKIT_BRIDGE_CONTEXT_MIN_INTERVAL_MS", 1200),
            bridge_context_force_interval_seconds=_env_int("LIVEKIT_BRIDGE_CONTEXT_FORCE_INTERVAL_SECONDS", 12),
            agent_runtime_context_injection_enabled=_env_bool("LIVEKIT_AGENT_CONTEXT_INJECTION_ENABLED", True),
            agent_runtime_context_max_chars=_env_int("LIVEKIT_AGENT_CONTEXT_MAX_CHARS", 420),
            bot_history_json_path=Path(
                _env_str("LIVEKIT_BOT_HISTORY_JSON_PATH", str(base_settings.bot_history_json_path))
            ).expanduser(),
        )

    def validate(self) -> None:
        if not self.enabled:
            raise ValueError("LIVEKIT_ENABLED is false; LiveKit runtime is disabled")
        if not self.url:
            raise ValueError("LIVEKIT_URL is required for LiveKit runtime")
        if not self.api_key:
            raise ValueError("LIVEKIT_API_KEY is required for LiveKit runtime")
        if not self.api_secret:
            raise ValueError("LIVEKIT_API_SECRET is required for LiveKit runtime")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is required for LiveKit Gemini runtime")
        if not self.google_realtime_model:
            raise ValueError("LIVEKIT_GOOGLE_REALTIME_MODEL (or GEMINI_LIVE_MODEL fallback) cannot be empty")
        if self.health_log_interval_seconds < 5:
            raise ValueError("LIVEKIT_HEALTH_LOG_INTERVAL_SECONDS must be >= 5")
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError("LIVEKIT_GOOGLE_TEMPERATURE must be in [0, 2]")
        if self.bridge_context_min_interval_ms < 100 or self.bridge_context_min_interval_ms > 60000:
            raise ValueError("LIVEKIT_BRIDGE_CONTEXT_MIN_INTERVAL_MS must be in [100, 60000]")
        if self.bridge_context_force_interval_seconds < 1 or self.bridge_context_force_interval_seconds > 300:
            raise ValueError("LIVEKIT_BRIDGE_CONTEXT_FORCE_INTERVAL_SECONDS must be in [1, 300]")
        if self.agent_runtime_context_max_chars < 120 or self.agent_runtime_context_max_chars > 1200:
            raise ValueError("LIVEKIT_AGENT_CONTEXT_MAX_CHARS must be in [120, 1200]")

    def validate_bridge(self) -> None:
        if not self.bridge_enabled:
            raise ValueError("LIVEKIT_BRIDGE_ENABLED is false; Discord<->LiveKit bridge is disabled")
        if not self.url:
            raise ValueError("LIVEKIT_URL is required for LiveKit bridge")
        if not self.api_key:
            raise ValueError("LIVEKIT_API_KEY is required for LiveKit bridge")
        if not self.api_secret:
            raise ValueError("LIVEKIT_API_SECRET is required for LiveKit bridge")
        if not self.room_prefix:
            raise ValueError("LIVEKIT_ROOM_PREFIX cannot be empty for bridge")
        if self.health_log_interval_seconds < 5:
            raise ValueError("LIVEKIT_HEALTH_LOG_INTERVAL_SECONDS must be >= 5")
        if not self.bridge_control_channel.strip():
            raise ValueError("LIVEKIT_BRIDGE_CONTROL_CHANNEL cannot be empty")
        if not self.bridge_context_topic.strip():
            raise ValueError("LIVEKIT_BRIDGE_CONTEXT_TOPIC cannot be empty")

    def export_env(self) -> None:
        os.environ["LIVEKIT_URL"] = self.url
        os.environ["LIVEKIT_API_KEY"] = self.api_key
        os.environ["LIVEKIT_API_SECRET"] = self.api_secret
        # Always export the resolved Google key (GOOGLE_API_KEY if set, otherwise GEMINI_API_KEY fallback).
        # `setdefault()` is not enough because `.env` may contain `GOOGLE_API_KEY=` (blank string).
        if self.google_api_key:
            os.environ["GOOGLE_API_KEY"] = self.google_api_key
            # Avoid noisy warning from google-genai when both keys are set in the same process.
            # LiveKit runtime uses GOOGLE_API_KEY explicitly after resolving fallback from GEMINI_API_KEY.
            with contextlib.suppress(Exception):
                if os.environ.get("GEMINI_API_KEY"):
                    os.environ.pop("GEMINI_API_KEY", None)
