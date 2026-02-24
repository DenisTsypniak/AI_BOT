from __future__ import annotations

from typing import Any

from .json_loader import load_prompt_json


_DEFAULTS: dict[str, Any] = {
    "language_line_template": "Always speak in {preferred_language}.",
    "pacing_line": (
        "Reply immediately when user finishes speaking. "
        "Keep each voice reply very short: 1 sentence, max 2 sentences only if necessary."
    ),
    "brevity_line": (
        "No long explanations unless user explicitly asks for details. "
        "No lists, no long intros, no recap."
    ),
    "completion_line": "Always finish your final sentence naturally. Never end a reply mid-sentence.",
    "live_line": (
        "Stay fully live: keep listening while speaking. "
        "If user continues speaking, immediately adapt and rebuild answer from updated input."
    ),
    "silence_line": (
        "If there is silence, keyboard clicks, breathing, random background noise, coughs, or unclear fragments, "
        "stay silent and keep listening."
    ),
    "echo_line": "Never answer your own echoed voice from speakers or microphone bleed.",
}


def _cfg() -> dict[str, Any]:
    return load_prompt_json("voice.json", _DEFAULTS)


def build_native_audio_system_instruction(base_prompt: str, preferred_language: str) -> str:
    cfg = _cfg()
    language_line = str(cfg.get("language_line_template", _DEFAULTS["language_line_template"])).format(
        preferred_language=preferred_language
    )
    pacing_line = str(cfg.get("pacing_line", _DEFAULTS["pacing_line"]))
    brevity_line = str(cfg.get("brevity_line", _DEFAULTS["brevity_line"]))
    completion_line = str(cfg.get("completion_line", _DEFAULTS["completion_line"]))
    live_line = str(cfg.get("live_line", _DEFAULTS["live_line"]))
    silence_line = str(cfg.get("silence_line", _DEFAULTS["silence_line"]))
    echo_line = str(cfg.get("echo_line", _DEFAULTS["echo_line"]))
    base = base_prompt.strip()
    if not base:
        return (
            f"{language_line}\n{pacing_line}\n{brevity_line}\n"
            f"{completion_line}\n{live_line}\n{silence_line}\n{echo_line}"
        ).strip()
    return (
        f"{base}\n\n{language_line}\n{pacing_line}\n{brevity_line}\n"
        f"{completion_line}\n{live_line}\n{silence_line}\n{echo_line}"
    ).strip()
