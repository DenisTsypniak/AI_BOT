from __future__ import annotations

import json
from typing import Iterable

from .json_loader import load_prompt_json

_DEFAULTS = {
    "fact_extractor_schema_hint_object": {
        "facts": [
            {
                "key": "string",
                "value": "string",
                "type": "identity|preference|goal|context|relationship|constraint",
                "confidence": 0.0,
                "importance": 0.0,
            }
        ]
    },
    "fact_extractor_system_prompt": (
        "Extract only durable user facts that can personalize future dialogue. "
        "Ignore temporary filler and repeated greetings. "
        "If no durable facts exist, return facts: []."
    ),
    "fact_extractor_user_prompt_template": "Preferred language: {preferred_language}\nUser utterance: {text}\nReturn JSON.",
    "json_only_system_prompt_template": (
        "Return only valid JSON object with no markdown and no additional commentary. "
        "Schema hint: {schema_hint}"
    ),
    "summary_update_system_prompt_template": (
        "Update rolling memory summary for one user's ongoing dialogue with assistant. "
        "Keep durable topics, preferences, goals, unresolved items, emotional context. "
        "Output plain text, max {max_chars} chars."
    ),
    "summary_update_user_prompt_template": (
        "Previous summary:\n{previous_summary}\n\nRecent dialogue:\n{dialogue_lines}\n\nReturn updated summary."
    ),
}


def _cfg() -> dict[str, object]:
    return load_prompt_json("memory.json", _DEFAULTS)


_CFG = _cfg()
_SCHEMA_OBJ = _CFG.get("fact_extractor_schema_hint_object", _DEFAULTS["fact_extractor_schema_hint_object"])
if not isinstance(_SCHEMA_OBJ, dict):
    _SCHEMA_OBJ = _DEFAULTS["fact_extractor_schema_hint_object"]

FACT_EXTRACTOR_SCHEMA_HINT = json.dumps(_SCHEMA_OBJ, ensure_ascii=False, separators=(",", ":"))
FACT_EXTRACTOR_SYSTEM_PROMPT = str(
    _CFG.get("fact_extractor_system_prompt", _DEFAULTS["fact_extractor_system_prompt"])
)
JSON_ONLY_SYSTEM_PROMPT_TEMPLATE = str(
    _CFG.get("json_only_system_prompt_template", _DEFAULTS["json_only_system_prompt_template"])
)
SUMMARY_UPDATE_SYSTEM_PROMPT_TEMPLATE = str(
    _CFG.get("summary_update_system_prompt_template", _DEFAULTS["summary_update_system_prompt_template"])
)


def build_fact_extractor_user_prompt(preferred_language: str, text: str) -> str:
    template = str(_cfg().get("fact_extractor_user_prompt_template", _DEFAULTS["fact_extractor_user_prompt_template"]))
    return template.format(preferred_language=preferred_language, text=text)


def build_json_only_system_prompt(schema_hint: str) -> str:
    return JSON_ONLY_SYSTEM_PROMPT_TEMPLATE.format(schema_hint=schema_hint)


def build_summary_update_system_prompt(max_chars: int) -> str:
    return SUMMARY_UPDATE_SYSTEM_PROMPT_TEMPLATE.format(max_chars=max_chars)


def build_summary_update_user_prompt(previous_summary: str, dialogue_lines: Iterable[str]) -> str:
    joined = "\n".join(str(line) for line in dialogue_lines if str(line))
    template = str(_cfg().get("summary_update_user_prompt_template", _DEFAULTS["summary_update_user_prompt_template"]))
    return template.format(
        previous_summary=previous_summary or "(none)",
        dialogue_lines=joined,
    )
