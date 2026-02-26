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
                "type": "identity|preference|style_preference|goal|context|relationship|constraint|boundary|episodic|topic",
                "confidence": 0.0,
                "importance": 0.0,
                "about_target": "self|assistant_self|other|unknown",
                "directness": "explicit|implicit|inferred",
                "evidence_quote": "short source quote from the text",
            }
        ]
    },
    "fact_extractor_system_prompt": (
        "Extract only user facts that can improve future personalization. "
        "Prioritize durable facts (identity, preferences, goals, constraints, relationship context) and reusable interaction style preferences "
        "(e.g., likes short replies, wants step-by-step help, language preference). "
        "You may keep lightweight episodic moments only if they are likely useful as future callbacks. "
        "Ignore temporary filler, greetings, obvious noise, and random one-off phrases. "
        "Avoid sensitive data unless the user explicitly shared it and it is directly relevant to future assistance. "
        "Return about_target for each fact (usually self for speaker facts). "
        "Return directness as explicit/implicit/inferred. "
        "Include evidence_quote as a short supporting quote copied from the provided text/dialogue window. "
        "If no durable facts exist, return facts: []."
    ),
    "fact_extractor_user_prompt_template": (
        "Preferred language: {preferred_language}\n"
        "Speaker role: human user\n"
        "Recent dialogue window (oldest -> newest, speaker-labeled):\n{dialogue_window}\n\n"
        "Focus user utterance (extract facts about the speaker only): {text}\n"
        "Return JSON."
    ),
    "persona_self_fact_extractor_system_prompt": (
        "Extract facts that the assistant/persona explicitly claims about itself in first person. "
        "Treat the assistant as a persistent character with continuity. "
        "Keep identity, preferences, biography, habits, relationships, and notable experiences it states as true. "
        "If the assistant says it did something, visited somewhere, likes/dislikes something, or has a past event, "
        "that becomes a valid autobiographical memory fact. "
        "Ignore pure rhetorical filler, meta formatting, and statements about the user unless they also define the persona. "
        "Set about_target=assistant_self for persona self-claims. "
        "Use directness explicit/implicit/inferred and include a short evidence_quote. "
        "Return facts: [] if no clear self-claims are present."
    ),
    "persona_self_fact_extractor_user_prompt_template": (
        "Preferred language: {preferred_language}\n"
        "Recent dialogue window (oldest -> newest, speaker-labeled):\n{dialogue_window}\n\n"
        "Assistant/persona utterance (extract self-facts only): {text}\nReturn JSON."
    ),
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
    "biography_summary_update_system_prompt_template": (
        "Update a persistent biography summary for one {subject_kind}. "
        "Preserve continuity, stable identity, recurring preferences, relationships, and life-history details. "
        "Do not copy raw fact list. Write a compact natural-language profile. "
        "Keep uncertainty low: include only facts supported by memory facts provided. "
        "Output plain text, max {max_chars} chars."
    ),
    "biography_summary_update_user_prompt_template": (
        "Subject kind: {subject_kind}\n"
        "Previous biography summary:\n{previous_summary}\n\n"
        "Memory facts:\n{fact_lines}\n\n"
        "Return updated biography summary."
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
PERSONA_SELF_FACT_EXTRACTOR_SYSTEM_PROMPT = str(
    _CFG.get("persona_self_fact_extractor_system_prompt", _DEFAULTS["persona_self_fact_extractor_system_prompt"])
)
BIOGRAPHY_SUMMARY_UPDATE_SYSTEM_PROMPT_TEMPLATE = str(
    _CFG.get(
        "biography_summary_update_system_prompt_template",
        _DEFAULTS["biography_summary_update_system_prompt_template"],
    )
)


def _format_dialogue_window(dialogue_window_lines: Iterable[str] | None) -> str:
    if not dialogue_window_lines:
        return "(no extra context)"
    joined = "\n".join(str(line) for line in dialogue_window_lines if str(line))
    return joined or "(no extra context)"


def build_fact_extractor_user_prompt(
    preferred_language: str,
    text: str,
    dialogue_window_lines: Iterable[str] | None = None,
) -> str:
    template = str(_cfg().get("fact_extractor_user_prompt_template", _DEFAULTS["fact_extractor_user_prompt_template"]))
    return template.format(
        preferred_language=preferred_language,
        text=text,
        dialogue_window=_format_dialogue_window(dialogue_window_lines),
    )


def build_persona_self_fact_extractor_user_prompt(
    preferred_language: str,
    text: str,
    dialogue_window_lines: Iterable[str] | None = None,
) -> str:
    template = str(
        _cfg().get(
            "persona_self_fact_extractor_user_prompt_template",
            _DEFAULTS["persona_self_fact_extractor_user_prompt_template"],
        )
    )
    return template.format(
        preferred_language=preferred_language,
        text=text,
        dialogue_window=_format_dialogue_window(dialogue_window_lines),
    )


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


def build_biography_summary_update_system_prompt(max_chars: int, subject_kind: str) -> str:
    return BIOGRAPHY_SUMMARY_UPDATE_SYSTEM_PROMPT_TEMPLATE.format(
        max_chars=max_chars,
        subject_kind=subject_kind,
    )


def build_biography_summary_update_user_prompt(
    *,
    subject_kind: str,
    previous_summary: str,
    fact_lines: Iterable[str],
) -> str:
    joined = "\n".join(str(line) for line in fact_lines if str(line))
    template = str(
        _cfg().get(
            "biography_summary_update_user_prompt_template",
            _DEFAULTS["biography_summary_update_user_prompt_template"],
        )
    )
    return template.format(
        subject_kind=subject_kind,
        previous_summary=previous_summary or "(none)",
        fact_lines=joined or "(no facts)",
    )
