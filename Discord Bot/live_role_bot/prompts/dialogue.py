from __future__ import annotations

from typing import Any

from .json_loader import load_prompt_json

_DEFAULTS: dict[str, Any] = {
    "text_conversation_behavior": (
        "Conversation behavior: be present and natural, keep momentum, ask meaningful follow-up questions, "
        "support the user emotionally when needed, and keep discussion intellectually honest."
    ),
    "text_debate_behavior": (
        "Debate behavior: when disagreement appears, provide one clear argument and one respectful counterpoint, "
        "then invite the user to respond."
    ),
    "voice_session_behavior_lines": [
        "Voice mode: respond immediately after user stop. Keep emotional presence.",
        "Voice brevity rule: 1 short sentence. 2 short sentences only when absolutely needed.",
        "Do not give long explanations unless user explicitly asks for details.",
        "Always complete the last sentence naturally; never stop mid-sentence.",
        "Live behavior: keep listening while speaking; if user continues, rebuild response from new details.",
        "Discussion mode: if user disagrees, provide one concise counterpoint and one clarifying question.",
    ],
    "known_participant_memory_header": "Known participant memory:",
    "rp_canon_section_template": "RP CANON (from bot_history.json, highest priority):\n{rp_canon_prompt}",
    "language_rule_template": "Always answer in {preferred_language} unless user explicitly requests another language.",
    "role_profile_templates": {
        "name": "Role name: {name}",
        "goal": "Role goal: {goal}",
        "style": "Role style: {style}",
        "constraints": "Role constraints: {constraints}",
    },
    "known_participant_fact_line_template": "- {label}: {fact_text}",
    "user_dialogue_summary_line_template": "User dialogue memory summary: {summary_text}",
    "user_biography_summary_line_template": "User biography memory summary: {summary_text}",
    "persona_biography_summary_line_template": "Your persistent autobiography summary: {summary_text}",
    "relevant_facts_section_header": "User memory facts relevant for this turn:",
    "persona_relevant_facts_section_header": "Your autobiographical continuity facts:",
    "relevant_fact_line_template": "- [{fact_type} | c={confidence:.2f}] {fact_value}",
}


def _cfg() -> dict[str, Any]:
    return load_prompt_json("dialogue.json", _DEFAULTS)


_CFG = _cfg()

TEXT_CONVERSATION_BEHAVIOR = str(_CFG.get("text_conversation_behavior", _DEFAULTS["text_conversation_behavior"]))
TEXT_DEBATE_BEHAVIOR = str(_CFG.get("text_debate_behavior", _DEFAULTS["text_debate_behavior"]))
VOICE_SESSION_BEHAVIOR_LINES = tuple(
    str(item).strip()
    for item in (_CFG.get("voice_session_behavior_lines") or _DEFAULTS["voice_session_behavior_lines"])
    if str(item).strip()
)
KNOWN_PARTICIPANT_MEMORY_HEADER = str(
    _CFG.get("known_participant_memory_header", _DEFAULTS["known_participant_memory_header"])
)


def build_rp_canon_section(rp_canon_prompt: str) -> str:
    rp = rp_canon_prompt.strip()
    if not rp:
        return ""
    template = str(_cfg().get("rp_canon_section_template", _DEFAULTS["rp_canon_section_template"]))
    return template.format(rp_canon_prompt=rp)


def build_language_rule(preferred_language: str) -> str:
    template = str(_cfg().get("language_rule_template", _DEFAULTS["language_rule_template"]))
    return template.format(preferred_language=preferred_language)


def build_role_profile_lines(role_profile: dict[str, str]) -> tuple[str, str, str, str]:
    raw_templates = _cfg().get("role_profile_templates")
    templates = raw_templates if isinstance(raw_templates, dict) else _DEFAULTS["role_profile_templates"]
    return (
        str(templates.get("name", _DEFAULTS["role_profile_templates"]["name"])).format(name=role_profile["name"]),
        str(templates.get("goal", _DEFAULTS["role_profile_templates"]["goal"])).format(goal=role_profile["goal"]),
        str(templates.get("style", _DEFAULTS["role_profile_templates"]["style"])).format(style=role_profile["style"]),
        str(templates.get("constraints", _DEFAULTS["role_profile_templates"]["constraints"])).format(
            constraints=role_profile["constraints"]
        ),
    )


def build_known_participant_fact_line(label: str, fact_text: str) -> str:
    template = str(
        _cfg().get("known_participant_fact_line_template", _DEFAULTS["known_participant_fact_line_template"])
    )
    return template.format(label=label, fact_text=fact_text)


def build_user_dialogue_summary_line(summary_text: str) -> str:
    template = str(
        _cfg().get("user_dialogue_summary_line_template", _DEFAULTS["user_dialogue_summary_line_template"])
    )
    return template.format(summary_text=summary_text)


def build_user_biography_summary_line(summary_text: str) -> str:
    template = str(
        _cfg().get("user_biography_summary_line_template", _DEFAULTS["user_biography_summary_line_template"])
    )
    return template.format(summary_text=summary_text)


def build_persona_biography_summary_line(summary_text: str) -> str:
    template = str(
        _cfg().get("persona_biography_summary_line_template", _DEFAULTS["persona_biography_summary_line_template"])
    )
    return template.format(summary_text=summary_text)


def build_relevant_facts_section(facts: list[dict[str, Any]]) -> str:
    cfg = _cfg()
    header = str(cfg.get("relevant_facts_section_header", _DEFAULTS["relevant_facts_section_header"]))
    line_template = str(cfg.get("relevant_fact_line_template", _DEFAULTS["relevant_fact_line_template"]))
    fact_lines = [
        line_template.format(
            fact_type=fact["fact_type"],
            confidence=float(fact["confidence"]),
            fact_value=fact["fact_value"],
        )
        for fact in facts
    ]
    return header + "\n" + "\n".join(fact_lines)


def build_persona_relevant_facts_section(facts: list[dict[str, Any]]) -> str:
    cfg = _cfg()
    header = str(cfg.get("persona_relevant_facts_section_header", _DEFAULTS["persona_relevant_facts_section_header"]))
    line_template = str(cfg.get("relevant_fact_line_template", _DEFAULTS["relevant_fact_line_template"]))
    fact_lines = [
        line_template.format(
            fact_type=fact["fact_type"],
            confidence=float(fact["confidence"]),
            fact_value=fact["fact_value"],
        )
        for fact in facts
    ]
    return header + "\n" + "\n".join(fact_lines)
