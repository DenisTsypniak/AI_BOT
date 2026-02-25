from __future__ import annotations

import json
from typing import Any

from .json_loader import load_prompt_json

_DEFAULTS = {
    "persona_reflection_llm_schema_hint_object": {
        "proposal_type": "persona_reflection_dry_run_v1",
        "dry_run": True,
        "overlay_summary_candidate": "string",
        "observations": {
            "relationship_note_candidates": [
                {
                    "guild_id": "string",
                    "user_id": "string",
                    "summary_hint": "string",
                    "confidence": 0.0,
                    "evidence_samples": 0,
                }
            ]
        },
        "trait_drift_candidates": [
            {
                "trait_key": "string",
                "delta": 0.0,
                "confidence": 0.0,
                "reason": "string",
                "evidence": {"kind": "string"},
            }
        ],
        "episode_promotion_candidates": [
            {
                "episode_id": 0,
                "from_status": "candidate",
                "to_status": "confirmed",
                "confidence": 0.0,
                "reason": "string",
            }
        ],
    },
    "persona_reflection_llm_system_prompt": (
        "You are a cautious persona-growth reflection proposer for a Discord AI character. "
        "You do not change core identity or safety rules. "
        "Your job is to propose small, evidence-based, bounded updates for a DRY-RUN only. "
        "Prefer no change when evidence is weak or inconsistent. "
        "Never invent facts or message IDs. "
        "Trait deltas must be tiny and conservative. "
        "Only propose episode promotions from candidate to confirmed when evidence is strong enough. "
        "Return a JSON object only."
    ),
    "persona_reflection_llm_user_prompt_template": (
        "Generate a DRY-RUN persona reflection proposal JSON.\n"
        "Constraints:\n"
        "- proposal_type must be `persona_reflection_dry_run_v1`\n"
        "- dry_run must be true\n"
        "- Do not include markdown\n"
        "- Keep trait_drift_candidates conservative (tiny deltas only)\n"
        "- If uncertain, return empty candidate lists\n\n"
        "Context packet:\n{context_json}\n\n"
        "Deterministic baseline proposal (safe fallback reference, you may keep/adjust):\n{baseline_json}\n\n"
        "Return JSON."
    ),
}


def _cfg() -> dict[str, object]:
    return load_prompt_json("persona.json", _DEFAULTS)


_CFG = _cfg()
_REFLECTION_SCHEMA_OBJ = _CFG.get(
    "persona_reflection_llm_schema_hint_object",
    _DEFAULTS["persona_reflection_llm_schema_hint_object"],
)
if not isinstance(_REFLECTION_SCHEMA_OBJ, dict):
    _REFLECTION_SCHEMA_OBJ = _DEFAULTS["persona_reflection_llm_schema_hint_object"]

PERSONA_REFLECTION_LLM_SCHEMA_HINT = json.dumps(_REFLECTION_SCHEMA_OBJ, ensure_ascii=False, separators=(",", ":"))
PERSONA_REFLECTION_LLM_SYSTEM_PROMPT = str(
    _CFG.get(
        "persona_reflection_llm_system_prompt",
        _DEFAULTS["persona_reflection_llm_system_prompt"],
    )
)


def build_persona_reflection_llm_user_prompt(context: dict[str, Any], baseline: dict[str, Any]) -> str:
    template = str(
        _cfg().get(
            "persona_reflection_llm_user_prompt_template",
            _DEFAULTS["persona_reflection_llm_user_prompt_template"],
        )
    )
    context_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    baseline_json = json.dumps(baseline, ensure_ascii=False, separators=(",", ":"))
    return template.format(context_json=context_json, baseline_json=baseline_json)

