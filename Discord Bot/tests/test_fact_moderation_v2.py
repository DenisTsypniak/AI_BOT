from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live_role_bot.memory.fact_moderation import (  # noqa: E402
    CandidateModerationInput,
    FactModerationPolicyV2,
)


def _candidate(**overrides):  # type: ignore[no-untyped-def]
    base = CandidateModerationInput(
        fact_key="context:night_shifts",
        fact_value="Працює в нічні зміни",
        fact_type="context",
        about_target="self",
        directness="implicit",
        confidence=0.75,
        importance=0.70,
        evidence_quote="знову нічна зміна",
        owner_kind="user",
        speaker_role="user",
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_moderation_v2_rejects_low_confidence_implicit_age() -> None:
    policy = FactModerationPolicyV2()
    decision = policy.evaluate(
        _candidate(
            fact_key="identity:age",
            fact_value="29",
            fact_type="identity",
            directness="implicit",
            confidence=0.80,
            importance=0.8,
        )
    )
    assert decision.accepted is False
    assert decision.field_kind == "age"
    assert decision.reason in {"confidence_below_threshold", "age_requires_explicit_or_high_conf"}


def test_moderation_v2_accepts_explicit_name_with_high_confidence() -> None:
    policy = FactModerationPolicyV2()
    decision = policy.evaluate(
        _candidate(
            fact_key="identity:name",
            fact_value="Микола",
            fact_type="identity",
            directness="explicit",
            confidence=0.95,
            importance=0.8,
            evidence_quote="Мене звати Микола",
        )
    )
    assert decision.accepted is True
    assert decision.field_kind == "name"


def test_moderation_v2_respects_whitelist_and_blacklist() -> None:
    policy = FactModerationPolicyV2(
        whitelist_patterns=("identity:*", "preference:*"),
        blacklist_patterns=("identity:age",),
    )
    blocked = policy.evaluate(
        _candidate(
            fact_key="identity:age",
            fact_value="29",
            fact_type="identity",
            directness="explicit",
            confidence=0.99,
            importance=0.9,
        )
    )
    skipped = policy.evaluate(
        _candidate(
            fact_key="topic:game",
            fact_value="Terraria",
            fact_type="topic",
            directness="explicit",
            confidence=0.9,
            importance=0.6,
        )
    )
    assert blocked.accepted is False
    assert blocked.reason.startswith("blacklist:")
    assert skipped.accepted is False
    assert skipped.reason == "not_in_whitelist"

