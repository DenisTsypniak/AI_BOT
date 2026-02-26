from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_key(value: str) -> str:
    return _normalize_text(value).casefold()


def _match_pattern(pattern: str, fact_key: str) -> bool:
    p = _normalize_key(pattern)
    key = _normalize_key(fact_key)
    if not p or not key:
        return False
    if p.endswith("*"):
        return key.startswith(p[:-1])
    return key == p


@dataclass(slots=True)
class CandidateModerationDecision:
    action: str
    reason: str
    field_kind: str
    min_confidence: float

    @property
    def accepted(self) -> bool:
        return self.action == "accept"


@dataclass(slots=True)
class CandidateModerationInput:
    fact_key: str
    fact_value: str
    fact_type: str
    about_target: str
    directness: str
    confidence: float
    importance: float
    evidence_quote: str
    owner_kind: str
    speaker_role: str


class FactModerationPolicyV2:
    """Quality gate for extractor candidates before they reach memory storage."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        whitelist_patterns: Iterable[str] = (),
        blacklist_patterns: Iterable[str] = (),
    ) -> None:
        self.enabled = bool(enabled)
        self.whitelist_patterns = tuple(p for p in (_normalize_text(x) for x in whitelist_patterns) if p)
        self.blacklist_patterns = tuple(p for p in (_normalize_text(x) for x in blacklist_patterns) if p)

        # Per-field thresholds tuned for implicit extraction. These are intentionally conservative
        # for identity-like fields that are easy to poison (name/age/location).
        self.thresholds: dict[str, dict[str, float]] = {
            "default": {"explicit": 0.40, "implicit": 0.58, "inferred": 0.72},
            "name": {"explicit": 0.76, "implicit": 0.95, "inferred": 0.99},
            "age": {"explicit": 0.86, "implicit": 0.98, "inferred": 0.995},
            "location": {"explicit": 0.68, "implicit": 0.84, "inferred": 0.91},
            # Persona self-facts can be a bit more permissive because continuity is the goal.
            "persona_default": {"explicit": 0.34, "implicit": 0.52, "inferred": 0.66},
        }

    @classmethod
    def from_settings(cls, settings: object) -> "FactModerationPolicyV2":
        return cls(
            enabled=bool(getattr(settings, "memory_fact_moderation_v2_enabled", True)),
            whitelist_patterns=getattr(settings, "memory_fact_key_whitelist", ()) or (),
            blacklist_patterns=getattr(settings, "memory_fact_key_blacklist", ()) or (),
        )

    @staticmethod
    def _field_kind(fact_key: str) -> str:
        key = _normalize_key(fact_key)
        if not key:
            return "default"
        if key.endswith(":name") or "identity:name" in key:
            return "name"
        if key.endswith(":age") or "identity:age" in key:
            return "age"
        location_markers = ("location", "city", "country", "region", "from_", "lives", "residence", "place")
        if any(marker in key for marker in location_markers):
            return "location"
        return "default"

    def _threshold_for(self, field_kind: str, directness: str, owner_kind: str) -> float:
        d = str(directness or "explicit").strip().lower() or "explicit"
        if d not in {"explicit", "implicit", "inferred"}:
            d = "explicit"
        if str(owner_kind or "").strip().lower() == "persona":
            persona_map = self.thresholds.get("persona_default", {})
            if field_kind in {"default", "location"}:
                return float(persona_map.get(d, 0.34))
        threshold_map = self.thresholds.get(field_kind, self.thresholds["default"])
        return float(threshold_map.get(d, self.thresholds["default"][d]))

    def evaluate(self, data: CandidateModerationInput) -> CandidateModerationDecision:
        if not self.enabled:
            return CandidateModerationDecision("accept", "moderation_disabled", "default", 0.0)

        fact_key = _normalize_key(data.fact_key)
        fact_value = _normalize_text(data.fact_value)
        directness = str(data.directness or "explicit").strip().lower() or "explicit"
        about_target = str(data.about_target or "unknown").strip().lower() or "unknown"
        owner_kind = str(data.owner_kind or "user").strip().lower() or "user"

        if not fact_key or not fact_value:
            return CandidateModerationDecision("reject", "empty_key_or_value", "default", 0.0)
        if len(fact_value) < 2:
            return CandidateModerationDecision("reject", "value_too_short", "default", 0.0)
        if len(fact_value) > 280:
            return CandidateModerationDecision("reject", "value_too_long", "default", 0.0)

        for pattern in self.blacklist_patterns:
            if _match_pattern(pattern, fact_key):
                return CandidateModerationDecision("reject", f"blacklist:{pattern}", "default", 0.0)
        if self.whitelist_patterns:
            if not any(_match_pattern(pattern, fact_key) for pattern in self.whitelist_patterns):
                return CandidateModerationDecision("reject", "not_in_whitelist", "default", 0.0)

        field_kind = self._field_kind(fact_key)
        min_conf = self._threshold_for(field_kind, directness, owner_kind)
        conf = _clamp(float(data.confidence))
        imp = _clamp(float(data.importance))

        if conf < min_conf:
            return CandidateModerationDecision("reject", "confidence_below_threshold", field_kind, min_conf)

        # Guard common high-risk identity misfires.
        if field_kind == "name":
            if any(ch.isdigit() for ch in fact_value):
                return CandidateModerationDecision("reject", "name_contains_digits", field_kind, min_conf)
            if len(fact_value) > 48:
                return CandidateModerationDecision("reject", "name_too_long", field_kind, min_conf)
            if directness != "explicit" and conf < 0.90:
                return CandidateModerationDecision("reject", "name_requires_explicit_or_high_conf", field_kind, min_conf)

        if field_kind == "age":
            digits = "".join(ch for ch in fact_value if ch.isdigit())
            if not digits:
                return CandidateModerationDecision("reject", "age_missing_digits", field_kind, min_conf)
            try:
                age = int(digits)
            except ValueError:
                return CandidateModerationDecision("reject", "age_parse_failed", field_kind, min_conf)
            if age < 3 or age > 120:
                return CandidateModerationDecision("reject", "age_out_of_range", field_kind, min_conf)

        if field_kind == "location" and directness in {"implicit", "inferred"} and imp < 0.45:
            return CandidateModerationDecision("reject", "location_low_importance", field_kind, min_conf)

        if about_target == "unknown" and conf < 0.60:
            return CandidateModerationDecision("reject", "unknown_target_low_conf", field_kind, min_conf)

        # Human cards should not ingest obvious assistant-self facts through user pipeline.
        if owner_kind == "user" and about_target == "assistant_self":
            return CandidateModerationDecision("reject", "assistant_self_in_user_card", field_kind, min_conf)

        return CandidateModerationDecision("accept", "accepted", field_kind, min_conf)

