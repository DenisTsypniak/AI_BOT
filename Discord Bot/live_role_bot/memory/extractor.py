from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from ..services.gemini_client import GeminiClient
from ..prompts.memory import (
    FACT_EXTRACTOR_SCHEMA_HINT,
    FACT_EXTRACTOR_SYSTEM_PROMPT,
    build_fact_extractor_user_prompt,
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_key(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text.strip().casefold())
    cleaned = re.sub(r"[^\w'\-\s]", "", collapsed, flags=re.UNICODE)
    return cleaned[:100].strip()


@dataclass(slots=True)
class FactCandidate:
    key: str
    value: str
    fact_type: str
    confidence: float
    importance: float


@dataclass(slots=True)
class ExtractionResult:
    facts: List[FactCandidate]


class MemoryExtractor:
    """Extracts durable user facts via Gemini in strict JSON mode."""

    def __init__(
        self,
        enabled: bool,
        llm: GeminiClient,
        candidate_limit: int,
    ) -> None:
        self.enabled = enabled
        self.llm = llm
        self.candidate_limit = max(1, candidate_limit)

    async def start(self) -> None:
        return

    async def close(self) -> None:
        return

    @staticmethod
    def _sanitize_text(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        return cleaned[:1600]

    async def extract(self, user_text: str, preferred_language: str) -> ExtractionResult | None:
        if not self.enabled:
            return None

        text = self._sanitize_text(user_text)
        if len(text) < 4:
            return None

        messages = [
            {
                "role": "system",
                "content": FACT_EXTRACTOR_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_fact_extractor_user_prompt(preferred_language, text),
            },
        ]

        payload = await self.llm.json_chat(
            messages,
            schema_hint=FACT_EXTRACTOR_SCHEMA_HINT,
            temperature=0.1,
            max_output_tokens=800,
        )
        if payload is None:
            return None

        raw_facts = payload.get("facts")
        if not isinstance(raw_facts, list):
            return ExtractionResult(facts=[])

        unique: dict[str, FactCandidate] = {}

        for item in raw_facts:
            if not isinstance(item, dict):
                continue
            value = str(item.get("value", "")).strip()
            fact_type = str(item.get("type", "fact")).strip().lower() or "fact"

            confidence_raw = item.get("confidence", 0.0)
            importance_raw = item.get("importance", 0.0)

            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError):
                confidence = 0.0
            try:
                importance = float(importance_raw)
            except (TypeError, ValueError):
                importance = 0.0

            key_raw = str(item.get("key", "")).strip() or value
            key = _normalize_key(key_raw)
            if not key:
                continue
            if not value:
                continue

            candidate = FactCandidate(
                key=f"{fact_type}:{key}",
                value=value[:280],
                fact_type=fact_type,
                confidence=_clamp(confidence, 0.0, 1.0),
                importance=_clamp(importance, 0.0, 1.0),
            )

            prev = unique.get(candidate.key)
            if prev is None or (candidate.confidence + candidate.importance) > (prev.confidence + prev.importance):
                unique[candidate.key] = candidate

        facts = sorted(
            unique.values(),
            key=lambda item: (item.importance, item.confidence),
            reverse=True,
        )
        return ExtractionResult(facts=facts[: self.candidate_limit])
