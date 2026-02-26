from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, List, Protocol
from ..prompts.memory import (
    FACT_EXTRACTOR_SCHEMA_HINT,
    FACT_EXTRACTOR_SYSTEM_PROMPT,
    PERSONA_SELF_FACT_EXTRACTOR_SYSTEM_PROMPT,
    build_fact_extractor_user_prompt,
    build_persona_self_fact_extractor_user_prompt,
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
    about_target: str = "self"
    directness: str = "explicit"
    evidence_quote: str = ""


@dataclass(slots=True)
class ExtractionResult:
    facts: List[FactCandidate]
    diagnostics: "ExtractionDiagnostics | None" = None


@dataclass(slots=True)
class ExtractionDiagnostics:
    backend_name: str
    model_name: str
    latency_ms: int
    llm_attempted: bool
    llm_ok: bool
    json_valid: bool
    fallback_used: bool
    error: str = ""
    llm_fact_count: int = 0
    returned_fact_count: int = 0


class _JsonChatBackend(Protocol):
    async def json_chat(
        self,
        messages: list[dict[str, str]],
        schema_hint: str,
        temperature: float = 0.1,
        max_output_tokens: int = 900,
    ) -> dict[str, object] | None: ...


class MemoryExtractor:
    """Extracts durable user/persona facts via a structured-output LLM backend."""

    def __init__(
        self,
        enabled: bool,
        llm: _JsonChatBackend | Any,
        candidate_limit: int,
    ) -> None:
        self.enabled = enabled
        self.llm = llm
        self.candidate_limit = max(1, candidate_limit)

    async def start(self) -> None:
        start_fn = getattr(self.llm, "start", None)
        if callable(start_fn):
            await start_fn()
        return

    async def close(self) -> None:
        close_fn = getattr(self.llm, "close", None)
        if callable(close_fn):
            await close_fn()
        return

    @property
    def backend_name(self) -> str:
        raw = str(getattr(self.llm, "backend_name", "") or "").strip().lower()
        if raw:
            return raw
        cls_name = self.llm.__class__.__name__.casefold()
        if "gemini" in cls_name:
            return "gemini"
        if "ollama" in cls_name:
            return "ollama"
        return "llm"

    @property
    def model_name(self) -> str:
        model = str(getattr(self.llm, "model", "") or "").strip()
        return model or ""

    @staticmethod
    def _sanitize_text(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        return cleaned[:1600]

    @staticmethod
    def _sanitize_evidence_quote(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        return cleaned[:220]

    @staticmethod
    def _normalize_about_target(value: str, *, persona_self: bool) -> str:
        raw = str(value or "").strip().casefold()
        default = "assistant_self" if persona_self else "self"
        if not raw:
            return default
        aliases = {
            "speaker": "self",
            "user": "self",
            "user_self": "self",
            "me": "self",
            "myself": "self",
            "assistant": "assistant_self",
            "persona": "assistant_self",
            "bot": "assistant_self",
            "another_person": "other",
        }
        normalized = aliases.get(raw, raw)
        allowed = {"self", "assistant_self", "other", "unknown"}
        return normalized if normalized in allowed else default

    @staticmethod
    def _normalize_directness(value: str) -> str:
        raw = str(value or "").strip().casefold()
        aliases = {
            "direct": "explicit",
            "stated": "explicit",
            "implied": "implicit",
            "hinted": "implicit",
            "deduced": "inferred",
            "guess": "inferred",
        }
        normalized = aliases.get(raw, raw or "explicit")
        return normalized if normalized in {"explicit", "implicit", "inferred"} else "explicit"

    @staticmethod
    def _directness_rank(value: str) -> int:
        return {"inferred": 1, "implicit": 2, "explicit": 3}.get(MemoryExtractor._normalize_directness(value), 3)

    @staticmethod
    def _heuristic_identity_facts(text: str, *, persona_self: bool) -> list[FactCandidate]:
        src = " ".join((text or "").strip().split())
        if not src:
            return []
        lowered = src.casefold()
        out: dict[str, FactCandidate] = {}

        def _put(
            key: str,
            value: str,
            *,
            confidence: float,
            importance: float,
            evidence_quote: str = "",
        ) -> None:
            v = str(value or "").strip()
            if not v:
                return
            fact_key = f"identity:{_normalize_key(key)}"
            if not fact_key:
                return
            cand = FactCandidate(
                key=fact_key,
                value=v[:280],
                fact_type="identity",
                confidence=_clamp(confidence, 0.0, 1.0),
                importance=_clamp(importance, 0.0, 1.0),
                about_target="assistant_self" if persona_self else "self",
                directness="explicit",
                evidence_quote=MemoryExtractor._sanitize_evidence_quote(evidence_quote or v),
            )
            prev = out.get(cand.key)
            if prev is None or (cand.confidence + cand.importance) > (prev.confidence + prev.importance):
                out[cand.key] = cand

        # Name extraction (UA/RU/EN) for first-person self-identification.
        name_patterns = [
            r"\b(?:мене\s+звати|меня\s+зовут)\s+([A-Za-zА-Яа-яІіЇїЄєҐґ'`-]{2,40})\b",
            r"\b(?:my\s+name\s+is)\s+([A-Za-zА-Яа-яІіЇїЄєҐґ'`-]{2,40})\b",
        ]
        # "я/я це/i am/i'm" patterns are less reliable, keep lower confidence and only short names.
        soft_name_patterns = [
            r"\bя\s*(?:це\s+)?([A-Za-zА-Яа-яІіЇїЄєҐґ'`-]{2,24})\b",
            r"\b(?:i am|i'm)\s+([A-Za-zА-Яа-яІіЇїЄєҐґ'`-]{2,24})\b",
        ]
        for pattern in name_patterns:
            m = re.search(pattern, src, flags=re.IGNORECASE | re.UNICODE)
            if m:
                _put("name", m.group(1), confidence=0.97 if not persona_self else 0.95, importance=0.92)
                break
        else:
            for pattern in soft_name_patterns:
                m = re.search(pattern, src, flags=re.IGNORECASE | re.UNICODE)
                if not m:
                    continue
                token = str(m.group(1) or "").strip(" .,!?:;")
                if token and len(token) <= 24:
                    _put(
                        "name",
                        token,
                        confidence=0.70 if not persona_self else 0.66,
                        importance=0.80,
                        evidence_quote=token,
                    )
                    break

        # Age extraction (UA/RU/EN) for first-person claims.
        age_patterns = [
            r"\b(?:мені|мне)\s+(\d{1,3})\b",
            r"\b(?:i am|i'm)\s+(\d{1,3})\b",
            r"\b(\d{1,3})\s*(?:рок(?:ів|и)?|лет|years?\s+old)\b",
        ]
        for pattern in age_patterns:
            m = re.search(pattern, lowered, flags=re.IGNORECASE | re.UNICODE)
            if not m:
                continue
            try:
                age_val = int(m.group(1))
            except (TypeError, ValueError):
                continue
            if 3 <= age_val <= 120:
                _put(
                    "age",
                    str(age_val),
                    confidence=0.95 if not persona_self else 0.90,
                    importance=0.90,
                    evidence_quote=m.group(0),
                )
                break

        return list(out.values())

    @staticmethod
    def _merge_fact_candidates(candidates: list[FactCandidate]) -> list[FactCandidate]:
        unique: dict[str, FactCandidate] = {}
        for candidate in candidates:
            if not candidate.key or not candidate.value:
                continue
            prev = unique.get(candidate.key)
            if prev is None:
                unique[candidate.key] = candidate
                continue
            prev_score = (prev.confidence + prev.importance, MemoryExtractor._directness_rank(prev.directness))
            next_score = (candidate.confidence + candidate.importance, MemoryExtractor._directness_rank(candidate.directness))
            if next_score > prev_score:
                unique[candidate.key] = candidate
        return sorted(
            unique.values(),
            key=lambda item: (item.importance, item.confidence, MemoryExtractor._directness_rank(item.directness)),
            reverse=True,
        )

    def _parse_fact_payload(self, payload: dict[str, object], *, persona_self: bool) -> list[FactCandidate]:
        raw_facts = payload.get("facts")
        if not isinstance(raw_facts, list):
            return []

        parsed: list[FactCandidate] = []
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
            about_target = self._normalize_about_target(str(item.get("about_target", "")), persona_self=persona_self)
            directness = self._normalize_directness(str(item.get("directness", "")))
            evidence_quote = self._sanitize_evidence_quote(str(item.get("evidence_quote", "")).strip())

            key_raw = str(item.get("key", "")).strip() or value
            key = _normalize_key(key_raw)
            if not key or not value:
                continue

            parsed.append(
                FactCandidate(
                    key=f"{fact_type}:{key}",
                    value=value[:280],
                    fact_type=fact_type,
                    confidence=_clamp(confidence, 0.0, 1.0),
                    importance=_clamp(importance, 0.0, 1.0),
                    about_target=about_target,
                    directness=directness,
                    evidence_quote=evidence_quote or self._sanitize_evidence_quote(value),
                )
            )
        return parsed

    async def _extract_with_prompt(
        self,
        *,
        text: str,
        preferred_language: str,
        system_prompt: str,
        user_prompt: str,
        heuristic_facts: list[FactCandidate] | None = None,
        persona_self: bool = False,
    ) -> ExtractionResult | None:
        if not self.enabled:
            return None

        text = self._sanitize_text(text)
        if len(text) < 4:
            return None

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
        started = time.perf_counter()
        payload: dict[str, object] | None = None
        llm_ok = False
        json_valid = False
        fallback_used = False
        error_text = ""
        llm_fact_count = 0
        try:
            payload = await self.llm.json_chat(
                messages,
                schema_hint=FACT_EXTRACTOR_SCHEMA_HINT,
                temperature=0.1,
                max_output_tokens=800,
            )
            llm_ok = True
            json_valid = isinstance(payload, dict)
        except Exception as exc:
            error_text = str(exc)[:220]
            payload = None

        if payload is None or not isinstance(payload, dict):
            fallback_used = True
            facts = self._merge_fact_candidates(list(heuristic_facts or []))
            diagnostics = ExtractionDiagnostics(
                backend_name=self.backend_name,
                model_name=self.model_name,
                latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
                llm_attempted=True,
                llm_ok=llm_ok,
                json_valid=json_valid,
                fallback_used=fallback_used,
                error=error_text,
                llm_fact_count=0,
                returned_fact_count=len(facts[: self.candidate_limit]),
            )
            if facts:
                return ExtractionResult(facts=facts[: self.candidate_limit], diagnostics=diagnostics)
            return ExtractionResult(facts=[], diagnostics=diagnostics)

        parsed = self._parse_fact_payload(payload, persona_self=persona_self)
        llm_fact_count = len(parsed)
        if heuristic_facts:
            parsed.extend(heuristic_facts)
        facts = self._merge_fact_candidates(parsed)
        diagnostics = ExtractionDiagnostics(
            backend_name=self.backend_name,
            model_name=self.model_name,
            latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
            llm_attempted=True,
            llm_ok=llm_ok,
            json_valid=True,
            fallback_used=False,
            error=error_text,
            llm_fact_count=llm_fact_count,
            returned_fact_count=len(facts[: self.candidate_limit]),
        )
        return ExtractionResult(facts=facts[: self.candidate_limit], diagnostics=diagnostics)

    @staticmethod
    def _build_dialogue_window_lines(dialogue_context: list[dict[str, object]] | None) -> list[str]:
        if not dialogue_context:
            return []
        out: list[str] = []
        for row in dialogue_context[-12:]:
            if not isinstance(row, dict):
                continue
            role = str(row.get("role", "")).strip().lower() or "user"
            label = str(row.get("author_label", "") or ("Assistant" if role == "assistant" else "User")).strip()
            content = MemoryExtractor._sanitize_text(str(row.get("content", "") or ""))
            if not content:
                continue
            out.append(f"{label} ({role}): {content}")
        return out[-12:]

    async def extract_user_facts(
        self,
        user_text: str,
        preferred_language: str,
        *,
        dialogue_context: list[dict[str, object]] | None = None,
    ) -> ExtractionResult | None:
        text = self._sanitize_text(user_text)
        heuristics = self._heuristic_identity_facts(text, persona_self=False)
        dialogue_window_lines = self._build_dialogue_window_lines(dialogue_context)
        return await self._extract_with_prompt(
            text=text,
            preferred_language=preferred_language,
            system_prompt=FACT_EXTRACTOR_SYSTEM_PROMPT,
            user_prompt=build_fact_extractor_user_prompt(preferred_language, text, dialogue_window_lines),
            heuristic_facts=heuristics,
            persona_self=False,
        )

    async def extract_persona_self_facts(
        self,
        assistant_text: str,
        preferred_language: str,
        *,
        dialogue_context: list[dict[str, object]] | None = None,
    ) -> ExtractionResult | None:
        text = self._sanitize_text(assistant_text)
        heuristics = self._heuristic_identity_facts(text, persona_self=True)
        dialogue_window_lines = self._build_dialogue_window_lines(dialogue_context)
        return await self._extract_with_prompt(
            text=text,
            preferred_language=preferred_language,
            system_prompt=PERSONA_SELF_FACT_EXTRACTOR_SYSTEM_PROMPT,
            user_prompt=build_persona_self_fact_extractor_user_prompt(preferred_language, text, dialogue_window_lines),
            heuristic_facts=heuristics,
            persona_self=True,
        )

    async def extract(self, user_text: str, preferred_language: str) -> ExtractionResult | None:
        # Backward-compatible alias for existing callers.
        return await self.extract_user_facts(user_text, preferred_language)
