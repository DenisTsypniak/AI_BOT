from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..prompts.persona import (
    PERSONA_REFLECTION_LLM_SCHEMA_HINT,
    PERSONA_REFLECTION_LLM_SYSTEM_PROMPT,
    build_persona_reflection_llm_user_prompt,
)
from .traits import DEFAULT_TRAIT_CATALOG

logger = logging.getLogger("live_role_bot")


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _tokenize_text(text: str) -> set[str]:
    words = re.findall(r"[\w']{2,}", (text or "").casefold(), flags=re.UNICODE)
    stop = {"the", "and", "for", "that", "this", "with", "you", "your", "just", "like"}
    return {w for w in words if w not in stop}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class PersonaGrowthEngine:
    """Phase 0/1 scaffold: bootstrap persona state + cached prompt snapshots."""

    def __init__(self, settings: Any, memory: Any, core_dna_path: Path, llm: Any | None = None) -> None:
        self.settings = settings
        self.memory = memory
        self.llm = llm
        self.core_dna_path = Path(core_dna_path)
        self.persona_id = str(getattr(settings, "persona_id", "liza") or "liza").strip()
        self._started = False
        self._backend_supported = False
        self._cache: dict[tuple[str, str, str, str, str], tuple[float, str]] = {}
        self._last_reflection_result: dict[str, object] = {}
        self._last_decay_result: dict[str, object] = {}
        self._episode_recall_throttle: dict[int, float] = {}

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.settings, "persona_growth_enabled", False) and self._backend_supported)

    @property
    def retrieval_enabled(self) -> bool:
        return bool(self.enabled and getattr(self.settings, "persona_retrieval_enabled", False))

    def _supports_backend(self) -> bool:
        if str(getattr(self.memory, "backend_name", "")).strip().lower() != "postgres":
            return False
        required = (
            "ensure_persona_mvp_bootstrap",
            "get_persona_global_state",
            "list_persona_traits",
            "get_persona_relationship",
            "get_persona_user_memory_pref",
            "list_persona_reflections",
            "ingest_persona_user_message",
            "list_persona_episode_callbacks",
            "list_persona_episodes",
        )
        return all(callable(getattr(self.memory, name, None)) for name in required)

    def _core_dna_hash(self) -> str:
        try:
            payload = self.core_dna_path.read_bytes()
        except Exception:
            payload = b""
        return hashlib.sha256(payload).hexdigest()

    async def start(self) -> None:
        self._backend_supported = self._supports_backend()
        if not self._backend_supported:
            self._started = True
            return
        if not getattr(self.settings, "persona_growth_enabled", False):
            self._started = True
            return

        ensure_fn = getattr(self.memory, "ensure_persona_mvp_bootstrap", None)
        if callable(ensure_fn):
            await ensure_fn(
                persona_id=self.persona_id,
                core_dna_hash=self._core_dna_hash(),
                core_dna_source=str(self.core_dna_path),
                policy_version=int(getattr(self.settings, "persona_policy_version", 1)),
                trait_catalog_entries=list(DEFAULT_TRAIT_CATALOG),
            )
        self._started = True

    async def close(self) -> None:
        self._cache.clear()
        self._last_reflection_result = {}
        self._last_decay_result = {}
        self._episode_recall_throttle.clear()
        self._started = False

    def status_snapshot(self) -> dict[str, object]:
        return {
            "configured": bool(getattr(self.settings, "persona_growth_enabled", False)),
            "enabled": bool(self.enabled),
            "started": bool(self._started),
            "backend_supported": bool(self._backend_supported),
            "backend": str(getattr(self.memory, "backend_name", type(self.memory).__name__)),
            "persona_id": self.persona_id,
            "shadow_mode": bool(getattr(self.settings, "persona_growth_shadow_mode", True)),
            "retrieval_enabled": bool(getattr(self.settings, "persona_retrieval_enabled", False)),
            "reflection_enabled": bool(getattr(self.settings, "persona_reflection_enabled", False)),
            "reflection_apply_enabled": bool(getattr(self.settings, "persona_reflection_apply_enabled", False)),
            "trait_drift_enabled": bool(getattr(self.settings, "persona_trait_drift_enabled", False)),
            "decay_enabled": bool(getattr(self.settings, "persona_decay_enabled", False)),
            "episode_recall_reconfirm_enabled": bool(
                getattr(self.settings, "persona_episode_recall_reconfirm_enabled", True)
            ),
            "reflection_llm_proposer_enabled": bool(
                getattr(self.settings, "persona_reflection_llm_proposer_enabled", False)
            ),
            "reflection_llm_proposer_available": bool(callable(getattr(self.llm, "json_chat", None))),
            "cache_entries": len(self._cache),
            "last_reflection_status": str(self._last_reflection_result.get("status", "")),
            "last_reflection_id": int(self._last_reflection_result.get("reflection_id", 0) or 0),
            "last_decay_status": str(self._last_decay_result.get("status", "")),
        }

    def invalidate_prompt_cache(self) -> None:
        self._cache.clear()

    def _cache_key(
        self,
        mode: str,
        guild_id: str,
        channel_id: str,
        user_id: str,
        query_fingerprint: str = "",
    ) -> tuple[str, str, str, str, str]:
        return (mode, guild_id, channel_id, user_id, query_fingerprint)

    def _cache_ttl(self) -> float:
        return float(max(1, int(getattr(self.settings, "persona_prompt_cache_ttl_seconds", 18))))

    @staticmethod
    def _query_fingerprint(text: str) -> str:
        cleaned = _collapse_spaces(text)
        if not cleaned:
            return ""
        tokens = sorted(list(_tokenize_text(cleaned)))[:8]
        if not tokens:
            return ""
        return hashlib.sha1((" ".join(tokens)).encode("utf-8", errors="ignore")).hexdigest()[:10]

    @staticmethod
    def _trim_to_budget(text: str, budget_chars: int) -> str:
        cleaned = " ".join((text or "").split())
        if len(cleaned) <= budget_chars:
            return cleaned
        if budget_chars <= 3:
            return cleaned[:budget_chars]
        return cleaned[: budget_chars - 3].rstrip() + "..."

    @staticmethod
    def _fmt_trait_line(row: dict[str, object]) -> str:
        key = str(row.get("trait_key") or "").strip()
        value = float(row.get("current_value") or 0.0)
        conf = float(row.get("confidence") or 0.0)
        status = str(row.get("status") or "emerging")
        return f"- {key}: {value:.2f} ({status}, c={conf:.2f})"

    @staticmethod
    def _trait_retrieval_policy(row: dict[str, object] | None, *, retrieval_enabled: bool) -> dict[str, object]:
        if not isinstance(row, dict) or not row:
            return {"policy": "missing", "reason": "trait_not_found", "prompt_exposure": "?"}
        prompt_exposure = str(row.get("prompt_exposure") or "relevant").strip().lower() or "relevant"
        status = str(row.get("status") or "emerging").strip().lower() or "emerging"
        confidence = float(row.get("confidence") or 0.0)
        if not retrieval_enabled:
            return {
                "policy": "disabled",
                "reason": "persona_retrieval_disabled",
                "prompt_exposure": prompt_exposure,
            }
        if prompt_exposure == "never":
            return {
                "policy": "suppressed",
                "reason": "prompt_exposure_never",
                "prompt_exposure": prompt_exposure,
            }
        if status == "contested":
            return {
                "policy": "suppressed",
                "reason": "trait_status_contested_pending_reconfirmation",
                "prompt_exposure": prompt_exposure,
            }
        if status in {"emerging"} or confidence < 0.45:
            return {
                "policy": "guarded",
                "reason": "low_stability_or_confidence",
                "prompt_exposure": prompt_exposure,
            }
        return {
            "policy": "shown",
            "reason": "eligible_stable_trait",
            "prompt_exposure": prompt_exposure,
        }

    @staticmethod
    def _fmt_relationship_line(row: dict[str, object] | None) -> list[str]:
        if not isinstance(row, dict) or not row:
            return []
        lines = [
            "Relationship memory (current user):",
            (
                "- familiarity={familiarity:.2f}, trust={trust:.2f}, warmth={warmth:.2f}, "
                "banter_license={banter_license:.2f}, c={confidence:.2f}"
            ).format(
                familiarity=float(row.get("familiarity") or 0.0),
                trust=float(row.get("trust") or 0.0),
                warmth=float(row.get("warmth") or 0.0),
                banter_license=float(row.get("banter_license") or 0.0),
                confidence=float(row.get("confidence") or 0.0),
            ),
        ]
        style_notes = str(row.get("preferred_style_notes") or "").strip()
        if style_notes:
            lines.append(f"- style_notes: {style_notes}")
        joke_notes = str(row.get("inside_joke_summary") or "").strip()
        if joke_notes:
            lines.append(f"- inside_jokes: {joke_notes}")
        return lines

    @staticmethod
    def _contains_any(text_cf: str, markers: tuple[str, ...]) -> bool:
        return any(marker in text_cf for marker in markers)

    def _build_relationship_update(
        self,
        *,
        text: str,
        modality: str,
        quality: float,
    ) -> tuple[dict[str, object], bool]:
        cleaned = _collapse_spaces(text)
        text_cf = cleaned.casefold()
        tokens = _tokenize_text(cleaned)
        if not cleaned:
            return ({}, False)

        length = len(cleaned)
        emotional_markers = ("сум", "тривог", "stress", "stressed", "депрес", "anxious", "тяжко", "погано")
        playful_markers = ("лол", "ахах", "haha", "lmao", "bruh", "gg", "крінж", "імба", "мем")
        rude_markers = ("idiot", "retard", "kill yourself", "сука", "дебіл", "лох", "підар", "fuck you")
        gratitude_markers = ("дякую", "спасиб", "thank", "ty", "дякс")
        boundary_markers = ("не хочу", "stop", "відстань", "не питай", "не треба", "leave me")
        self_disclosure_markers = ("я ", "мені", "моя", "i ", "my ", "me ")
        plan_markers = ("завтра", "сьогодні", "потім", "буду", "plan", "tomorrow", "later")

        quality_weight = _clamp(float(quality), 0.0, 1.0)
        if str(modality).strip().lower() == "voice":
            quality_weight = _clamp(quality_weight * 0.95, 0.0, 1.0)
            if quality_weight < max(0.54, float(getattr(self.settings, "transcription_min_confidence", 0.46))):
                quality_weight *= 0.55

        toxicity_risk = 0.0
        if self._contains_any(text_cf, rude_markers):
            toxicity_risk += 0.65
        if cleaned.count("!") >= 3 and any(tok in {"nah", "bro", "bruh"} for tok in tokens):
            toxicity_risk += 0.12
        toxicity_risk = _clamp(toxicity_risk, 0.0, 1.0)

        trust_delta = 0.004
        warmth_delta = 0.003
        familiarity_delta = 0.006 if length >= 20 else 0.003
        banter_delta = 0.0
        support_delta = 0.0
        boundary_delta = 0.0
        topic_delta = 0.0
        signal_conf = 0.40

        if self._contains_any(text_cf, gratitude_markers):
            trust_delta += 0.018
            warmth_delta += 0.012
            signal_conf += 0.12
        if self._contains_any(text_cf, emotional_markers):
            support_delta += 0.020
            warmth_delta += 0.010
            banter_delta -= 0.010
            signal_conf += 0.10
        if self._contains_any(text_cf, playful_markers):
            banter_delta += 0.018
            warmth_delta += 0.008
            signal_conf += 0.08
        if self._contains_any(text_cf, boundary_markers):
            boundary_delta += 0.025
            banter_delta -= 0.015
            signal_conf += 0.14
        if self._contains_any(text_cf, self_disclosure_markers):
            trust_delta += 0.010
            familiarity_delta += 0.006
            signal_conf += 0.08
        if self._contains_any(text_cf, plan_markers):
            topic_delta += 0.012
            familiarity_delta += 0.005

        if "?" in cleaned:
            trust_delta += 0.004
            topic_delta += 0.006
        if toxicity_risk >= 0.65:
            trust_delta -= 0.020
            warmth_delta -= 0.015
            banter_delta -= 0.012
            boundary_delta += 0.018
            signal_conf = max(signal_conf, 0.55)

        if length < 8:
            signal_conf *= 0.55
        elif length < 14:
            signal_conf *= 0.75
        elif length > 180:
            signal_conf += 0.08

        if quality_weight < 0.55:
            signal_conf *= 0.70

        interaction_size = 0.35 + min(1.0, (length / 110.0))
        influence_weight = _clamp(
            quality_weight * interaction_size * (1.0 - (toxicity_risk * 0.70)),
            0.0,
            1.5,
        )
        consistency_score = 0.68 if toxicity_risk < 0.5 else 0.38
        reason_parts: list[str] = []
        if self._contains_any(text_cf, emotional_markers):
            reason_parts.append("emotional_disclosure")
        if self._contains_any(text_cf, playful_markers):
            reason_parts.append("playful_tone")
        if self._contains_any(text_cf, gratitude_markers):
            reason_parts.append("gratitude")
        if self._contains_any(text_cf, boundary_markers):
            reason_parts.append("boundary_signal")
        if self._contains_any(text_cf, plan_markers):
            reason_parts.append("future_or_shared_context")
        if not reason_parts:
            reason_parts.append("general_interaction")

        eligible_for_growth = bool(length >= 8 and not cleaned.startswith(("!", "/", ".", "#")))
        return (
            {
                "familiarity": familiarity_delta,
                "trust": trust_delta,
                "warmth": warmth_delta,
                "banter_license": banter_delta,
                "support_sensitivity": support_delta,
                "boundary_sensitivity": boundary_delta,
                "topic_alignment": topic_delta,
                "signal_confidence": _clamp(signal_conf, 0.0, 1.0),
                "quality_weight": quality_weight,
                "influence_weight": influence_weight,
                "toxicity_risk": toxicity_risk,
                "consistency_score": consistency_score,
                "reason_text": ",".join(reason_parts)[:240],
                "risk_flags": {
                    "contains_question": "?" in cleaned,
                    "text_len": min(length, 9999),
                },
            },
            eligible_for_growth,
        )

    def _build_episode_candidate(
        self,
        *,
        guild_id: str,
        channel_id: str,
        user_id: str,
        message_id: int,
        text: str,
        quality: float,
        relationship_update: dict[str, object],
    ) -> dict[str, object] | None:
        if not bool(getattr(self.settings, "persona_episodic_enabled", False)):
            return None
        cleaned = _collapse_spaces(text)
        if len(cleaned) < 12 or cleaned.startswith(("!", "/", ".", "#")):
            return None
        text_cf = cleaned.casefold()
        tokens = _tokenize_text(cleaned)
        notable = 0.0
        ep_type = "moment"
        privacy_level = "participants_only"
        callback_safety = "safe"
        valence = 0.0

        if any(mark in text_cf for mark in ("завтра", "сьогодні", "буду", "tomorrow", "later", "plan")):
            notable += 0.22
            ep_type = "plan"
        if "?" in cleaned and len(cleaned) > 18:
            notable += 0.10
            ep_type = "question"
        if any(mark in text_cf for mark in ("аха", "ахах", "лол", "haha", "lmao", "bruh")):
            notable += 0.24
            ep_type = "joke"
            valence = max(valence, 0.35)
        if any(mark in text_cf for mark in ("дякую", "thank", "спасиб")):
            notable += 0.14
            valence = max(valence, 0.25)
        if any(mark in text_cf for mark in ("сум", "тривог", "stress", "депрес", "погано")):
            notable += 0.20
            ep_type = "support"
            callback_safety = "careful"
            valence = min(valence, -0.35)
        if len(cleaned) >= 48:
            notable += 0.08
        if cleaned.count("!") >= 2 or "..." in cleaned:
            notable += 0.06
        if any(tok in {"крінж", "імба", "gg", "bruh"} for tok in tokens):
            notable += 0.06
        if float(relationship_update.get("toxicity_risk", 0.0) or 0.0) >= 0.65:
            notable -= 0.22
            callback_safety = "avoid"
        notable *= _clamp(float(quality), 0.0, 1.0)
        if notable < 0.24:
            return None

        short = cleaned[:120]
        title = short
        for sep in ("...", ". ", "! ", "? ", "; "):
            idx = title.find(sep)
            if idx >= 0:
                title = title[: idx + (1 if sep[-1] in ".!?" else 0)]
                break
        title = title.strip(" -")
        if len(title) < 8:
            title = short
        callback_line = title[:110]
        summary = cleaned[:360]
        dedupe_key_seed = f"{guild_id}:{channel_id}:{user_id}:{ep_type}:{message_id}:{callback_line.casefold()}"
        dedupe_key = hashlib.sha1(dedupe_key_seed.encode("utf-8", errors="ignore")).hexdigest()
        importance = _clamp(0.22 + notable * 0.90, 0.0, 1.0)
        vividness = _clamp(0.28 + notable * 0.85, 0.0, 1.0)
        confidence = _clamp(0.30 + notable * 0.70, 0.0, 1.0)
        return {
            "dedupe_key": dedupe_key,
            "episode_type": ep_type,
            "title": title[:140],
            "summary": summary,
            "callback_line": callback_line,
            "status": "candidate",
            "privacy_level": privacy_level,
            "callback_safety": callback_safety,
            "valence": valence,
            "importance": importance,
            "vividness": vividness,
            "confidence": confidence,
            "snippet": cleaned[:220],
        }

    async def ingest_user_message(
        self,
        *,
        guild_id: str,
        channel_id: str,
        user_id: str,
        message_id: int,
        user_text: str,
        user_label: str = "",
        modality: str = "text",
        source: str = "unknown",
        quality: float = 1.0,
    ) -> dict[str, object]:
        if not self.enabled:
            return {"applied": False, "reason": "disabled"}
        if not bool(getattr(self.settings, "persona_relationship_enabled", True)):
            return {"applied": False, "reason": "relationship_disabled"}
        ingest_fn = getattr(self.memory, "ingest_persona_user_message", None)
        if not callable(ingest_fn):
            return {"applied": False, "reason": "unsupported_backend"}

        rel_update, eligible = self._build_relationship_update(text=user_text, modality=modality, quality=quality)
        episode_candidate = self._build_episode_candidate(
            guild_id=guild_id,
            channel_id=channel_id,
            user_id=user_id,
            message_id=message_id,
            text=user_text,
            quality=quality,
            relationship_update=rel_update,
        )
        result = await ingest_fn(
            self.persona_id,
            guild_id,
            channel_id,
            user_id,
            int(message_id),
            user_text,
            user_label=user_label,
            modality=modality,
            source=source,
            quality=float(quality),
            relationship_update=rel_update,
            episode_candidate=episode_candidate,
            daily_influence_cap=float(getattr(self.settings, "persona_relationship_daily_influence_cap", 1.0)),
            eligible_for_growth=eligible,
        )
        if result.get("applied") and not result.get("deduped"):
            self.invalidate_prompt_cache()
        return result

    @staticmethod
    def _parse_dt(value: object) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _supports_reflection_backend(self) -> bool:
        required = (
            "get_persona_reflection_window",
            "mark_persona_reflection_started",
            "update_persona_global_reflection_checkpoint",
            "create_persona_reflection",
            "update_persona_reflection_status",
        )
        return all(callable(getattr(self.memory, name, None)) for name in required)

    def _supports_reflection_apply_backend(self) -> bool:
        required = (
            "get_persona_reflection_details",
            "apply_persona_reflection_changes",
            "update_persona_reflection_status",
            "update_persona_global_reflection_checkpoint",
        )
        return all(callable(getattr(self.memory, name, None)) for name in required)

    def _supports_decay_backend(self) -> bool:
        required = (
            "run_persona_decay_cycle",
            "get_persona_global_state",
        )
        return all(callable(getattr(self.memory, name, None)) for name in required)

    def _supports_episode_recall_reconfirm_backend(self) -> bool:
        return callable(getattr(self.memory, "touch_persona_episode_recalls", None))

    def _episode_recall_reconfirm_enabled(self) -> bool:
        return bool(
            self.enabled
            and getattr(self.settings, "persona_retrieval_enabled", False)
            and getattr(self.settings, "persona_episode_recall_reconfirm_enabled", True)
            and self._supports_episode_recall_reconfirm_backend()
        )

    def _episode_recall_throttle_seconds(self) -> float:
        return float(max(5, int(getattr(self.settings, "persona_episode_recall_reconfirm_throttle_seconds", 45))))

    async def _safe_touch_episode_recalls(self, episode_ids: list[int]) -> None:
        if not self._episode_recall_reconfirm_enabled():
            return
        touch_fn = getattr(self.memory, "touch_persona_episode_recalls", None)
        if not callable(touch_fn):
            return
        try:
            await touch_fn(self.persona_id, episode_ids)
        except Exception as exc:
            logger.debug("Persona episode recall reconfirmation failed: %s", exc)

    def _schedule_episode_recall_reconfirm(self, episode_ids: list[int]) -> None:
        if not self._episode_recall_reconfirm_enabled():
            return
        now = time.monotonic()
        ttl = self._episode_recall_throttle_seconds()
        if len(self._episode_recall_throttle) > 512:
            cutoff = now - (ttl * 2.0)
            stale_keys = [key for key, ts in self._episode_recall_throttle.items() if ts < cutoff]
            for key in stale_keys[:256]:
                self._episode_recall_throttle.pop(key, None)
        throttled: list[int] = []
        for episode_id in sorted({int(item) for item in (episode_ids or []) if int(item) > 0}):
            last = self._episode_recall_throttle.get(episode_id)
            if last is not None and (now - last) < ttl:
                continue
            self._episode_recall_throttle[episode_id] = now
            throttled.append(episode_id)
        if not throttled:
            return
        # Best-effort, no await in prompt path.
        with contextlib.suppress(Exception):
            asyncio.create_task(self._safe_touch_episode_recalls(throttled))

    @staticmethod
    def _top_items_by_abs_delta(evidence_rows: list[dict[str, object]], *, limit: int = 8) -> list[dict[str, object]]:
        scored: list[tuple[float, dict[str, object]]] = []
        for row in evidence_rows:
            if not isinstance(row, dict):
                continue
            score = abs(float(row.get("delta_sum") or 0.0)) * max(0.1, float(row.get("avg_signal_confidence") or 0.0))
            score += min(0.5, float(row.get("avg_influence_weight") or 0.0) * 0.15)
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[: max(1, int(limit))]]

    def _build_reflection_overlay_summary(
        self,
        *,
        window: dict[str, object],
        top_evidence: list[dict[str, object]],
        episodes: list[dict[str, object]],
    ) -> str:
        msg_count = int(window.get("ingested_count", 0) or 0)
        users = int(window.get("unique_users", 0) or 0)
        parts: list[str] = [f"Recent reflection window: {msg_count} messages across {users} users."]

        if top_evidence:
            dim_labels = {
                "trust": "trust",
                "warmth": "warmth",
                "familiarity": "familiarity",
                "banter_license": "banter comfort",
                "support_sensitivity": "support sensitivity",
                "boundary_sensitivity": "boundary caution",
                "topic_alignment": "topic alignment",
            }
            snippets: list[str] = []
            seen_users: set[str] = set()
            for row in top_evidence:
                user_id = str(row.get("user_id") or "")
                if user_id in seen_users and len(snippets) >= 2:
                    continue
                seen_users.add(user_id)
                dim = str(row.get("dimension_key") or "")
                dim_label = dim_labels.get(dim, dim)
                delta = float(row.get("delta_sum") or 0.0)
                direction = "up" if delta >= 0 else "down"
                snippets.append(f"{dim_label} {direction} ({abs(delta):.2f})")
                if len(snippets) >= 3:
                    break
            if snippets:
                parts.append("Relationship signals: " + "; ".join(snippets) + ".")

        if episodes:
            top_eps = sorted(
                [row for row in episodes if isinstance(row, dict)],
                key=lambda row: (float(row.get("importance") or 0.0), float(row.get("confidence") or 0.0)),
                reverse=True,
            )[:2]
            if top_eps:
                ep_names = ", ".join(str(row.get("episode_type") or "moment") for row in top_eps)
                parts.append(f"Episodic candidates observed: {ep_names}.")
        return _collapse_spaces(" ".join(parts))[:760]

    def _supports_llm_reflection_proposer(self) -> bool:
        return bool(
            getattr(self.settings, "persona_reflection_llm_proposer_enabled", False)
            and callable(getattr(self.llm, "json_chat", None))
        )

    @staticmethod
    def _trim_for_llm(text: object, limit: int) -> str:
        cleaned = _collapse_spaces(str(text or ""))
        if len(cleaned) <= max(1, int(limit)):
            return cleaned
        trimmed = cleaned[: max(1, int(limit))].rstrip()
        return trimmed + "..."

    def _build_reflection_llm_context(
        self,
        *,
        state: dict[str, object],
        traits: list[dict[str, object]],
        window: dict[str, object],
        baseline: dict[str, object],
    ) -> dict[str, object]:
        evidence_rows = [row for row in (window.get("relationship_evidence") or []) if isinstance(row, dict)]
        messages = [row for row in (window.get("messages") or []) if isinstance(row, dict)]
        episodes = [row for row in (window.get("episodes") or []) if isinstance(row, dict)]
        relationships = [row for row in (window.get("relationships") or []) if isinstance(row, dict)]
        msg_limit = max(4, min(40, int(getattr(self.settings, "persona_reflection_llm_message_sample_limit", 18))))

        recent_messages = []
        for row in messages[-msg_limit:]:
            recent_messages.append(
                {
                    "message_id": int(row.get("message_id") or 0),
                    "user_id": str(row.get("user_id") or ""),
                    "author_label": self._trim_for_llm(row.get("author_label"), 48),
                    "modality": str(row.get("modality") or "text"),
                    "quality": round(float(row.get("quality") or 0.0), 3),
                    "content": self._trim_for_llm(row.get("content"), 220),
                }
            )

        top_evidence = self._top_items_by_abs_delta(evidence_rows, limit=14)
        compact_evidence = [
            {
                "guild_id": str(row.get("guild_id") or ""),
                "user_id": str(row.get("user_id") or ""),
                "dimension_key": str(row.get("dimension_key") or ""),
                "sample_count": int(row.get("sample_count") or 0),
                "delta_sum": round(float(row.get("delta_sum") or 0.0), 4),
                "avg_signal_confidence": round(float(row.get("avg_signal_confidence") or 0.0), 4),
                "avg_quality_weight": round(float(row.get("avg_quality_weight") or 0.0), 4),
                "avg_influence_weight": round(float(row.get("avg_influence_weight") or 0.0), 4),
                "max_message_id": int(row.get("max_message_id") or 0),
            }
            for row in top_evidence
        ]

        compact_traits = []
        for row in traits:
            if not isinstance(row, dict):
                continue
            compact_traits.append(
                {
                    "trait_key": str(row.get("trait_key") or ""),
                    "current_value": round(float(row.get("current_value") or 0.0), 4),
                    "anchor_value": round(float(row.get("anchor_value") or 0.0), 4),
                    "confidence": round(float(row.get("confidence") or 0.0), 4),
                    "status": str(row.get("status") or "emerging"),
                    "protected_mode": str(row.get("protected_mode") or "soft"),
                    "max_abs_drift": round(float(row.get("max_abs_drift") or 0.0), 4),
                    "max_step_per_reflection": round(float(row.get("max_step_per_reflection") or 0.0), 4),
                    "prompt_exposure": str(row.get("prompt_exposure") or "relevant"),
                }
            )

        compact_relationships = []
        for row in sorted(
            relationships,
            key=lambda item: (
                float(item.get("confidence") or 0.0) + min(1.0, float(item.get("interaction_count") or 0) / 30.0)
            ),
            reverse=True,
        )[:10]:
            compact_relationships.append(
                {
                    "guild_id": str(row.get("guild_id") or ""),
                    "user_id": str(row.get("user_id") or ""),
                    "user_label_cache": self._trim_for_llm(row.get("user_label_cache"), 60),
                    "status": str(row.get("status") or "active"),
                    "consent_scope": str(row.get("consent_scope") or "full"),
                    "familiarity": round(float(row.get("familiarity") or 0.0), 4),
                    "trust": round(float(row.get("trust") or 0.0), 4),
                    "warmth": round(float(row.get("warmth") or 0.0), 4),
                    "banter_license": round(float(row.get("banter_license") or 0.0), 4),
                    "support_sensitivity": round(float(row.get("support_sensitivity") or 0.0), 4),
                    "boundary_sensitivity": round(float(row.get("boundary_sensitivity") or 0.0), 4),
                    "topic_alignment": round(float(row.get("topic_alignment") or 0.0), 4),
                    "confidence": round(float(row.get("confidence") or 0.0), 4),
                    "interaction_count": int(row.get("interaction_count") or 0),
                    "effective_influence_weight": round(float(row.get("effective_influence_weight") or 0.0), 4),
                }
            )

        compact_episodes = []
        for row in sorted(
            episodes,
            key=lambda item: (float(item.get("importance") or 0.0), float(item.get("confidence") or 0.0)),
            reverse=True,
        )[:12]:
            compact_episodes.append(
                {
                    "episode_id": int(row.get("episode_id") or 0),
                    "episode_type": str(row.get("episode_type") or "moment"),
                    "status": str(row.get("status") or "candidate"),
                    "privacy_level": str(row.get("privacy_level") or "participants_only"),
                    "callback_safety": str(row.get("callback_safety") or "safe"),
                    "importance": round(float(row.get("importance") or 0.0), 4),
                    "confidence": round(float(row.get("confidence") or 0.0), 4),
                    "source_message_count": int(row.get("source_message_count") or 0),
                    "participant_count": int(row.get("participant_count") or 0),
                    "title": self._trim_for_llm(row.get("title"), 120),
                    "summary": self._trim_for_llm(row.get("summary"), 220),
                }
            )

        baseline_compact = {
            "overlay_summary_candidate": self._trim_for_llm(baseline.get("overlay_summary_candidate"), 420),
            "trait_drift_candidates": [
                {
                    "trait_key": str(item.get("trait_key") or ""),
                    "delta": round(float(item.get("delta") or 0.0), 4),
                    "confidence": round(float(item.get("confidence") or 0.0), 4),
                    "reason": self._trim_for_llm(item.get("reason"), 140),
                }
                for item in (baseline.get("trait_drift_candidates") or [])[:10]
                if isinstance(item, dict)
            ],
            "episode_promotion_candidates": [
                {
                    "episode_id": int(item.get("episode_id") or 0),
                    "confidence": round(float(item.get("confidence") or 0.0), 4),
                    "reason": self._trim_for_llm(item.get("reason"), 140),
                }
                for item in (baseline.get("episode_promotion_candidates") or [])[:10]
                if isinstance(item, dict)
            ],
        }

        return {
            "persona_id": self.persona_id,
            "policy_version": int(getattr(self.settings, "persona_policy_version", 1)),
            "core_guardrails": {
                "core_dna_immutable": True,
                "safety_immutable": True,
                "reflection_mode": "dry_run_only",
            },
            "current_state": {
                "overlay_summary": self._trim_for_llm(state.get("overlay_summary"), 420),
                "last_reflection_at": str(state.get("last_reflection_at") or ""),
                "reflection_cursor_message_id": int(state.get("reflection_cursor_message_id", 0) or 0),
            },
            "window": {
                "after_message_id": int(window.get("after_message_id", 0) or 0),
                "max_message_id": int(window.get("max_message_id", 0) or 0),
                "ingested_count": int(window.get("ingested_count", 0) or 0),
                "unique_users": int(window.get("unique_users", 0) or 0),
                "voice_messages": int(window.get("voice_messages", 0) or 0),
                "text_messages": int(window.get("text_messages", 0) or 0),
            },
            "traits": compact_traits,
            "recent_messages": recent_messages,
            "relationship_evidence_top": compact_evidence,
            "relationships_snapshot": compact_relationships,
            "episode_candidates": compact_episodes,
            "deterministic_baseline_summary": baseline_compact,
        }

    def _merge_llm_reflection_proposal(
        self,
        *,
        baseline: dict[str, object],
        payload: dict[str, object],
    ) -> dict[str, object]:
        merged = copy.deepcopy(baseline)
        merged["proposal_type"] = "persona_reflection_dry_run_v1"
        merged["dry_run"] = True
        merged["window"] = copy.deepcopy(baseline.get("window") or {})

        overlay_summary = payload.get("overlay_summary_candidate")
        if isinstance(overlay_summary, str) and overlay_summary.strip():
            merged["overlay_summary_candidate"] = _collapse_spaces(overlay_summary)[:760]

        trait_candidates = payload.get("trait_drift_candidates")
        if isinstance(trait_candidates, list):
            merged["trait_drift_candidates"] = [item for item in trait_candidates[:16] if isinstance(item, dict)]

        episode_candidates = payload.get("episode_promotion_candidates")
        if isinstance(episode_candidates, list):
            merged["episode_promotion_candidates"] = [item for item in episode_candidates[:12] if isinstance(item, dict)]

        merged_obs = copy.deepcopy(baseline.get("observations") or {})
        raw_obs = payload.get("observations")
        if isinstance(raw_obs, dict):
            rel_notes = raw_obs.get("relationship_note_candidates")
            if isinstance(rel_notes, list):
                merged_obs["relationship_note_candidates"] = [item for item in rel_notes[:6] if isinstance(item, dict)]
        if not isinstance(merged_obs, dict):
            merged_obs = {}
        merged["observations"] = merged_obs
        return merged

    async def _try_build_reflection_proposal_llm(
        self,
        *,
        state: dict[str, object],
        traits: list[dict[str, object]],
        window: dict[str, object],
        baseline: dict[str, object],
    ) -> tuple[dict[str, object] | None, dict[str, object]]:
        if not self._supports_llm_reflection_proposer():
            return None, {
                "enabled": False,
                "source": "deterministic",
                "reason": "llm_reflection_proposer_disabled_or_unavailable",
                "model_name": "deterministic-reflection",
                "prompt_version": "persona_reflection_dry_run_v1",
            }

        context_packet = self._build_reflection_llm_context(
            state=state,
            traits=traits,
            window=window,
            baseline=baseline,
        )
        messages = [
            {"role": "system", "content": PERSONA_REFLECTION_LLM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_persona_reflection_llm_user_prompt(context_packet, baseline),
            },
        ]
        started = time.monotonic()
        model_name = str(
            getattr(self.llm, "model", "") or getattr(self.settings, "gemini_model", "") or "llm"
        )
        prompt_version = "persona_reflection_llm_v1"
        try:
            payload = await self.llm.json_chat(  # type: ignore[union-attr]
                messages,
                schema_hint=PERSONA_REFLECTION_LLM_SCHEMA_HINT,
                temperature=float(getattr(self.settings, "persona_reflection_llm_temperature", 0.12)),
                max_output_tokens=int(getattr(self.settings, "persona_reflection_llm_max_output_tokens", 1200)),
            )
            duration_ms = int(max(0.0, (time.monotonic() - started)) * 1000.0)
            if payload is None:
                return None, {
                    "enabled": True,
                    "source": "deterministic",
                    "reason": "llm_json_parse_failed",
                    "model_name": model_name,
                    "prompt_version": prompt_version,
                    "llm_duration_ms": duration_ms,
                }
            proposal = self._merge_llm_reflection_proposal(baseline=baseline, payload=payload)
            return proposal, {
                "enabled": True,
                "source": "llm",
                "reason": "",
                "model_name": model_name,
                "prompt_version": prompt_version,
                "llm_duration_ms": duration_ms,
            }
        except Exception as exc:
            logger.warning("Persona reflection LLM proposer failed: %s", exc)
            return None, {
                "enabled": True,
                "source": "deterministic",
                "reason": f"llm_error:{str(exc)[:160]}",
                "model_name": model_name,
                "prompt_version": prompt_version,
            }

    def _build_reflection_proposal(
        self,
        *,
        state: dict[str, object],
        traits: list[dict[str, object]],
        window: dict[str, object],
    ) -> dict[str, object]:
        evidence_rows = [row for row in (window.get("relationship_evidence") or []) if isinstance(row, dict)]
        episodes = [row for row in (window.get("episodes") or []) if isinstance(row, dict)]
        messages = [row for row in (window.get("messages") or []) if isinstance(row, dict)]
        top_evidence = self._top_items_by_abs_delta(evidence_rows, limit=10)
        unique_users = int(window.get("unique_users", 0) or 0)
        ingested_count = int(window.get("ingested_count", 0) or 0)

        trait_map = {
            str(row.get("trait_key") or ""): row
            for row in traits
            if isinstance(row, dict) and str(row.get("trait_key") or "")
        }
        evidence_by_dim: dict[str, dict[str, float]] = {}
        for row in evidence_rows:
            dim = str(row.get("dimension_key") or "")
            if not dim:
                continue
            slot = evidence_by_dim.setdefault(dim, {"delta_sum": 0.0, "samples": 0.0, "conf_sum": 0.0})
            slot["delta_sum"] += float(row.get("delta_sum") or 0.0)
            slot["samples"] += float(row.get("sample_count") or 0.0)
            slot["conf_sum"] += float(row.get("avg_signal_confidence") or 0.0)

        episode_type_counts: dict[str, int] = {}
        for row in episodes:
            kind = str(row.get("episode_type") or "moment")
            episode_type_counts[kind] = episode_type_counts.get(kind, 0) + 1

        candidate_deltas: list[dict[str, object]] = []
        def _candidate(trait_key: str, delta: float, confidence: float, reason: str, evidence_ref: dict[str, object]) -> None:
            if abs(delta) < 0.0009:
                return
            candidate_deltas.append(
                {
                    "trait_key": trait_key,
                    "delta": float(delta),
                    "confidence": _clamp(float(confidence), 0.0, 1.0),
                    "reason": reason[:180],
                    "evidence": evidence_ref,
                }
            )

        support = evidence_by_dim.get("support_sensitivity")
        if support and support["samples"] >= 2:
            _candidate(
                "empathy_expression",
                _clamp(support["delta_sum"] * 0.25, -0.03, 0.03),
                0.45 + min(0.35, support["samples"] * 0.04),
                "support_sensitivity evidence accumulated in relationship updates",
                {"dimension_key": "support_sensitivity", "samples": int(support["samples"])},
            )
        banter = evidence_by_dim.get("banter_license")
        if banter and banter["samples"] >= 2:
            _candidate(
                "banter_intensity",
                _clamp(banter["delta_sum"] * 0.22, -0.03, 0.03),
                0.44 + min(0.30, banter["samples"] * 0.035),
                "banter comfort trend detected across recent interactions",
                {"dimension_key": "banter_license", "samples": int(banter["samples"])},
            )
        if episode_type_counts.get("joke", 0) >= 2:
            _candidate(
                "inside_joke_recall",
                min(0.04, 0.01 + (episode_type_counts.get("joke", 0) * 0.007)),
                0.46 + min(0.28, episode_type_counts.get("joke", 0) * 0.04),
                "multiple joke/meme episodic candidates suggest stronger callback opportunities",
                {"episode_type": "joke", "count": int(episode_type_counts.get("joke", 0))},
            )
        if episode_type_counts.get("support", 0) >= 1 and support:
            _candidate(
                "sarcasm_sharpness",
                -min(0.02, 0.006 + episode_type_counts.get("support", 0) * 0.004),
                0.40 + min(0.18, episode_type_counts.get("support", 0) * 0.05),
                "support-oriented interactions suggest softer edge in recent window",
                {"episode_type": "support", "count": int(episode_type_counts.get("support", 0))},
            )

        promoted_episodes: list[dict[str, object]] = []
        for row in episodes:
            if str(row.get("status") or "") != "candidate":
                continue
            if int(row.get("source_message_count") or 0) < 2:
                continue
            confidence = float(row.get("confidence") or 0.0)
            importance = float(row.get("importance") or 0.0)
            if confidence < 0.55 and importance < 0.60:
                continue
            promoted_episodes.append(
                {
                    "episode_id": int(row.get("episode_id") or 0),
                    "from_status": "candidate",
                    "to_status": "confirmed",
                    "confidence": _clamp(max(confidence, importance), 0.0, 1.0),
                    "reason": "multiple supporting message evidences in reflection window",
                }
            )
        promoted_episodes = sorted(promoted_episodes, key=lambda item: float(item.get("confidence") or 0.0), reverse=True)[:6]

        # Relationship note proposals (read-only for Phase 5 dry-run) from strongest user+dim signals.
        rel_notes: list[dict[str, object]] = []
        rel_seen_users: set[str] = set()
        for row in top_evidence:
            user_id = str(row.get("user_id") or "")
            if not user_id or user_id in rel_seen_users:
                continue
            rel_seen_users.add(user_id)
            dim = str(row.get("dimension_key") or "")
            delta = float(row.get("delta_sum") or 0.0)
            rel_notes.append(
                {
                    "guild_id": str(row.get("guild_id") or ""),
                    "user_id": user_id,
                    "summary_hint": f"{dim} {'increased' if delta >= 0 else 'decreased'} in recent interactions",
                    "confidence": _clamp(float(row.get("avg_signal_confidence") or 0.0), 0.0, 1.0),
                    "evidence_samples": int(row.get("sample_count") or 0),
                }
            )
            if len(rel_notes) >= 4:
                break

        proposal = {
            "proposal_type": "persona_reflection_dry_run_v1",
            "dry_run": True,
            "window": {
                "after_message_id": int(window.get("after_message_id", 0) or 0),
                "max_message_id": int(window.get("max_message_id", 0) or 0),
                "ingested_count": ingested_count,
                "unique_users": unique_users,
                "voice_messages": int(window.get("voice_messages", 0) or 0),
                "text_messages": int(window.get("text_messages", 0) or 0),
                "message_samples_included": len(messages),
                "relationship_evidence_groups": len(evidence_rows),
                "episode_candidates_seen": len(episodes),
            },
            "overlay_summary_candidate": self._build_reflection_overlay_summary(
                window=window,
                top_evidence=top_evidence,
                episodes=episodes,
            ),
            "observations": {
                "top_relationship_evidence": top_evidence[:8],
                "episode_type_counts": episode_type_counts,
                "relationship_note_candidates": rel_notes,
            },
            "trait_drift_candidates": candidate_deltas,
            "episode_promotion_candidates": promoted_episodes,
        }
        return proposal

    def _validate_reflection_proposal(
        self,
        *,
        state: dict[str, object],
        traits: list[dict[str, object]],
        window: dict[str, object],
        proposal: dict[str, object],
    ) -> dict[str, object]:
        errors: list[str] = []
        warnings: list[str] = []
        accepted_traits: list[dict[str, object]] = []
        contested_traits: list[dict[str, object]] = []
        rejected_traits: list[dict[str, object]] = []

        if str(proposal.get("proposal_type") or "") != "persona_reflection_dry_run_v1":
            errors.append("invalid_proposal_type")
        window_obj = proposal.get("window")
        if not isinstance(window_obj, dict):
            errors.append("missing_window_object")
            window_obj = {}

        trait_candidates = proposal.get("trait_drift_candidates")
        if not isinstance(trait_candidates, list):
            errors.append("trait_drift_candidates_not_list")
            trait_candidates = []
        trait_map = {
            str(row.get("trait_key") or ""): row
            for row in traits
            if isinstance(row, dict) and str(row.get("trait_key") or "")
        }
        unique_users = int(window.get("unique_users", 0) or 0)
        ingested_count = int(window.get("ingested_count", 0) or 0)
        diversity_gate_ok = unique_users >= 2 or ingested_count >= max(
            int(getattr(self.settings, "persona_reflection_min_new_messages", 30)),
            20,
        )
        if not diversity_gate_ok and trait_candidates:
            warnings.append("global_trait_drift_diversity_gate_not_met")

        for raw in trait_candidates:
            if not isinstance(raw, dict):
                rejected_traits.append({"reason": "candidate_not_object"})
                continue
            trait_key = str(raw.get("trait_key") or "").strip()
            row = trait_map.get(trait_key)
            if row is None:
                rejected_traits.append({"trait_key": trait_key, "reason": "unknown_trait"})
                continue
            protected_mode = str(row.get("protected_mode") or "soft")
            if protected_mode == "locked":
                rejected_traits.append({"trait_key": trait_key, "reason": "locked_trait"})
                continue
            try:
                delta = float(raw.get("delta", 0.0))
                confidence = float(raw.get("confidence", 0.0))
            except (TypeError, ValueError):
                rejected_traits.append({"trait_key": trait_key, "reason": "invalid_numeric_values"})
                continue
            if not diversity_gate_ok:
                rejected_traits.append({"trait_key": trait_key, "reason": "diversity_gate_not_met"})
                continue
            max_step = abs(float(row.get("max_step_per_reflection", 0.03) or 0.03))
            anchor = float(row.get("anchor_value", 0.5) or 0.5)
            current = float(row.get("current_value", anchor) or anchor)
            support_score = float(row.get("support_score", 0.0) or 0.0)
            contradiction_score = float(row.get("contradiction_score", 0.0) or 0.0)
            current_drift = current - anchor
            established = (
                float(row.get("confidence", 0.0) or 0.0) >= 0.55
                or int(row.get("evidence_count", 0) or 0) >= 2
                or support_score >= 0.05
            )
            current_status = str(row.get("status") or "emerging").strip().lower() or "emerging"
            opposes_drift = (
                abs(current_drift) >= 0.025
                and abs(delta) >= min(0.02, max_step)
                and ((current_drift > 0 and delta < 0) or (current_drift < 0 and delta > 0))
            )
            contradiction_ratio = (
                (contradiction_score / support_score)
                if support_score > 1e-9
                else (999.0 if contradiction_score > 0.0 else 0.0)
            )
            contradiction_pressure = (
                contradiction_score >= 0.04
                and contradiction_ratio >= 0.75
                and abs(delta) >= min(max_step, 0.015)
                and confidence >= 0.32
            )
            contested_guard = (
                current_status == "contested"
                and abs(delta) >= min(max_step, 0.01)
                and confidence >= 0.25
            )
            conflict_kind = ""
            if established and opposes_drift and confidence >= 0.30:
                conflict_kind = "direction_flip"
            elif contested_guard:
                conflict_kind = "contested_trait_guard"
            elif contradiction_pressure:
                conflict_kind = "contradiction_pressure"
            if conflict_kind:
                contested_traits.append(
                    {
                        "trait_key": trait_key,
                        "proposed_delta": _clamp(delta, -max_step, max_step),
                        "confidence": _clamp(confidence, 0.0, 1.0),
                        "reason": str(raw.get("reason") or "")[:180],
                        "conflict_kind": conflict_kind,
                        "current_drift": round(current_drift, 6),
                        "support_score": round(support_score, 6),
                        "contradiction_score": round(contradiction_score, 6),
                        "contradiction_ratio": round(contradiction_ratio, 6),
                        "status_before": current_status,
                        "evidence": raw.get("evidence") if isinstance(raw.get("evidence"), (dict, list)) else {},
                    }
                )
                continue
            if abs(delta) > max_step + 1e-9:
                rejected_traits.append(
                    {"trait_key": trait_key, "reason": "delta_exceeds_max_step", "delta": delta, "max_step": max_step}
                )
                continue
            max_abs_drift = abs(float(row.get("max_abs_drift", 0.2) or 0.2))
            new_drift = (current + delta) - anchor
            if abs(new_drift) > max_abs_drift + 1e-9:
                rejected_traits.append(
                    {
                        "trait_key": trait_key,
                        "reason": "drift_bound_exceeded",
                        "proposed_drift": new_drift,
                        "max_abs_drift": max_abs_drift,
                    }
                )
                continue
            accepted_traits.append(
                {
                    "trait_key": trait_key,
                    "delta": _clamp(delta, -max_step, max_step),
                    "confidence": _clamp(confidence, 0.0, 1.0),
                    "reason": str(raw.get("reason") or "")[:180],
                }
            )

        episode_candidates = proposal.get("episode_promotion_candidates")
        if not isinstance(episode_candidates, list):
            errors.append("episode_promotion_candidates_not_list")
            episode_candidates = []
        accepted_episodes: list[dict[str, object]] = []
        rejected_episodes: list[dict[str, object]] = []
        for raw in episode_candidates:
            if not isinstance(raw, dict):
                rejected_episodes.append({"reason": "candidate_not_object"})
                continue
            try:
                episode_id = int(raw.get("episode_id") or 0)
                confidence = float(raw.get("confidence", 0.0))
            except (TypeError, ValueError):
                rejected_episodes.append({"reason": "invalid_numeric_values"})
                continue
            if episode_id <= 0:
                rejected_episodes.append({"reason": "invalid_episode_id"})
                continue
            if confidence < 0.45:
                rejected_episodes.append({"episode_id": episode_id, "reason": "low_confidence"})
                continue
            accepted_episodes.append(
                {
                    "episode_id": episode_id,
                    "to_status": "confirmed",
                    "confidence": _clamp(confidence, 0.0, 1.0),
                    "reason": str(raw.get("reason") or "")[:180],
                }
            )

        overlay_summary = _collapse_spaces(str(proposal.get("overlay_summary_candidate") or ""))[:760]
        if not overlay_summary:
            warnings.append("empty_overlay_summary_candidate")
        if contested_traits:
            warnings.append("contested_trait_candidates_present")

        valid = not errors
        return {
            "validator_version": "persona_reflection_validator_v1",
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
            "accepted_trait_candidates": accepted_traits,
            "accepted_contested_trait_candidates": contested_traits,
            "rejected_trait_candidates": rejected_traits,
            "accepted_episode_promotions": accepted_episodes,
            "rejected_episode_promotions": rejected_episodes,
            "diversity_gate_ok": diversity_gate_ok,
            "window_stats": {
                "ingested_count": ingested_count,
                "unique_users": unique_users,
                "max_message_id": int(window.get("max_message_id", 0) or 0),
            },
            "overlay_summary_candidate": overlay_summary,
        }

    def _build_reflection_apply_plan(
        self,
        *,
        proposal: dict[str, object],
        validator_report: dict[str, object],
    ) -> dict[str, object]:
        apply_enabled = bool(getattr(self.settings, "persona_reflection_apply_enabled", False))
        trait_drift_enabled = bool(getattr(self.settings, "persona_trait_drift_enabled", False))
        episodic_enabled = bool(getattr(self.settings, "persona_episodic_enabled", False))

        accepted_traits = [
            row
            for row in (validator_report.get("accepted_trait_candidates") or [])
            if isinstance(row, dict)
        ]
        contested_traits = [
            row
            for row in (validator_report.get("accepted_contested_trait_candidates") or [])
            if isinstance(row, dict)
        ]
        accepted_episodes = [
            row
            for row in (validator_report.get("accepted_episode_promotions") or [])
            if isinstance(row, dict)
        ]
        overlay_summary = _collapse_spaces(str(validator_report.get("overlay_summary_candidate") or ""))[:760]

        warnings: list[str] = []
        if not apply_enabled:
            return {
                "apply_enabled": False,
                "trait_drift_enabled": trait_drift_enabled,
                "episodic_enabled": episodic_enabled,
                "traits_to_apply": [],
                "contested_traits_to_apply": [],
                "episodes_to_apply": [],
                "overlay_summary_to_apply": "",
                "warnings": ["reflection_apply_disabled"],
                "has_any_changes": False,
            }
        traits_to_apply = accepted_traits if trait_drift_enabled else []
        contested_traits_to_apply = contested_traits
        episodes_to_apply = accepted_episodes if episodic_enabled else []
        if accepted_traits and not trait_drift_enabled:
            warnings.append("trait_drift_apply_disabled")
        if accepted_episodes and not episodic_enabled:
            warnings.append("episodic_apply_disabled")
        if contested_traits and not traits_to_apply and not episodes_to_apply and not overlay_summary:
            warnings.append("contested_updates_only")
        return {
            "apply_enabled": True,
            "trait_drift_enabled": trait_drift_enabled,
            "episodic_enabled": episodic_enabled,
            "traits_to_apply": traits_to_apply,
            "contested_traits_to_apply": contested_traits_to_apply,
            "episodes_to_apply": episodes_to_apply,
            "overlay_summary_to_apply": overlay_summary,
            "warnings": warnings,
            "has_any_changes": bool(traits_to_apply or contested_traits_to_apply or episodes_to_apply or overlay_summary),
            "accepted_trait_candidates_count": len(accepted_traits),
            "accepted_contested_trait_candidates_count": len(contested_traits),
            "accepted_episode_promotions_count": len(accepted_episodes),
        }

    async def _apply_validated_reflection(
        self,
        *,
        reflection_id: int,
        proposal: dict[str, object],
        validator_report: dict[str, object],
        reflection_status: str,
        window_end_message_id: int,
        model_name: str,
        prompt_version: str,
        trigger_type: str,
        trigger_reason: str,
        actor_user_id: str | None = None,
        proposer_source: str = "",
        proposer_reason: str = "",
        base_duration_ms: int | None = None,
        extra_warnings: list[str] | None = None,
    ) -> dict[str, object]:
        if not self.enabled:
            result = {"status": "skipped", "reason": "persona_disabled"}
            self._last_reflection_result = result
            return result
        if not self._supports_reflection_apply_backend():
            result = {"status": "skipped", "reason": "reflection_apply_backend_unsupported"}
            self._last_reflection_result = result
            return result

        status_clean = str(reflection_status or "").strip().lower()
        if status_clean == "applied":
            result = {"status": "skipped", "reason": "reflection_already_applied", "reflection_id": int(reflection_id)}
            self._last_reflection_result = result
            return result
        if status_clean not in {"dry_run", "proposed"}:
            result = {
                "status": "skipped",
                "reason": "reflection_status_not_applicable",
                "reflection_id": int(reflection_id),
                "reflection_status": status_clean,
            }
            self._last_reflection_result = result
            return result
        if not bool(validator_report.get("valid")):
            result = {
                "status": "skipped",
                "reason": "validator_report_not_valid",
                "reflection_id": int(reflection_id),
            }
            self._last_reflection_result = result
            return result

        apply_plan = self._build_reflection_apply_plan(proposal=proposal, validator_report=validator_report)
        if not bool(apply_plan.get("apply_enabled")):
            result = {
                "status": "skipped",
                "reason": "reflection_apply_disabled",
                "reflection_id": int(reflection_id),
                "warnings": list(apply_plan.get("warnings") or []),
            }
            self._last_reflection_result = result
            return result
        if not bool(apply_plan.get("has_any_changes")):
            result = {
                "status": "skipped",
                "reason": "no_validated_changes_to_apply",
                "reflection_id": int(reflection_id),
                "warnings": [*list(apply_plan.get("warnings") or []), *(extra_warnings or [])],
            }
            self._last_reflection_result = result
            return result

        started_apply = time.monotonic()
        applied_changes = await self.memory.apply_persona_reflection_changes(
            self.persona_id,
            int(reflection_id),
            accepted_trait_candidates=list(apply_plan.get("traits_to_apply") or []),
            contested_trait_candidates=list(apply_plan.get("contested_traits_to_apply") or []),
            accepted_episode_promotions=list(apply_plan.get("episodes_to_apply") or []),
            overlay_summary=str(apply_plan.get("overlay_summary_to_apply") or ""),
        )
        if not bool(applied_changes.get("ok")):
            reason = str(applied_changes.get("error") or "apply_failed")
            if reason in {"reflection_already_applied", "reflection_not_found"}:
                result = {
                    "status": "skipped",
                    "reason": reason,
                    "reflection_id": int(reflection_id),
                    "warnings": [*list(apply_plan.get("warnings") or []), *(extra_warnings or [])],
                }
                self._last_reflection_result = result
                return result
            await self.memory.update_persona_reflection_status(
                int(reflection_id),
                status="failed",
                rejection_reason=reason[:220],
                applied_changes_json=applied_changes,
                model_name=model_name,
                prompt_version=prompt_version,
            )
            result = {
                "status": "failed",
                "reason": reason,
                "reflection_id": int(reflection_id),
                "warnings": [*list(apply_plan.get("warnings") or []), *(extra_warnings or [])],
            }
            self._last_reflection_result = result
            return result

        await self.memory.update_persona_reflection_status(
            int(reflection_id),
            status="applied",
            proposal_json=proposal,
            validator_report_json=validator_report,
            applied_changes_json=applied_changes,
            duration_ms=(base_duration_ms or 0) + int(max(0.0, (time.monotonic() - started_apply)) * 1000.0),
            model_name=model_name,
            prompt_version=prompt_version,
        )
        await self.memory.update_persona_global_reflection_checkpoint(
            self.persona_id,
            reflection_cursor_message_id=max(0, int(window_end_message_id)),
            overlay_summary=str(apply_plan.get("overlay_summary_to_apply") or "") or None,
        )
        if callable(getattr(self.memory, "append_persona_audit_log", None)):
            try:
                await self.memory.append_persona_audit_log(
                    self.persona_id,
                    actor_type="admin" if actor_user_id else "system",
                    actor_user_id=actor_user_id,
                    action="reflection_applied",
                    entity_type="persona_reflections",
                    entity_id=str(reflection_id),
                    after_json={
                        "status": "applied",
                        "applied_trait_updates": len(applied_changes.get("applied_trait_updates") or []),
                        "applied_contested_trait_updates": len(applied_changes.get("applied_contested_trait_updates") or []),
                        "applied_episode_promotions": len(applied_changes.get("applied_episode_promotions") or []),
                        "proposer_source": proposer_source,
                        "proposer_model": model_name,
                    },
                    evidence_refs=[{"kind": "reflection_id", "value": int(reflection_id)}],
                    reason=f"trigger={trigger_type}; {trigger_reason}"[:300],
                )
            except Exception:
                pass
        self.invalidate_prompt_cache()
        result = {
            "status": "applied",
            "reflection_id": int(reflection_id),
            "window": proposal.get("window"),
            "proposer_source": proposer_source,
            "proposer_model": model_name,
            "proposer_prompt_version": prompt_version,
            "proposer_reason": proposer_reason,
            "accepted_trait_candidates": len(validator_report.get("accepted_trait_candidates") or []),
            "accepted_contested_trait_candidates": len(validator_report.get("accepted_contested_trait_candidates") or []),
            "accepted_episode_promotions": len(validator_report.get("accepted_episode_promotions") or []),
            "applied_trait_updates": len(applied_changes.get("applied_trait_updates") or []),
            "applied_contested_trait_updates": len(applied_changes.get("applied_contested_trait_updates") or []),
            "applied_episode_promotions_count": len(applied_changes.get("applied_episode_promotions") or []),
            "overlay_summary_candidate": str(validator_report.get("overlay_summary_candidate") or ""),
            "overlay_summary_applied": bool(applied_changes.get("overlay_summary_applied")),
            "warnings": [*list(validator_report.get("warnings") or []), *list(apply_plan.get("warnings") or []), *(extra_warnings or [])],
            "duration_ms": (base_duration_ms or 0) + int(max(0.0, (time.monotonic() - started_apply)) * 1000.0),
            "applied_changes": applied_changes,
        }
        self._last_reflection_result = result
        return result

    async def apply_reflection(
        self,
        *,
        reflection_id: int | None = None,
        actor_user_id: str | None = None,
    ) -> dict[str, object]:
        if not self.enabled:
            result = {"status": "skipped", "reason": "persona_disabled"}
            self._last_reflection_result = result
            return result
        if not self._supports_reflection_apply_backend():
            result = {"status": "skipped", "reason": "reflection_apply_backend_unsupported"}
            self._last_reflection_result = result
            return result

        selected_reflection_id = int(reflection_id) if reflection_id is not None else None
        if selected_reflection_id is None and callable(getattr(self.memory, "list_persona_reflections", None)):
            try:
                recent = await self.memory.list_persona_reflections(self.persona_id, limit=8)
            except Exception:
                recent = []
            for row in recent:
                if not isinstance(row, dict):
                    continue
                status = str(row.get("status") or "").strip().lower()
                if status in {"dry_run", "proposed"}:
                    try:
                        selected_reflection_id = int(row.get("reflection_id") or 0)
                    except (TypeError, ValueError):
                        selected_reflection_id = None
                    if selected_reflection_id:
                        break

        details = await self.memory.get_persona_reflection_details(
            self.persona_id,
            reflection_id=selected_reflection_id,
        )
        if not isinstance(details, dict):
            result = {
                "status": "skipped",
                "reason": "reflection_not_found",
                "reflection_id": selected_reflection_id,
            }
            self._last_reflection_result = result
            return result

        proposal = details.get("proposal_json") if isinstance(details.get("proposal_json"), dict) else {}
        validator_report = details.get("validator_report_json") if isinstance(details.get("validator_report_json"), dict) else {}
        model_name = str(details.get("model_name") or "deterministic-reflection")
        proposer_source = "deterministic" if model_name == "deterministic-reflection" else "llm"
        return await self._apply_validated_reflection(
            reflection_id=int(details.get("reflection_id") or 0),
            proposal=proposal if isinstance(proposal, dict) else {},
            validator_report=validator_report if isinstance(validator_report, dict) else {},
            reflection_status=str(details.get("status") or ""),
            window_end_message_id=int(details.get("window_end_message_id") or 0),
            model_name=model_name,
            prompt_version=str(details.get("prompt_version") or ""),
            trigger_type=str(details.get("trigger_type") or "manual"),
            trigger_reason=str(details.get("trigger_reason") or "manual_apply"),
            actor_user_id=actor_user_id,
            proposer_source=proposer_source,
            proposer_reason="",
            base_duration_ms=int(details.get("duration_ms") or 0) if details.get("duration_ms") is not None else None,
            extra_warnings=[],
        )

    async def admin_trait_why_snapshot(self, trait_key: str, *, limit: int = 8) -> dict[str, object]:
        if not self.enabled:
            return {"enabled": False, "status": self.status_snapshot()}
        trait_key_clean = str(trait_key or "").strip()
        traits = await self.memory.list_persona_traits(self.persona_id, limit=64)
        trait_row = None
        if trait_key_clean:
            for row in traits:
                if str(row.get("trait_key") or "") == trait_key_clean:
                    trait_row = row
                    break
            if trait_row is None:
                key_cf = trait_key_clean.casefold()
                for row in traits:
                    row_key = str(row.get("trait_key") or "")
                    if row_key.casefold() == key_cf or row_key.casefold().startswith(key_cf):
                        trait_row = row
                        break
        evidence_rows: list[dict[str, object]] = []
        evidence_fn = getattr(self.memory, "list_persona_trait_evidence", None)
        if trait_row is not None and callable(evidence_fn):
            evidence_rows = await evidence_fn(
                self.persona_id,
                str(trait_row.get("trait_key") or ""),
                limit=max(1, min(int(limit), 12)),
            )
        retrieval_policy = self._trait_retrieval_policy(
            trait_row if isinstance(trait_row, dict) else None,
            retrieval_enabled=self.retrieval_enabled,
        )
        return {
            "enabled": True,
            "status": self.status_snapshot(),
            "trait_key_requested": trait_key_clean,
            "trait": trait_row,
            "evidence": evidence_rows,
            "retrieval_policy": retrieval_policy,
            "available_trait_keys": [str(row.get("trait_key") or "") for row in traits[:24]],
        }

    async def run_reflection_once(
        self,
        *,
        trigger_type: str = "scheduled",
        trigger_reason: str = "",
        force: bool = False,
        dry_run: bool = True,
    ) -> dict[str, object]:
        started = time.monotonic()
        if not self.enabled:
            result = {"status": "skipped", "reason": "persona_disabled"}
            self._last_reflection_result = result
            return result
        if not self._supports_reflection_backend():
            result = {"status": "skipped", "reason": "reflection_backend_unsupported"}
            self._last_reflection_result = result
            return result

        state = await self.memory.get_persona_global_state(self.persona_id)
        if not isinstance(state, dict):
            result = {"status": "skipped", "reason": "persona_state_missing"}
            self._last_reflection_result = result
            return result

        last_reflection_at = self._parse_dt(state.get("last_reflection_at"))
        now_dt = datetime.now(timezone.utc)
        min_interval_minutes = max(1, int(getattr(self.settings, "persona_reflection_min_interval_minutes", 60)))
        if not force and last_reflection_at is not None:
            elapsed = (now_dt - last_reflection_at).total_seconds() / 60.0
            if elapsed < float(min_interval_minutes):
                result = {
                    "status": "skipped",
                    "reason": "min_interval_not_reached",
                    "minutes_remaining": max(0.0, round(float(min_interval_minutes) - elapsed, 2)),
                }
                self._last_reflection_result = result
                return result

        cursor = int(state.get("reflection_cursor_message_id", 0) or 0)
        window = await self.memory.get_persona_reflection_window(
            self.persona_id,
            after_message_id=cursor,
        )
        ingested_count = int(window.get("ingested_count", 0) or 0)
        max_message_id = int(window.get("max_message_id", 0) or 0)
        min_new_messages = max(1, int(getattr(self.settings, "persona_reflection_min_new_messages", 30)))
        if ingested_count <= 0 or max_message_id <= cursor:
            result = {"status": "skipped", "reason": "no_new_ingested_messages", "cursor": cursor}
            self._last_reflection_result = result
            return result
        if not force and ingested_count < min_new_messages:
            result = {
                "status": "skipped",
                "reason": "not_enough_new_messages",
                "new_messages": ingested_count,
                "required": min_new_messages,
                "cursor": cursor,
                "max_message_id": max_message_id,
            }
            self._last_reflection_result = result
            return result

        traits = await self.memory.list_persona_traits(self.persona_id, limit=32)
        dedupe_key = f"{self.persona_id}:{trigger_type}:{cursor}:{max_message_id}:{int(bool(dry_run))}"
        reflection_id = await self.memory.create_persona_reflection(
            self.persona_id,
            trigger_type=trigger_type,
            trigger_reason=trigger_reason,
            status="queued",
            dedupe_key=dedupe_key,
            window_start_message_id=cursor + 1 if max_message_id > cursor else cursor,
            window_end_message_id=max_message_id,
            input_counts={
                "ingested_count": ingested_count,
                "unique_users": int(window.get("unique_users", 0) or 0),
                "message_samples": len(window.get("messages") or []),
                "relationship_evidence_groups": len(window.get("relationship_evidence") or []),
                "episodes": len(window.get("episodes") or []),
            },
            model_name="reflection-pending",
            prompt_version="persona_reflection_v2",
        )
        if reflection_id <= 0:
            result = {"status": "skipped", "reason": "reflection_create_failed", "dedupe_key": dedupe_key}
            self._last_reflection_result = result
            return result

        await self.memory.mark_persona_reflection_started(reflection_id)

        baseline_proposal = self._build_reflection_proposal(state=state, traits=traits, window=window)
        proposal = baseline_proposal
        proposer_meta: dict[str, object] = {
            "enabled": False,
            "source": "deterministic",
            "reason": "",
            "model_name": "deterministic-reflection",
            "prompt_version": "persona_reflection_dry_run_v1",
        }
        proposer_warnings: list[str] = []

        llm_proposal, llm_meta = await self._try_build_reflection_proposal_llm(
            state=state,
            traits=traits,
            window=window,
            baseline=baseline_proposal,
        )
        if isinstance(llm_meta, dict):
            proposer_meta = llm_meta
        if isinstance(llm_proposal, dict):
            proposal = llm_proposal
        elif bool(llm_meta.get("enabled")) and str(llm_meta.get("reason") or "").strip():
            proposer_warnings.append(str(llm_meta.get("reason"))[:180])

        validator_report = self._validate_reflection_proposal(
            state=state,
            traits=traits,
            window=window,
            proposal=proposal,
        )
        if not bool(validator_report.get("valid")) and proposal is not baseline_proposal:
            baseline_validator_report = self._validate_reflection_proposal(
                state=state,
                traits=traits,
                window=window,
                proposal=baseline_proposal,
            )
            if bool(baseline_validator_report.get("valid")):
                proposer_warnings.append("llm_proposal_validator_rejected_fallback")
                proposal = baseline_proposal
                validator_report = baseline_validator_report
                proposer_meta = {
                    "enabled": bool(llm_meta.get("enabled")),
                    "source": "deterministic",
                    "reason": "llm_validator_rejected_fallback",
                    "model_name": "deterministic-reflection",
                    "prompt_version": "persona_reflection_dry_run_v1",
                }

        duration_ms = int(max(0.0, (time.monotonic() - started)) * 1000.0)
        effective_model_name = str(proposer_meta.get("model_name") or "deterministic-reflection")
        effective_prompt_version = str(proposer_meta.get("prompt_version") or "persona_reflection_dry_run_v1")
        proposer_source = str(proposer_meta.get("source") or "deterministic")
        proposer_reason = str(proposer_meta.get("reason") or "")

        if not bool(validator_report.get("valid")):
            reason_text = ",".join(str(item) for item in (validator_report.get("errors") or []))[:220]
            if proposer_reason:
                proposer_warnings.append(proposer_reason[:180])
            await self.memory.update_persona_reflection_status(
                reflection_id,
                status="rejected",
                rejection_reason=reason_text or "validator_rejected",
                proposal_json=proposal,
                validator_report_json=validator_report,
                duration_ms=duration_ms,
                model_name=effective_model_name,
                prompt_version=effective_prompt_version,
            )
            result = {
                "status": "rejected",
                "reflection_id": reflection_id,
                "reason": reason_text or "validator_rejected",
                "window": proposal.get("window"),
                "proposer_source": proposer_source,
                "proposer_model": effective_model_name,
                "proposer_prompt_version": effective_prompt_version,
                "proposer_reason": proposer_reason,
                "warnings": proposer_warnings,
            }
            self._last_reflection_result = result
            return result

        if not dry_run and bool(getattr(self.settings, "persona_reflection_apply_enabled", False)):
            apply_result = await self._apply_validated_reflection(
                reflection_id=int(reflection_id),
                proposal=proposal,
                validator_report=validator_report,
                reflection_status="proposed",
                window_end_message_id=max_message_id,
                model_name=effective_model_name,
                prompt_version=effective_prompt_version,
                trigger_type=trigger_type,
                trigger_reason=trigger_reason,
                actor_user_id=None,
                proposer_source=proposer_source,
                proposer_reason=proposer_reason,
                base_duration_ms=duration_ms,
                extra_warnings=proposer_warnings,
            )
            if str(apply_result.get("status") or "") == "applied":
                return apply_result
            proposer_warnings.extend(
                str(item)[:160]
                for item in (apply_result.get("warnings") or [])
                if str(item or "").strip()
            )
            skip_reason = str(apply_result.get("reason") or "").strip()
            if skip_reason:
                proposer_warnings.append(f"apply_skipped:{skip_reason[:140]}")

        final_status = "dry_run" if dry_run else "proposed"
        await self.memory.update_persona_reflection_status(
            reflection_id,
            status=final_status,
            proposal_json=proposal,
            validator_report_json=validator_report,
            duration_ms=duration_ms,
            model_name=effective_model_name,
            prompt_version=effective_prompt_version,
        )
        await self.memory.update_persona_global_reflection_checkpoint(
            self.persona_id,
            reflection_cursor_message_id=max_message_id,
            overlay_summary=None,
        )
        if callable(getattr(self.memory, "append_persona_audit_log", None)):
            try:
                await self.memory.append_persona_audit_log(
                    self.persona_id,
                    actor_type="system",
                    actor_user_id=None,
                    action="reflection_dry_run" if dry_run else "reflection_proposed",
                    entity_type="persona_reflections",
                    entity_id=str(reflection_id),
                    after_json={
                        "status": final_status,
                        "window_end_message_id": max_message_id,
                        "proposer_source": proposer_source,
                        "proposer_model": effective_model_name,
                        "accepted_trait_candidates": len(validator_report.get("accepted_trait_candidates") or []),
                        "accepted_contested_trait_candidates": len(
                            validator_report.get("accepted_contested_trait_candidates") or []
                        ),
                        "accepted_episode_promotions": len(validator_report.get("accepted_episode_promotions") or []),
                    },
                    reason=f"trigger={trigger_type}; {trigger_reason}"[:300],
                )
            except Exception:
                pass
        self.invalidate_prompt_cache()
        result = {
            "status": final_status,
            "reflection_id": reflection_id,
            "window": proposal.get("window"),
            "proposer_source": proposer_source,
            "proposer_model": effective_model_name,
            "proposer_prompt_version": effective_prompt_version,
            "proposer_reason": proposer_reason,
            "accepted_trait_candidates": len(validator_report.get("accepted_trait_candidates") or []),
            "accepted_contested_trait_candidates": len(
                validator_report.get("accepted_contested_trait_candidates") or []
            ),
            "accepted_episode_promotions": len(validator_report.get("accepted_episode_promotions") or []),
            "applied_trait_updates": 0,
            "applied_contested_trait_updates": 0,
            "applied_episode_promotions_count": 0,
            "overlay_summary_candidate": str(validator_report.get("overlay_summary_candidate") or ""),
            "warnings": [*list(validator_report.get("warnings") or []), *proposer_warnings],
            "duration_ms": duration_ms,
        }
        self._last_reflection_result = result
        return result

    async def run_scheduled_reflection(self) -> dict[str, object]:
        if not bool(getattr(self.settings, "persona_reflection_enabled", False)):
            result = {"status": "skipped", "reason": "reflection_disabled"}
            self._last_reflection_result = result
            return result
        return await self.run_reflection_once(
            trigger_type="scheduled",
            trigger_reason="background_loop",
            force=False,
            dry_run=True,
        )

    async def run_decay_once(
        self,
        *,
        trigger_type: str = "scheduled",
        trigger_reason: str = "",
        force: bool = False,
        actor_user_id: str | None = None,
    ) -> dict[str, object]:
        started = time.monotonic()
        if not self.enabled:
            result = {"status": "skipped", "reason": "persona_disabled"}
            self._last_decay_result = result
            return result
        if not self._supports_decay_backend():
            result = {"status": "skipped", "reason": "decay_backend_unsupported"}
            self._last_decay_result = result
            return result
        if not force and not bool(getattr(self.settings, "persona_decay_enabled", False)):
            result = {"status": "skipped", "reason": "decay_disabled"}
            self._last_decay_result = result
            return result

        decay_fn = getattr(self.memory, "run_persona_decay_cycle", None)
        if not callable(decay_fn):
            result = {"status": "skipped", "reason": "decay_fn_missing"}
            self._last_decay_result = result
            return result

        cycle = await decay_fn(
            self.persona_id,
            min_interval_minutes=(0 if force else int(getattr(self.settings, "persona_decay_min_interval_minutes", 180))),
        )
        if not isinstance(cycle, dict):
            result = {"status": "failed", "reason": "invalid_decay_result"}
            self._last_decay_result = result
            return result
        if not bool(cycle.get("ok")):
            result = {
                "status": "failed",
                "reason": str(cycle.get("error") or "decay_failed"),
            }
            self._last_decay_result = result
            return result
        if bool(cycle.get("skipped")):
            result = {
                "status": "skipped",
                "reason": str(cycle.get("reason") or "skipped"),
                "minutes_remaining": cycle.get("minutes_remaining"),
            }
            self._last_decay_result = result
            return result

        self.invalidate_prompt_cache()
        duration_ms = int(max(0.0, (time.monotonic() - started)) * 1000.0)
        result = {
            "status": "applied",
            "trigger_type": trigger_type,
            "trigger_reason": trigger_reason,
            "trait_updates": int(cycle.get("trait_updates") or 0),
            "relationship_updates": int(cycle.get("relationship_updates") or 0),
            "episode_updates": int(cycle.get("episode_updates") or 0),
            "episode_archived": int(cycle.get("episode_archived") or 0),
            "trait_examples": list(cycle.get("trait_examples") or []),
            "relationship_examples": list(cycle.get("relationship_examples") or []),
            "episode_examples": list(cycle.get("episode_examples") or []),
            "duration_ms": duration_ms,
        }
        if callable(getattr(self.memory, "append_persona_audit_log", None)):
            try:
                await self.memory.append_persona_audit_log(
                    self.persona_id,
                    actor_type="admin" if actor_user_id else "system",
                    actor_user_id=actor_user_id,
                    action="persona_decay_cycle",
                    entity_type="persona_global_state",
                    entity_id=self.persona_id,
                    after_json={
                        "trait_updates": result["trait_updates"],
                        "relationship_updates": result["relationship_updates"],
                        "episode_updates": result["episode_updates"],
                        "episode_archived": result["episode_archived"],
                    },
                    reason=f"trigger={trigger_type}; {trigger_reason}"[:300],
                )
            except Exception:
                pass
        self._last_decay_result = result
        return result

    async def run_scheduled_decay(self) -> dict[str, object]:
        return await self.run_decay_once(
            trigger_type="scheduled",
            trigger_reason="background_loop",
            force=False,
            actor_user_id=None,
        )

    async def build_prompt_overlay(
        self,
        *,
        mode: str,
        guild_id: str,
        channel_id: str,
        user_id: str,
        query_text: str = "",
    ) -> str:
        if not self.retrieval_enabled:
            return ""

        pref: dict[str, object] | None = None
        allow_callbacks = True
        pref_fn = getattr(self.memory, "get_persona_user_memory_pref", None)
        if callable(pref_fn) and user_id not in {"", "*"}:
            pref = await pref_fn(guild_id, user_id)
            memory_mode = str((pref or {}).get("memory_mode") or "full").strip().lower()
            if memory_mode == "none":
                return ""
            allow_callbacks = bool((pref or {}).get("allow_episodic_callbacks", True))

        key = self._cache_key(mode, guild_id, channel_id, user_id, self._query_fingerprint(query_text))
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached is not None and (now - cached[0]) <= self._cache_ttl():
            return cached[1]

        state_fn = getattr(self.memory, "get_persona_global_state", None)
        traits_fn = getattr(self.memory, "list_persona_traits", None)
        rel_fn = getattr(self.memory, "get_persona_relationship", None)
        episodes_fn = getattr(self.memory, "list_persona_episode_callbacks", None)
        if not (callable(state_fn) and callable(traits_fn) and callable(rel_fn)):
            return ""

        state = await state_fn(self.persona_id)
        trait_limit = 4 if mode == "voice" else 6
        fetch_limit = max(trait_limit + 6, trait_limit * 2)
        traits = await traits_fn(self.persona_id, limit=fetch_limit)
        relationship = None
        if user_id not in {"", "*"}:
            relationship = await rel_fn(self.persona_id, guild_id, user_id)

        lines: list[str] = [
            "PERSONA GROWTH OVERLAY (lower priority than RP CANON and safety):",
        ]
        overlay_summary = str((state or {}).get("overlay_summary") or "").strip()
        if overlay_summary:
            lines.append(f"- current_growth_summary: {overlay_summary}")
        else:
            lines.append("- current_growth_summary: early-stage growth profile (core-led)")

        if traits:
            lines.append("Stable growth traits (bounded, evidence-based):")
            suppressed_contested: list[str] = []
            added_trait_lines = 0
            for row in traits:
                if added_trait_lines >= trait_limit:
                    break
                if str(row.get("prompt_exposure") or "relevant") == "never":
                    continue
                if str(row.get("status") or "").strip().lower() == "contested":
                    suppressed_contested.append(str(row.get("trait_key") or "?"))
                    continue
                lines.append(self._fmt_trait_line(row))
                added_trait_lines += 1
            if suppressed_contested:
                lines.append(
                    "- contested traits withheld pending reconfirmation."
                    if added_trait_lines <= 0
                    else f"- contested traits withheld: {', '.join(suppressed_contested[:3])}"
                )

        lines.extend(self._fmt_relationship_line(relationship))
        if (
            bool(getattr(self.settings, "persona_episodic_enabled", False))
            and allow_callbacks
            and callable(episodes_fn)
        ):
            ep_top_k = int(
                getattr(
                    self.settings,
                    "persona_episode_voice_top_k" if mode == "voice" else "persona_episode_text_top_k",
                    1 if mode == "voice" else 2,
                )
            )
            if ep_top_k > 0:
                episodes = await episodes_fn(
                    self.persona_id,
                    guild_id,
                    channel_id,
                    user_id,
                    limit=ep_top_k,
                    query_text=query_text,
                )
                if isinstance(episodes, list) and episodes:
                    recalled_episode_ids: list[int] = []
                    lines.append("Shared episodic callbacks (use only if natural):")
                    for row in episodes:
                        if not isinstance(row, dict):
                            continue
                        line = str(row.get("callback_line") or "").strip()
                        if not line:
                            continue
                        with contextlib.suppress(Exception):
                            recalled_episode_ids.append(int(row.get("episode_id") or 0))
                        conf = float(row.get("confidence") or 0.0)
                        kind = str(row.get("episode_type") or "moment")
                        lines.append(f"- [{kind} | c={conf:.2f}] {line}")
                    self._schedule_episode_recall_reconfirm(recalled_episode_ids)

        budget_chars = int(
            getattr(
                self.settings,
                "persona_voice_prompt_budget_chars" if mode == "voice" else "persona_text_prompt_budget_chars",
                260 if mode == "voice" else 700,
            )
        )
        text = self._trim_to_budget("\n".join(line for line in lines if line), max(60, budget_chars))
        self._cache[key] = (now, text)
        return text

    async def admin_persona_snapshot(self) -> dict[str, object]:
        if not self.enabled:
            return {"enabled": False, "status": self.status_snapshot()}
        state = await self.memory.get_persona_global_state(self.persona_id)
        traits = await self.memory.list_persona_traits(self.persona_id, limit=12)
        reflections = await self.memory.list_persona_reflections(self.persona_id, limit=5)
        return {
            "enabled": True,
            "status": self.status_snapshot(),
            "state": state,
            "traits": traits,
            "reflections": reflections,
        }

    async def admin_relationship_snapshot(self, guild_id: str, user_id: str) -> dict[str, object]:
        if not self.enabled:
            return {"enabled": False, "status": self.status_snapshot()}
        pref = await self.memory.get_persona_user_memory_pref(guild_id, user_id)
        relationship = await self.memory.get_persona_relationship(self.persona_id, guild_id, user_id)
        return {
            "enabled": True,
            "status": self.status_snapshot(),
            "memory_pref": pref,
            "relationship": relationship,
        }

    async def admin_episodes_snapshot(
        self,
        guild_id: str,
        *,
        user_id: str | None = None,
        channel_id: str | None = None,
        limit: int = 10,
    ) -> dict[str, object]:
        if not self.enabled:
            return {"enabled": False, "status": self.status_snapshot()}
        episodes = await self.memory.list_persona_episodes(
            self.persona_id,
            guild_id,
            user_id=user_id,
            channel_id=channel_id,
            limit=max(1, min(int(limit), 20)),
        )
        return {
            "enabled": True,
            "status": self.status_snapshot(),
            "episodes": episodes,
            "filter_user_id": user_id,
            "filter_channel_id": channel_id,
        }
