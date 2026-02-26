from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..common import PendingProfileUpdate, PendingSummaryUpdate, as_int, collapse_spaces, truncate
from ...prompts.memory import (
    build_biography_summary_update_system_prompt,
    build_biography_summary_update_user_prompt,
    build_summary_update_system_prompt,
    build_summary_update_user_prompt,
)
from ...memory.fact_moderation import CandidateModerationInput, FactModerationPolicyV2
from ...memory.storage.utils import (
    normalize_memory_fact_about_target,
    normalize_memory_fact_directness,
    sanitize_memory_fact_evidence_quote,
)

logger = logging.getLogger("live_role_bot")


class WorkersMixin:
    def _memory_extractor_dry_run_enabled(self) -> bool:
        return bool(getattr(self.settings, "memory_extractor_dry_run_enabled", False))

    def _memory_extractor_audit_enabled(self) -> bool:
        return bool(getattr(self.settings, "memory_extractor_audit_enabled", True))

    def _fact_moderation_policy(self) -> FactModerationPolicyV2:
        cached = getattr(self, "_memory_fact_moderation_policy_v2", None)
        if isinstance(cached, FactModerationPolicyV2):
            return cached
        policy = FactModerationPolicyV2.from_settings(self.settings)
        setattr(self, "_memory_fact_moderation_policy_v2", policy)
        return policy

    async def _record_memory_extractor_audit(
        self,
        *,
        item: PendingProfileUpdate,
        fact_owner_kind: str,
        fact_owner_id: str,
        backend_name: str,
        model_name: str,
        dry_run: bool,
        diagnostics: object | None,
        candidates: list[dict[str, object]],
        accepted_count: int,
        saved_count: int,
        filtered_count: int,
        fallback_error: str = "",
    ) -> None:
        if not self._memory_extractor_audit_enabled():
            return
        recorder = getattr(self.memory, "record_memory_extractor_run", None)
        if not callable(recorder):
            return

        def _as_bool(name: str, default: bool = False) -> bool:
            try:
                return bool(getattr(diagnostics, name)) if diagnostics is not None else default
            except Exception:
                return default

        def _as_int(name: str, default: int = 0) -> int:
            try:
                return int(getattr(diagnostics, name)) if diagnostics is not None else default
            except Exception:
                return default

        def _as_str(name: str, default: str = "") -> str:
            try:
                raw = getattr(diagnostics, name) if diagnostics is not None else default
            except Exception:
                raw = default
            return str(raw or default)

        try:
            await recorder(
                guild_id=item.guild_id,
                channel_id=item.channel_id,
                speaker_user_id=item.user_id,
                fact_owner_kind=str(fact_owner_kind or "user"),
                fact_owner_id=str(fact_owner_id or item.user_id),
                speaker_role=str(getattr(item, "speaker_role", "user") or "user"),
                modality=str(getattr(item, "modality", "text") or "text"),
                source=str(getattr(item, "source", "unknown") or "unknown"),
                backend_name=backend_name,
                model_name=model_name,
                dry_run=bool(dry_run),
                llm_attempted=_as_bool("llm_attempted", diagnostics is not None),
                llm_ok=_as_bool("llm_ok", False),
                json_valid=_as_bool("json_valid", False),
                fallback_used=_as_bool("fallback_used", False),
                latency_ms=_as_int("latency_ms", 0),
                candidate_count=len(candidates),
                accepted_count=max(0, int(accepted_count)),
                saved_count=max(0, int(saved_count)),
                filtered_count=max(0, int(filtered_count)),
                error_text=(fallback_error or _as_str("error", ""))[:400],
                candidates=candidates,
            )
        except Exception:
            logger.exception(
                "Memory extractor audit log write failed for guild=%s channel=%s user=%s message_id=%s",
                item.guild_id,
                item.channel_id,
                item.user_id,
                item.message_id,
            )

    def _persona_memory_subject_user_id(self) -> str:
        persona_id = str(getattr(self.settings, "persona_id", "") or "persona").strip() or "persona"
        return f"persona::{persona_id}"

    def _biography_refresh_state(self) -> dict[str, float]:
        state = getattr(self, "_biography_refresh_monotonic_by_subject", None)
        if isinstance(state, dict):
            return state
        state = {}
        setattr(self, "_biography_refresh_monotonic_by_subject", state)
        return state

    def _biography_refresh_allowed(self, subject_key: str) -> bool:
        try:
            min_interval = int(getattr(self.settings, "memory_biography_summary_refresh_min_interval_seconds", 90))
        except (TypeError, ValueError):
            min_interval = 90
        if min_interval <= 0:
            return True
        now = time.monotonic()
        state = self._biography_refresh_state()
        last_ts = float(state.get(subject_key, 0.0) or 0.0)
        if (now - last_ts) < float(min_interval):
            return False
        state[subject_key] = now
        return True

    async def _refresh_global_biography_summary_from_facts(
        self,
        *,
        subject_kind: str,
        subject_id: str,
        facts_user_id: str,
        source_guild_id: str | None = None,
    ) -> None:
        if not bool(getattr(self.settings, "memory_biography_summary_enabled", True)):
            return
        kind = str(subject_kind or "").strip().lower()
        subject = str(subject_id or "").strip()
        facts_owner = str(facts_user_id or "").strip()
        if not (kind and subject and facts_owner):
            return
        subject_key = f"{kind}:{subject}"
        if not self._biography_refresh_allowed(subject_key):
            return

        get_facts = getattr(self.memory, "get_user_facts_global_by_user_id", None)
        get_bio = getattr(self.memory, "get_global_biography_summary", None)
        upsert_bio = getattr(self.memory, "upsert_global_biography_summary", None)
        if not (callable(get_facts) and callable(upsert_bio)):
            return

        facts = await get_facts(facts_owner, limit=18)
        if not facts:
            return

        previous = await get_bio(kind, subject) if callable(get_bio) else None
        previous_summary = str((previous or {}).get("summary_text") or "").strip()
        previous_fact_count = as_int((previous or {}).get("source_fact_count"), 0)
        previous_source_updated = str((previous or {}).get("source_updated_at") or "").strip()

        latest_fact_updated = ""
        for fact in facts:
            ts = str(fact.get("updated_at") or "").strip()
            if ts and ts > latest_fact_updated:
                latest_fact_updated = ts

        # If nothing materially changed and the previous biography is present, skip regeneration.
        if (
            previous_summary
            and previous_fact_count >= len(facts)
            and previous_source_updated
            and latest_fact_updated
            and previous_source_updated >= latest_fact_updated
        ):
            return

        max_chars = max(180, as_int(getattr(self.settings, "memory_biography_summary_max_chars", 520), 520))
        fact_lines: list[str] = []
        for fact in facts[:14]:
            value = str(fact.get("fact_value") or "").strip()
            if not value:
                continue
            fact_type = str(fact.get("fact_type") or "fact").strip().lower() or "fact"
            conf = float(fact.get("confidence") or 0.0)
            status = str(fact.get("status") or "candidate").strip().lower() or "candidate"
            directness = str(fact.get("directness") or "explicit").strip().lower() or "explicit"
            evidence_count = as_int(fact.get("evidence_count"), 0)
            if status == "candidate" and directness in {"implicit", "inferred"}:
                if conf < 0.80 or evidence_count < 2:
                    continue
            source_guild = str(fact.get("guild_id") or "").strip()
            source_suffix = f" | guild={source_guild}" if source_guild else ""
            directness_suffix = f" | d={directness}" if directness and directness != "explicit" else ""
            fact_lines.append(f"- [{fact_type} | c={conf:.2f}{directness_suffix}{source_suffix}] {value}")
        if not fact_lines:
            return

        prompt_messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": build_biography_summary_update_system_prompt(
                    max_chars=max_chars,
                    subject_kind="persona" if kind == "persona" else "user",
                ),
            },
            {
                "role": "user",
                "content": build_biography_summary_update_user_prompt(
                    subject_kind="persona" if kind == "persona" else "user",
                    previous_summary=previous_summary,
                    fact_lines=fact_lines,
                ),
            },
        ]
        summary = collapse_spaces(await self.llm.chat(prompt_messages, temperature=0.12, max_output_tokens=420))
        if not summary:
            return
        if len(summary) > max_chars:
            summary = truncate(summary, max_chars)

        await upsert_bio(
            subject_kind=kind,
            subject_id=subject,
            summary_text=summary,
            source_fact_count=len(facts),
            source_summary_count=0,
            source_updated_at=latest_fact_updated or None,
            last_source_guild_id=(source_guild_id or ""),
        )
        logger.info("[memory.bio] subject=%s facts=%s chars=%s", subject_key, len(facts), len(summary))

    def _persona_queue_diag_state(self) -> dict[str, Any]:
        state = getattr(self, "_persona_queue_diag", None)
        if isinstance(state, dict):
            return state
        state = {
            "started_at_monotonic": time.monotonic(),
            "counters": {},
            "last_event": {},
            "recent": [],
        }
        setattr(self, "_persona_queue_diag", state)
        return state

    def _record_persona_queue_diag(
        self,
        *,
        stage: str,
        outcome: str,
        reason: str = "",
        **fields: object,
    ) -> None:
        state = self._persona_queue_diag_state()
        counters = state.setdefault("counters", {})
        if not isinstance(counters, dict):
            counters = {}
            state["counters"] = counters
        stage_key = str(stage or "unknown").strip() or "unknown"
        outcome_key = str(outcome or "unknown").strip() or "unknown"
        reason_key = str(reason or "none").strip() or "none"
        key = f"{stage_key}.{outcome_key}.{reason_key}"
        try:
            counters[key] = int(counters.get(key, 0) or 0) + 1
            counters["__total__"] = int(counters.get("__total__", 0) or 0) + 1
        except (TypeError, ValueError):
            counters[key] = 1
            counters["__total__"] = int(counters.get("__total__", 0) or 0) + 1
        compact_fields: dict[str, object] = {}
        for k, v in fields.items():
            if v is None:
                continue
            compact_fields[str(k)] = str(v)[:120]
        event = {
            "ts_unix": round(time.time(), 3),
            "stage": stage_key,
            "outcome": outcome_key,
            "reason": reason_key,
            "fields": compact_fields,
        }
        state["last_event"] = event
        recent = state.setdefault("recent", [])
        if isinstance(recent, list):
            recent.append(event)
            if len(recent) > 12:
                del recent[:-12]
        log_line = (
            f"[persona.queue] stage={stage_key} outcome={outcome_key} reason={reason_key}"
            + "".join(f" {k}={v}" for k, v in compact_fields.items())
        )
        if outcome_key in {"error", "exception"}:
            logger.warning(log_line)
        else:
            logger.debug(log_line)

    def _persona_queue_diag_snapshot(self) -> dict[str, Any]:
        state = self._persona_queue_diag_state()
        counters = state.get("counters", {}) if isinstance(state.get("counters"), dict) else {}
        started = float(state.get("started_at_monotonic", 0.0) or 0.0)
        uptime_sec = max(0.0, time.monotonic() - started) if started > 0 else 0.0

        def _sum_prefix(prefix: str) -> int:
            total = 0
            for key, value in counters.items():
                if not isinstance(key, str) or not key.startswith(prefix):
                    continue
                try:
                    total += int(value or 0)
                except (TypeError, ValueError):
                    continue
            return total

        return {
            "uptime_sec": round(uptime_sec, 1),
            "total_events": int(counters.get("__total__", 0) or 0),
            "queued": _sum_prefix("persona_queue.queued."),
            "drops": _sum_prefix("persona_queue.drop."),
            "errors": _sum_prefix("persona_queue.error."),
            "worker_applied": _sum_prefix("persona_worker.ingest.applied"),
            "worker_deduped": _sum_prefix("persona_worker.ingest.deduped"),
            "worker_skipped": _sum_prefix("persona_worker.ingest.skipped"),
            "fallback_profile_worker": _sum_prefix("persona_worker.fallback.profile_worker"),
            "last_event": dict(state.get("last_event", {})) if isinstance(state.get("last_event"), dict) else {},
        }

    def _persona_queue_isolation_enabled(self) -> bool:
        if not bool(getattr(self.settings, "persona_queue_isolation_enabled", False)):
            return False
        persona_engine = getattr(self, "persona_engine", None)
        return bool(persona_engine is not None and bool(getattr(persona_engine, "enabled", False)))

    def _enqueue_persona_event(self, item: PendingProfileUpdate) -> bool:
        if not self._persona_queue_isolation_enabled():
            return False
        queue = getattr(self, "persona_event_queue", None)
        if queue is None:
            self._record_persona_queue_diag(stage="persona_queue", outcome="error", reason="queue_missing")
            return False
        is_voice = str(item.modality or "").strip().lower() == "voice"

        if queue.full():
            self._record_persona_queue_diag(
                stage="persona_queue",
                outcome="drop",
                reason="queue_full_drop_oldest",
                guild_id=item.guild_id,
                channel_id=item.channel_id,
                user_id=item.user_id,
                source=item.source,
            )
            if is_voice and callable(getattr(self, "_record_voice_memory_diag", None)):
                self._record_voice_memory_diag(
                    stage="persona_queue",
                    outcome="drop",
                    reason="queue_full_drop_oldest",
                    guild_id=item.guild_id,
                    channel_id=item.channel_id,
                    user_id=item.user_id,
                    source=item.source,
                )
            try:
                dropped_item = queue.get_nowait()
                if isinstance(dropped_item, PendingProfileUpdate):
                    # The same object is also in profile_queue; clear the marker so profile_worker can fallback-ingest.
                    dropped_item.persona_ingest_enqueued = False
                queue.task_done()
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(item)
            self._record_persona_queue_diag(
                stage="persona_queue",
                outcome="queued",
                reason="ok",
                guild_id=item.guild_id,
                channel_id=item.channel_id,
                user_id=item.user_id,
                source=item.source,
                modality=item.modality,
                message_id=item.message_id,
            )
            if is_voice and callable(getattr(self, "_record_voice_memory_diag", None)):
                self._record_voice_memory_diag(
                    stage="persona_queue",
                    outcome="queued",
                    reason="ok",
                    guild_id=item.guild_id,
                    channel_id=item.channel_id,
                    user_id=item.user_id,
                    source=item.source,
                    message_id=item.message_id,
                )
            return True
        except asyncio.QueueFull:
            self._record_persona_queue_diag(
                stage="persona_queue",
                outcome="drop",
                reason="queue_full_put_failed",
                guild_id=item.guild_id,
                channel_id=item.channel_id,
                user_id=item.user_id,
                source=item.source,
            )
            if is_voice and callable(getattr(self, "_record_voice_memory_diag", None)):
                self._record_voice_memory_diag(
                    stage="persona_queue",
                    outcome="drop",
                    reason="queue_full_put_failed",
                    guild_id=item.guild_id,
                    channel_id=item.channel_id,
                    user_id=item.user_id,
                    source=item.source,
                )
            return False

    async def _persona_ingest_worker(self) -> None:
        while True:
            item = await self.persona_event_queue.get()
            try:
                persona_engine = getattr(self, "persona_engine", None)
                if persona_engine is None or not bool(getattr(persona_engine, "enabled", False)):
                    self._record_persona_queue_diag(
                        stage="persona_worker",
                        outcome="ingest",
                        reason="skipped_persona_disabled",
                        guild_id=item.guild_id,
                        channel_id=item.channel_id,
                        user_id=item.user_id,
                        message_id=item.message_id,
                    )
                    continue
                result = await persona_engine.ingest_user_message(
                    guild_id=item.guild_id,
                    channel_id=item.channel_id,
                    user_id=item.user_id,
                    message_id=item.message_id,
                    user_text=item.user_text,
                    user_label=item.user_label,
                    modality=item.modality,
                    source=item.source,
                    quality=float(item.quality),
                )
                if bool(result.get("deduped")):
                    reason = "deduped"
                elif bool(result.get("applied")):
                    reason = "applied"
                else:
                    skip_reason = str(result.get("reason") or "skipped")
                    reason = f"skipped:{skip_reason[:80]}"
                self._record_persona_queue_diag(
                    stage="persona_worker",
                    outcome="ingest",
                    reason=reason,
                    guild_id=item.guild_id,
                    channel_id=item.channel_id,
                    user_id=item.user_id,
                    message_id=item.message_id,
                    modality=item.modality,
                    source=item.source,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(
                    "Persona ingest worker error for guild=%s channel=%s user=%s message_id=%s",
                    getattr(item, "guild_id", ""),
                    getattr(item, "channel_id", ""),
                    getattr(item, "user_id", ""),
                    getattr(item, "message_id", 0),
                )
                self._record_persona_queue_diag(
                    stage="persona_worker",
                    outcome="error",
                    reason="ingest_failed",
                    guild_id=getattr(item, "guild_id", ""),
                    channel_id=getattr(item, "channel_id", ""),
                    user_id=getattr(item, "user_id", ""),
                    message_id=getattr(item, "message_id", 0),
                    error=exc,
                )
                if str(getattr(item, "modality", "") or "").strip().lower() == "voice" and callable(
                    getattr(self, "_record_voice_memory_diag", None)
                ):
                    self._record_voice_memory_diag(
                        stage="persona_queue",
                        outcome="error",
                        reason="ingest_failed",
                        guild_id=getattr(item, "guild_id", ""),
                        channel_id=getattr(item, "channel_id", ""),
                        user_id=getattr(item, "user_id", ""),
                        source=getattr(item, "source", ""),
                        message_id=getattr(item, "message_id", 0),
                        error=exc,
                    )
            finally:
                self.persona_event_queue.task_done()

    async def _persona_decay_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(47.0)
                persona_engine = getattr(self, "persona_engine", None)
                if persona_engine is None:
                    continue
                result = await persona_engine.run_scheduled_decay()
                status = str(result.get("status") or "")
                if status == "applied":
                    logger.info(
                        "[persona.decay] traits=%s rel=%s episodes=%s archived=%s",
                        result.get("trait_updates"),
                        result.get("relationship_updates"),
                        result.get("episode_updates"),
                        result.get("episode_archived"),
                    )
                elif status in {"failed"}:
                    logger.warning("[persona.decay] status=%s reason=%s", status, result.get("reason"))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Persona decay worker error")

    async def _persona_reflection_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(22.0)
                persona_engine = getattr(self, "persona_engine", None)
                if persona_engine is None:
                    continue
                result = await persona_engine.run_scheduled_reflection()
                status = str(result.get("status") or "")
                if status in {"dry_run", "proposed"}:
                    logger.info(
                        "[persona.reflect] status=%s id=%s traits=%s episodes=%s window=%s",
                        status,
                        result.get("reflection_id"),
                        result.get("accepted_trait_candidates"),
                        result.get("accepted_episode_promotions"),
                        (result.get("window") or {}).get("ingested_count")
                        if isinstance(result.get("window"), dict)
                        else "?",
                    )
                elif status in {"rejected"}:
                    logger.warning(
                        "[persona.reflect] status=%s id=%s reason=%s",
                        status,
                        result.get("reflection_id"),
                        result.get("reason"),
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Persona reflection worker error")

    def _enqueue_profile_update(
        self,
        guild_id: str,
        channel_id: str,
        user_id: str,
        message_id: int,
        user_text: str,
        *,
        user_label: str = "",
        modality: str = "text",
        source: str = "unknown",
        quality: float = 1.0,
        speaker_role: str = "user",
        fact_owner_kind: str = "user",
        fact_owner_id: str = "",
    ) -> None:
        if not self.settings.memory_enabled:
            return
        item = PendingProfileUpdate(
            guild_id=guild_id,
            channel_id=channel_id,
            user_id=user_id,
            message_id=message_id,
            user_text=user_text,
            user_label=user_label,
            modality=modality,
            source=source,
            quality=float(quality),
            speaker_role=str(speaker_role or "user").strip().lower() or "user",
            fact_owner_kind=str(fact_owner_kind or "user").strip().lower() or "user",
            fact_owner_id=str(fact_owner_id or "").strip(),
        )
        if item.speaker_role == "user" and item.fact_owner_kind == "user":
            item.persona_ingest_enqueued = self._enqueue_persona_event(item)
        is_voice = str(modality or "").strip().lower() == "voice"

        if self.profile_queue.full():
            if is_voice and callable(getattr(self, "_record_voice_memory_diag", None)):
                self._record_voice_memory_diag(
                    stage="profile_queue",
                    outcome="drop",
                    reason="queue_full_drop_oldest",
                    guild_id=guild_id,
                    channel_id=channel_id,
                    user_id=user_id,
                    source=source,
                )
            try:
                self.profile_queue.get_nowait()
                self.profile_queue.task_done()
            except asyncio.QueueEmpty:
                pass

        try:
            self.profile_queue.put_nowait(item)
            if is_voice and callable(getattr(self, "_record_voice_memory_diag", None)):
                self._record_voice_memory_diag(
                    stage="profile_queue",
                    outcome="queued",
                    reason="ok",
                    guild_id=guild_id,
                    channel_id=channel_id,
                    user_id=user_id,
                    source=source,
                    quality=f"{float(quality):.3f}",
                )
        except asyncio.QueueFull:
            if is_voice and callable(getattr(self, "_record_voice_memory_diag", None)):
                self._record_voice_memory_diag(
                    stage="profile_queue",
                    outcome="drop",
                    reason="queue_full_put_failed",
                    guild_id=guild_id,
                    channel_id=channel_id,
                    user_id=user_id,
                    source=source,
                )

    def _should_extract_profile_update(self, item: PendingProfileUpdate) -> bool:
        text = collapse_spaces(item.user_text)
        if len(text) < 6:
            return False
        speaker_role = str(getattr(item, "speaker_role", "user")).strip().lower() or "user"
        fact_owner_kind = str(getattr(item, "fact_owner_kind", "user")).strip().lower() or "user"
        if speaker_role == "assistant" and fact_owner_kind == "persona":
            lower_text = f" {text.casefold()} "
            first_person_markers = (
                " я ",
                " мене ",
                " мені ",
                " мій ",
                " моя ",
                " моє ",
                " мої ",
                " мне ",
                " меня ",
                " мой ",
                " моя ",
                " i ",
                " i'm ",
                " ive ",
                " i've ",
                " my ",
                " me ",
            )
            if not any(marker in lower_text for marker in first_person_markers):
                return False
        if speaker_role == "user" and text.startswith(("!", "/", ".", "#")):
            return False

        lower = text.casefold()
        # Skip common backchannel/filler turns that create noisy facts.
        filler = {
            "ага",
            "агаа",
            "ок",
            "окей",
            "окейй",
            "пон",
            "поняв",
            "понявв",
            "ясно",
            "та",
            "угу",
            "мм",
            "мг",
            "yes",
            "no",
            "ok",
            "okay",
            "yeah",
            "nah",
            "lol",
            "gg",
            "bruh",
        }
        if speaker_role == "user" and lower in filler:
            return False

        if item.modality == "voice":
            # Be slightly stricter on STT-driven memory extraction to reduce garbage facts.
            min_conf = max(float(self.settings.transcription_min_confidence), 0.54)
            if float(item.quality) < min_conf:
                return False
            if len(text) < 10 and "?" not in text and "!" not in text:
                return False

        return True

    async def _get_profile_extraction_dialogue_context(
        self,
        item: PendingProfileUpdate,
    ) -> list[dict[str, object]] | None:
        try:
            limit = int(getattr(self.settings, "memory_extractor_dialogue_window_messages", 6))
        except (TypeError, ValueError):
            limit = 6
        if limit <= 0:
            return None
        getter = getattr(self.memory, "get_recent_dialogue_messages", None)
        if not callable(getter):
            return None
        try:
            rows = await getter(item.guild_id, item.channel_id, item.user_id, max(2, limit))
        except Exception:
            logger.exception(
                "Profile extraction dialogue window retrieval failed for guild=%s channel=%s user=%s message_id=%s",
                item.guild_id,
                item.channel_id,
                item.user_id,
                item.message_id,
            )
            return None
        if not isinstance(rows, list):
            return None
        return rows

    async def _profile_worker(self) -> None:
        while True:
            item = await self.profile_queue.get()
            try:
                persona_engine = getattr(self, "persona_engine", None)
                persona_isolation_on = self._persona_queue_isolation_enabled()
                run_persona_fallback = not (persona_isolation_on and bool(getattr(item, "persona_ingest_enqueued", False)))
                if (
                    str(getattr(item, "speaker_role", "user")).strip().lower() == "user"
                    and str(getattr(item, "fact_owner_kind", "user")).strip().lower() == "user"
                    and persona_engine is not None
                    and bool(getattr(persona_engine, "enabled", False))
                    and run_persona_fallback
                ):
                    if persona_isolation_on:
                        self._record_persona_queue_diag(
                            stage="persona_worker",
                            outcome="fallback",
                            reason="profile_worker",
                            guild_id=item.guild_id,
                            channel_id=item.channel_id,
                            user_id=item.user_id,
                            message_id=item.message_id,
                            modality=item.modality,
                            source=item.source,
                        )
                        if str(item.modality or "").strip().lower() == "voice" and callable(
                            getattr(self, "_record_voice_memory_diag", None)
                        ):
                            self._record_voice_memory_diag(
                                stage="persona_queue",
                                outcome="fallback",
                                reason="profile_worker",
                                guild_id=item.guild_id,
                                channel_id=item.channel_id,
                                user_id=item.user_id,
                                source=item.source,
                                message_id=item.message_id,
                            )
                    result = await persona_engine.ingest_user_message(
                        guild_id=item.guild_id,
                        channel_id=item.channel_id,
                        user_id=item.user_id,
                        message_id=item.message_id,
                        user_text=item.user_text,
                        user_label=item.user_label,
                        modality=item.modality,
                        source=item.source,
                        quality=float(item.quality),
                    )
                    if bool(result.get("applied")) and not bool(result.get("deduped")):
                        if result.get("episode_id"):
                            logger.debug(
                                "[persona.ingest] user=%s episode_id=%s created=%s",
                                item.user_id,
                                result.get("episode_id"),
                                result.get("episode_created"),
                            )
                if not self._should_extract_profile_update(item):
                    continue
                owner_kind = str(getattr(item, "fact_owner_kind", "user")).strip().lower() or "user"
                owner_id = str(getattr(item, "fact_owner_id", "") or "").strip()
                speaker_role = str(getattr(item, "speaker_role", "user")).strip().lower() or "user"
                is_persona_self = owner_kind == "persona"
                dialogue_context = await self._get_profile_extraction_dialogue_context(item)
                moderation_policy = self._fact_moderation_policy()
                dry_run = self._memory_extractor_dry_run_enabled()
                extractor_backend_name = str(getattr(self.memory_extractor, "backend_name", "llm") or "llm").strip().lower()
                extractor_backend_name = extractor_backend_name or "llm"
                extractor_model_name = str(getattr(self.memory_extractor, "model_name", "") or "").strip()
                if is_persona_self:
                    if not bool(getattr(self.settings, "memory_persona_self_facts_enabled", True)):
                        continue
                    fact_owner_user_id = self._persona_memory_subject_user_id()
                    biography_subject_kind = "persona"
                    biography_subject_id = owner_id or str(getattr(self.settings, "persona_id", "") or "persona")
                    extractor_name_base = f"{extractor_backend_name}_persona_self_extractor"
                    try:
                        extract_result = await self.memory_extractor.extract_persona_self_facts(
                            assistant_text=item.user_text,
                            preferred_language=self.settings.preferred_response_language,
                            dialogue_context=dialogue_context,
                        )
                    except Exception as exc:
                        logger.exception(
                            "Persona self-fact extraction failed for guild=%s channel=%s persona=%s message_id=%s",
                            item.guild_id,
                            item.channel_id,
                            biography_subject_id,
                            item.message_id,
                        )
                        await self._record_memory_extractor_audit(
                            item=item,
                            fact_owner_kind=owner_kind,
                            fact_owner_id=biography_subject_id,
                            backend_name=extractor_backend_name,
                            model_name=extractor_model_name,
                            dry_run=dry_run,
                            diagnostics=None,
                            candidates=[],
                            accepted_count=0,
                            saved_count=0,
                            filtered_count=0,
                            fallback_error=f"{exc.__class__.__name__}: {exc}",
                        )
                        continue
                else:
                    fact_owner_user_id = item.user_id
                    biography_subject_kind = "user"
                    biography_subject_id = item.user_id
                    extractor_name_base = f"{extractor_backend_name}_profile_extractor"
                    try:
                        extract_result = await self.memory_extractor.extract_user_facts(
                            user_text=item.user_text,
                            preferred_language=self.settings.preferred_response_language,
                            dialogue_context=dialogue_context,
                        )
                    except Exception as exc:
                        logger.exception(
                            "User fact extraction failed for guild=%s channel=%s user=%s message_id=%s",
                            item.guild_id,
                            item.channel_id,
                            item.user_id,
                            item.message_id,
                        )
                        await self._record_memory_extractor_audit(
                            item=item,
                            fact_owner_kind=owner_kind,
                            fact_owner_id=biography_subject_id,
                            backend_name=extractor_backend_name,
                            model_name=extractor_model_name,
                            dry_run=dry_run,
                            diagnostics=None,
                            candidates=[],
                            accepted_count=0,
                            saved_count=0,
                            filtered_count=0,
                            fallback_error=f"{exc.__class__.__name__}: {exc}",
                        )
                        continue

                diagnostics = getattr(extract_result, "diagnostics", None) if extract_result is not None else None
                if diagnostics is not None:
                    extractor_backend_name = str(getattr(diagnostics, "backend_name", extractor_backend_name) or extractor_backend_name)
                    if not extractor_model_name:
                        extractor_model_name = str(getattr(diagnostics, "model_name", "") or "")

                facts = list(getattr(extract_result, "facts", []) or [])
                audit_candidates: list[dict[str, object]] = []
                accepted_for_apply = 0
                saved = 0

                for fact in facts:
                    if not fact.value:
                        audit_candidates.append(
                            {
                                "fact_key": str(getattr(fact, "key", "") or ""),
                                "fact_value": "",
                                "fact_type": str(getattr(fact, "fact_type", "fact") or "fact"),
                                "about_target": str(getattr(fact, "about_target", "unknown") or "unknown"),
                                "directness": str(getattr(fact, "directness", "explicit") or "explicit"),
                                "evidence_quote": str(getattr(fact, "evidence_quote", "") or ""),
                                "confidence": float(getattr(fact, "confidence", 0.0) or 0.0),
                                "importance": float(getattr(fact, "importance", 0.0) or 0.0),
                                "moderation_action": "reject",
                                "moderation_reason": "empty_value",
                                "selected_for_apply": False,
                                "saved_to_memory": False,
                            }
                        )
                        continue
                    about_target = normalize_memory_fact_about_target(
                        str(getattr(fact, "about_target", "") or ""),
                        default="assistant_self" if is_persona_self else "self",
                    )
                    directness = normalize_memory_fact_directness(str(getattr(fact, "directness", "") or ""))
                    evidence_quote = sanitize_memory_fact_evidence_quote(str(getattr(fact, "evidence_quote", "") or ""))
                    moderation_reason = ""
                    selected_for_apply = False
                    saved_to_memory = False

                    if is_persona_self:
                        if about_target == "self":
                            about_target = "assistant_self"
                        if about_target not in {"assistant_self", "unknown"}:
                            moderation_reason = "persona_target_filter"
                        elif float(fact.confidence) < 0.42 and float(fact.importance) < 0.42:
                            moderation_reason = "persona_low_conf_low_importance"
                    else:
                        if about_target == "assistant_self":
                            moderation_reason = "assistant_self_target_filter"
                        elif about_target == "other":
                            moderation_reason = "other_person_target_filter"
                        elif about_target == "unknown":
                            fact.confidence = min(float(fact.confidence), 0.58)
                            if directness == "explicit":
                                directness = "implicit"

                    decision = None
                    if not moderation_reason:
                        decision = moderation_policy.evaluate(
                            CandidateModerationInput(
                                fact_key=str(fact.key),
                                fact_value=str(fact.value),
                                fact_type=str(fact.fact_type),
                                about_target=about_target,
                                directness=directness,
                                confidence=float(fact.confidence),
                                importance=float(fact.importance),
                                evidence_quote=evidence_quote,
                                owner_kind=owner_kind,
                                speaker_role=speaker_role,
                            )
                        )
                        if not decision.accepted:
                            moderation_reason = decision.reason

                    if not moderation_reason:
                        selected_for_apply = True
                        extractor_name = extractor_name_base
                        if item.modality == "voice":
                            extractor_name = f"{extractor_name}:voice"
                        if item.source:
                            extractor_name = f"{extractor_name}:{item.source}"
                        if not dry_run:
                            await self.memory.upsert_user_fact(
                                guild_id=item.guild_id,
                                user_id=fact_owner_user_id,
                                fact_key=fact.key,
                                fact_value=fact.value,
                                fact_type=fact.fact_type,
                                confidence=fact.confidence,
                                importance=fact.importance,
                                message_id=item.message_id,
                                extractor=extractor_name[:120],
                                about_target=about_target,
                                directness=directness,
                                evidence_quote=evidence_quote,
                            )
                            saved_to_memory = True
                            saved += 1
                        accepted_for_apply += 1

                    audit_candidates.append(
                        {
                            "fact_key": str(fact.key),
                            "fact_value": str(fact.value),
                            "fact_type": str(fact.fact_type),
                            "about_target": about_target,
                            "directness": directness,
                            "evidence_quote": evidence_quote,
                            "confidence": float(fact.confidence),
                            "importance": float(fact.importance),
                            "moderation_action": "accept" if selected_for_apply else "reject",
                            "moderation_reason": moderation_reason or (decision.reason if decision is not None else "accepted"),
                            "selected_for_apply": selected_for_apply,
                            "saved_to_memory": saved_to_memory,
                        }
                    )

                await self._record_memory_extractor_audit(
                    item=item,
                    fact_owner_kind=owner_kind,
                    fact_owner_id=biography_subject_id,
                    backend_name=extractor_backend_name,
                    model_name=extractor_model_name,
                    dry_run=dry_run,
                    diagnostics=diagnostics,
                    candidates=audit_candidates,
                    accepted_count=accepted_for_apply,
                    saved_count=saved,
                    filtered_count=max(0, len(audit_candidates) - accepted_for_apply),
                )

                if not facts:
                    continue

                if saved:
                    if is_persona_self:
                        logger.info(
                            "[memory.persona_facts] persona=%s source_guild=%s saved=%s speaker_role=%s",
                            biography_subject_id,
                            item.guild_id,
                            saved,
                            speaker_role,
                        )
                    else:
                        label = await self._resolve_user_label(item.guild_id, item.user_id)
                        logger.info("[memory.facts] user=%s saved=%s", label, saved)
                elif dry_run and accepted_for_apply:
                    logger.info(
                        "[memory.extractor.dry_run] guild=%s channel=%s owner_kind=%s owner_id=%s accepted=%s candidates=%s backend=%s",
                        item.guild_id,
                        item.channel_id,
                        owner_kind,
                        biography_subject_id,
                        accepted_for_apply,
                        len(audit_candidates),
                        extractor_backend_name,
                    )

                if saved:
                    try:
                        await self._refresh_global_biography_summary_from_facts(
                            subject_kind=biography_subject_kind,
                            subject_id=biography_subject_id,
                            facts_user_id=fact_owner_user_id,
                            source_guild_id=item.guild_id,
                        )
                    except Exception:
                        logger.exception(
                            "Biography summary refresh failed for subject=%s:%s",
                            biography_subject_kind,
                            biography_subject_id,
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Profile worker error for guild=%s channel=%s user=%s message_id=%s",
                    getattr(item, "guild_id", ""),
                    getattr(item, "channel_id", ""),
                    getattr(item, "user_id", ""),
                    getattr(item, "message_id", 0),
                )
            finally:
                self.profile_queue.task_done()

    def _enqueue_summary_update(
        self,
        guild_id: str,
        channel_id: str,
        user_id: str,
        *,
        modality: str = "text",
        source: str = "unknown",
        quality: float | None = None,
    ) -> None:
        if not self.settings.summary_enabled:
            return
        is_voice = str(modality or "").strip().lower() == "voice"
        key = (guild_id, channel_id, user_id)
        if key in self.summary_pending_keys:
            if is_voice and callable(getattr(self, "_record_voice_memory_diag", None)):
                self._record_voice_memory_diag(
                    stage="summary_queue",
                    outcome="skip",
                    reason="already_pending",
                    guild_id=guild_id,
                    channel_id=channel_id,
                    user_id=user_id,
                    source=source,
                )
            return
        self.summary_pending_keys.add(key)

        item = PendingSummaryUpdate(guild_id=guild_id, channel_id=channel_id, user_id=user_id)
        if self.summary_queue.full():
            if is_voice and callable(getattr(self, "_record_voice_memory_diag", None)):
                self._record_voice_memory_diag(
                    stage="summary_queue",
                    outcome="drop",
                    reason="queue_full_drop_oldest",
                    guild_id=guild_id,
                    channel_id=channel_id,
                    user_id=user_id,
                    source=source,
                )
            try:
                dropped = self.summary_queue.get_nowait()
                self.summary_queue.task_done()
                self.summary_pending_keys.discard((dropped.guild_id, dropped.channel_id, dropped.user_id))
            except asyncio.QueueEmpty:
                pass

        try:
            self.summary_queue.put_nowait(item)
            if is_voice and callable(getattr(self, "_record_voice_memory_diag", None)):
                self._record_voice_memory_diag(
                    stage="summary_queue",
                    outcome="queued",
                    reason="ok",
                    guild_id=guild_id,
                    channel_id=channel_id,
                    user_id=user_id,
                    source=source,
                    quality=f"{float(quality):.3f}" if quality is not None else None,
                )
        except asyncio.QueueFull:
            if is_voice and callable(getattr(self, "_record_voice_memory_diag", None)):
                self._record_voice_memory_diag(
                    stage="summary_queue",
                    outcome="drop",
                    reason="queue_full_put_failed",
                    guild_id=guild_id,
                    channel_id=channel_id,
                    user_id=user_id,
                    source=source,
                )
            pass

    async def _summary_worker(self) -> None:
        while True:
            item = await self.summary_queue.get()
            key = (item.guild_id, item.channel_id, item.user_id)
            try:
                await self._refresh_summary(item.guild_id, item.channel_id, item.user_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Summary worker error for guild=%s channel=%s user=%s",
                    getattr(item, "guild_id", ""),
                    getattr(item, "channel_id", ""),
                    getattr(item, "user_id", ""),
                )
            finally:
                self.summary_pending_keys.discard(key)
                self.summary_queue.task_done()

    async def _refresh_summary(self, guild_id: str, channel_id: str, user_id: str) -> None:
        existing = await self.memory.get_dialogue_summary(guild_id, user_id, channel_id)
        previous_count = as_int(existing.get("source_user_messages", 0), 0) if existing else 0
        current_count = await self.memory.count_user_messages_in_channel(guild_id, channel_id, user_id)

        if current_count - previous_count < self.settings.summary_min_new_user_messages:
            return

        rows = await self.memory.get_recent_dialogue_messages(
            guild_id=guild_id,
            channel_id=channel_id,
            user_id=user_id,
            limit=self.settings.summary_window_messages,
        )
        if len(rows) < 2:
            return

        previous_summary = str(existing.get("summary_text", "")).strip() if existing else ""
        dialogue_lines: list[str] = []
        for row in rows:
            role = str(row.get("role") or "")
            label = str(row.get("author_label") or "user")
            content = str(row.get("content") or "").strip()
            if not content:
                continue
            speaker = "assistant" if role == "assistant" else label
            dialogue_lines.append(f"{speaker}: {content}")

        if not dialogue_lines:
            return

        max_chars = max(220, self.settings.summary_max_chars)
        prompt_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": build_summary_update_system_prompt(max_chars),
            },
            {
                "role": "user",
                "content": build_summary_update_user_prompt(previous_summary, dialogue_lines),
            },
        ]

        summary = await self.llm.chat(prompt_messages, temperature=0.2, max_output_tokens=620)
        summary = collapse_spaces(summary)
        if not summary:
            return
        if len(summary) > max_chars:
            summary = truncate(summary, max_chars)

        last_message_id = as_int(rows[-1].get("message_id"), 0)
        await self.memory.upsert_dialogue_summary(
            guild_id=guild_id,
            user_id=user_id,
            channel_id=channel_id,
            summary_text=summary,
            source_user_messages=current_count,
            last_message_id=last_message_id,
        )
        label = await self._resolve_user_label(guild_id, user_id)
        logger.info("[memory.summary] user=%s channel=%s chars=%s", label, channel_id, len(summary))
