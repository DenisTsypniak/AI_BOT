from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..common import PendingProfileUpdate, PendingSummaryUpdate, as_int, collapse_spaces, truncate
from ...prompts.memory import build_summary_update_system_prompt, build_summary_update_user_prompt

logger = logging.getLogger("live_role_bot")


class WorkersMixin:
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
        except Exception:
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
                except Exception:
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
            except Exception as exc:
                logger.debug("Persona decay worker error: %s", exc)

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
            except Exception as exc:
                logger.debug("Persona reflection worker error: %s", exc)

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
        )
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
        if text.startswith(("!", "/", ".", "#")):
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
        if lower in filler:
            return False

        if item.modality == "voice":
            # Be slightly stricter on STT-driven memory extraction to reduce garbage facts.
            min_conf = max(float(self.settings.transcription_min_confidence), 0.54)
            if float(item.quality) < min_conf:
                return False
            if len(text) < 10 and "?" not in text and "!" not in text:
                return False

        return True

    async def _profile_worker(self) -> None:
        while True:
            item = await self.profile_queue.get()
            try:
                persona_engine = getattr(self, "persona_engine", None)
                persona_isolation_on = self._persona_queue_isolation_enabled()
                run_persona_fallback = not (persona_isolation_on and bool(getattr(item, "persona_ingest_enqueued", False)))
                if persona_engine is not None and bool(getattr(persona_engine, "enabled", False)) and run_persona_fallback:
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
                result = await self.memory_extractor.extract(
                    user_text=item.user_text,
                    preferred_language=self.settings.preferred_response_language,
                )
                if result is None or not result.facts:
                    continue

                saved = 0
                for fact in result.facts:
                    if not fact.value:
                        continue
                    extractor_name = "gemini_profile_extractor"
                    if item.modality == "voice":
                        extractor_name = f"{extractor_name}:voice"
                    if item.source:
                        extractor_name = f"{extractor_name}:{item.source}"
                    await self.memory.upsert_user_fact(
                        guild_id=item.guild_id,
                        user_id=item.user_id,
                        fact_key=fact.key,
                        fact_value=fact.value,
                        fact_type=fact.fact_type,
                        confidence=fact.confidence,
                        importance=fact.importance,
                        message_id=item.message_id,
                        extractor=extractor_name[:120],
                    )
                    saved += 1

                if saved:
                    label = await self._resolve_user_label(item.guild_id, item.user_id)
                    logger.info("[memory.facts] user=%s saved=%s", label, saved)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Profile worker error: %s", exc)
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
            except Exception as exc:
                logger.debug("Summary worker error: %s", exc)
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
