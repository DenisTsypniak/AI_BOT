from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

import discord

from ..common import ConversationSessionState, chunk_text, collapse_spaces, tokenize, truncate

logger = logging.getLogger("live_role_bot")


class MessageMixin:
    def _voice_memory_diag_state(self) -> dict[str, Any]:
        state = getattr(self, "_voice_memory_diag", None)
        if isinstance(state, dict):
            return state
        state = {
            "started_at_monotonic": time.monotonic(),
            "counters": {},
            "last_event": {},
            "recent": [],
        }
        setattr(self, "_voice_memory_diag", state)
        return state

    def _record_voice_memory_diag(
        self,
        *,
        stage: str,
        outcome: str,
        reason: str = "",
        **fields: object,
    ) -> None:
        state = self._voice_memory_diag_state()
        counters = state.setdefault("counters", {})
        if not isinstance(counters, dict):
            counters = {}
            state["counters"] = counters
        stage_key = str(stage or "unknown").strip() or "unknown"
        outcome_key = str(outcome or "unknown").strip() or "unknown"
        reason_key = str(reason or "").strip() or "none"
        key = f"{stage_key}.{outcome_key}.{reason_key}"
        try:
            counters[key] = int(counters.get(key, 0) or 0) + 1
        except (TypeError, ValueError):
            counters[key] = 1
        try:
            counters["__total__"] = int(counters.get("__total__", 0) or 0) + 1
        except (TypeError, ValueError):
            counters["__total__"] = 1

        compact_fields: dict[str, object] = {}
        for k, v in fields.items():
            if v is None:
                continue
            text = str(v)
            compact_fields[str(k)] = text[:120]
        event = {
            "ts_monotonic": round(time.monotonic(), 3),
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
            f"[voice.mem] stage={stage_key} outcome={outcome_key} reason={reason_key}"
            + "".join(f" {k}={v}" for k, v in compact_fields.items())
        )
        if outcome_key in {"error", "exception"}:
            logger.warning(log_line)
        elif outcome_key in {"drop", "skip"}:
            logger.debug(log_line)
        else:
            logger.debug(log_line)

    def _voice_memory_diag_snapshot(self) -> dict[str, Any]:
        state = self._voice_memory_diag_state()
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

        drop_entries = []
        error_entries = []
        for key, value in counters.items():
            if not isinstance(key, str):
                continue
            if key == "__total__":
                continue
            try:
                count = int(value or 0)
            except (TypeError, ValueError):
                continue
            if ".drop." in key:
                drop_entries.append((count, key))
            if ".error." in key:
                error_entries.append((count, key))
        drop_entries.sort(reverse=True)
        error_entries.sort(reverse=True)
        drop_total = 0
        for count, _key in drop_entries:
            drop_total += count
        error_total = 0
        for count, _key in error_entries:
            error_total += count

        def _find_last_event_by_outcome(outcomes: set[str]) -> dict[str, Any]:
            recent = state.get("recent", [])
            if not isinstance(recent, list):
                return {}
            for item in reversed(recent):
                if not isinstance(item, dict):
                    continue
                if str(item.get("outcome") or "") not in outcomes:
                    continue
                return dict(item)
            return {}

        def _compact_recent_events() -> list[dict[str, Any]]:
            recent = state.get("recent", [])
            if not isinstance(recent, list):
                return []
            out: list[dict[str, Any]] = []
            for item in recent[-8:]:
                if not isinstance(item, dict):
                    continue
                fields = item.get("fields", {}) if isinstance(item.get("fields"), dict) else {}
                out.append(
                    {
                        "ts_unix": item.get("ts_unix"),
                        "stage": str(item.get("stage") or ""),
                        "outcome": str(item.get("outcome") or ""),
                        "reason": str(item.get("reason") or ""),
                        "guild_id": str(fields.get("guild_id") or ""),
                        "channel_id": str(fields.get("channel_id") or ""),
                        "user_id": str(fields.get("user_id") or ""),
                        "source": str(fields.get("source") or ""),
                        "message_id": str(fields.get("message_id") or ""),
                    }
                )
            return out

        def _event_with_iso(event: dict[str, Any]) -> dict[str, Any]:
            if not event:
                return {}
            result = dict(event)
            try:
                ts_val = float(result.get("ts_unix", 0.0) or 0.0)
            except (TypeError, ValueError):
                ts_val = 0.0
            if ts_val > 0:
                result["ts_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts_val))
            return result

        queue_full_drops = (
            _sum_prefix("voice_turn_queue.drop.queue_full_drop_")
            + _sum_prefix("profile_queue.drop.queue_full_drop_")
            + _sum_prefix("summary_queue.drop.queue_full_drop_")
            + _sum_prefix("persona_queue.drop.queue_full_drop_")
        )
        queue_drop_oldest = (
            _sum_prefix("voice_turn_queue.drop.queue_full_drop_oldest")
            + _sum_prefix("profile_queue.drop.queue_full_drop_oldest")
            + _sum_prefix("summary_queue.drop.queue_full_drop_oldest")
            + _sum_prefix("persona_queue.drop.queue_full_drop_oldest")
        )
        queue_put_errors = (
            _sum_prefix("voice_turn_queue.drop.queue_full_put_failed")
            + _sum_prefix("profile_queue.drop.queue_full_put_failed")
            + _sum_prefix("summary_queue.drop.queue_full_put_failed")
            + _sum_prefix("persona_queue.drop.queue_full_put_failed")
        )
        enqueue_errors = (
            _sum_prefix("profile_queue.error.enqueue_failed")
            + _sum_prefix("summary_queue.error.enqueue_failed")
            + _sum_prefix("persona_queue.error.")
        )
        save_message_errors = (
            _sum_prefix("native_user_message.error.save_message_failed")
            + _sum_prefix("native_assistant_message.error.save_message_failed")
        )
        duplicate_transcript_drops = (
            _sum_prefix("native_user_callback.drop.duplicate_transcript")
            + _sum_prefix("native_assistant_callback.drop.duplicate_transcript")
        )
        echo_filter_drops = _sum_prefix("native_user_callback.drop.echo_filter")
        no_binding_drops = (
            _sum_prefix("native_user_callback.drop.no_voice_text_channel_binding")
            + _sum_prefix("voice_stt.drop.no_voice_text_channel_binding")
        )
        stt_errors = (
            _sum_prefix("voice_stt.error.")
            + _sum_prefix("voice_stt.drop.empty_transcript")
            + _sum_prefix("voice_stt.drop.low_confidence")
        )
        reason_groups = {
            "stt": int(stt_errors),
            "queue": int(queue_full_drops + queue_put_errors),
            "save": int(save_message_errors + _sum_prefix("voice_stt.error.save_stt_turn_failed")),
            "enqueue": int(enqueue_errors),
            "filters": int(
                echo_filter_drops
                + duplicate_transcript_drops
                + no_binding_drops
            ),
        }
        last_correlation = {}
        last_err = _find_last_event_by_outcome({"error", "exception"})
        last_drop = _find_last_event_by_outcome({"drop"})
        chosen = last_err or last_drop
        if chosen:
            chosen_fmt = _event_with_iso(chosen)
            fields = chosen_fmt.get("fields", {}) if isinstance(chosen_fmt.get("fields"), dict) else {}
            last_correlation = {
                "ts_iso": str(chosen_fmt.get("ts_iso") or ""),
                "stage": str(chosen_fmt.get("stage") or ""),
                "outcome": str(chosen_fmt.get("outcome") or ""),
                "reason": str(chosen_fmt.get("reason") or ""),
                "guild_id": str(fields.get("guild_id") or ""),
                "channel_id": str(fields.get("channel_id") or ""),
                "user_id": str(fields.get("user_id") or ""),
                "message_id": str(fields.get("message_id") or ""),
                "source": str(fields.get("source") or ""),
            }

        return {
            "uptime_sec": round(uptime_sec, 1),
            "total_events": int(counters.get("__total__", 0) or 0),
            "drop_events": drop_total,
            "error_events": error_total,
            "saved_user_messages": _sum_prefix("native_user_message.saved."),
            "saved_assistant_messages": _sum_prefix("native_assistant_message.saved."),
            "stt_saved": _sum_prefix("voice_stt.save_stt_turn."),
            "stt_dropped_low_conf": _sum_prefix("voice_stt.drop.low_confidence"),
            "stt_dropped_empty": _sum_prefix("voice_stt.drop.empty_transcript"),
            "queue_full_drops": queue_full_drops,
            "queue_drop_oldest": queue_drop_oldest,
            "queue_put_errors": queue_put_errors,
            "enqueue_errors": enqueue_errors,
            "save_message_errors": save_message_errors,
            "duplicate_transcript_drops": duplicate_transcript_drops,
            "echo_filter_drops": echo_filter_drops,
            "no_voice_text_channel_binding_drops": no_binding_drops,
            "reason_groups": reason_groups,
            "top_drop_reasons": [
                {"key": key, "count": count}
                for count, key in drop_entries[:6]
            ],
            "top_error_reasons": [
                {"key": key, "count": count}
                for count, key in error_entries[:6]
            ],
            "last_event": _event_with_iso(
                dict(state.get("last_event", {})) if isinstance(state.get("last_event"), dict) else {}
            ),
            "last_drop_event": _event_with_iso(_find_last_event_by_outcome({"drop"})),
            "last_error_event": _event_with_iso(_find_last_event_by_outcome({"error", "exception"})),
            "last_correlation": last_correlation,
            "recent_events": _compact_recent_events(),
        }

    def _format_voice_diag_admin_report(self, *, include_recent: bool = True) -> str:
        diag = self._voice_memory_diag_snapshot()
        lines = [
            "Voice Memory Diagnostics (admin)",
            f"- uptime_sec: {diag.get('uptime_sec', 0)}",
            f"- total_events: {diag.get('total_events', 0)}",
            f"- drop_events: {diag.get('drop_events', 0)}",
            f"- error_events: {diag.get('error_events', 0)}",
            f"- user_saved: {diag.get('saved_user_messages', 0)} assistant_saved: {diag.get('saved_assistant_messages', 0)} stt_saved: {diag.get('stt_saved', 0)}",
            (
                f"- drop_buckets: low_conf={diag.get('stt_dropped_low_conf', 0)} "
                f"empty_stt={diag.get('stt_dropped_empty', 0)} queue_full={diag.get('queue_full_drops', 0)} "
                f"queue_drop_oldest={diag.get('queue_drop_oldest', 0)} queue_put_error={diag.get('queue_put_errors', 0)}"
            ),
            (
                f"- error_buckets: save_message={diag.get('save_message_errors', 0)} "
                f"enqueue={diag.get('enqueue_errors', 0)}"
            ),
            (
                f"- filter_buckets: duplicate={diag.get('duplicate_transcript_drops', 0)} "
                f"echo={diag.get('echo_filter_drops', 0)} "
                f"no_binding={diag.get('no_voice_text_channel_binding_drops', 0)}"
            ),
        ]
        groups = diag.get("reason_groups", {})
        if isinstance(groups, dict):
            lines.append(
                "- grouped: "
                + f"stt={int(groups.get('stt', 0) or 0)} "
                + f"queue={int(groups.get('queue', 0) or 0)} "
                + f"save={int(groups.get('save', 0) or 0)} "
                + f"enqueue={int(groups.get('enqueue', 0) or 0)} "
                + f"filters={int(groups.get('filters', 0) or 0)}"
            )
        last_error = diag.get("last_error_event", {}) if isinstance(diag.get("last_error_event"), dict) else {}
        if last_error:
            lines.append(
                "- last_error: "
                + f"{last_error.get('ts_iso', '?')} "
                + f"{last_error.get('stage', '?')}.{last_error.get('reason', '?')}"
            )
        last_drop = diag.get("last_drop_event", {}) if isinstance(diag.get("last_drop_event"), dict) else {}
        if last_drop:
            lines.append(
                "- last_drop: "
                + f"{last_drop.get('ts_iso', '?')} "
                + f"{last_drop.get('stage', '?')}.{last_drop.get('reason', '?')}"
            )
        last_corr = diag.get("last_correlation", {}) if isinstance(diag.get("last_correlation"), dict) else {}
        if last_corr:
            lines.append(
                "- correlation: "
                + f"{last_corr.get('ts_iso', '?')} "
                + f"{last_corr.get('stage', '?')}.{last_corr.get('outcome', '?')}.{last_corr.get('reason', '?')}"
                + (f" g={last_corr.get('guild_id')}" if last_corr.get("guild_id") else "")
                + (f" c={last_corr.get('channel_id')}" if last_corr.get("channel_id") else "")
                + (f" u={last_corr.get('user_id')}" if last_corr.get("user_id") else "")
                + (f" mid={last_corr.get('message_id')}" if last_corr.get("message_id") else "")
            )
        top_drop = diag.get("top_drop_reasons", [])
        if isinstance(top_drop, list) and top_drop:
            lines.append("Top drops:")
            for row in top_drop[:6]:
                if not isinstance(row, dict):
                    continue
                lines.append(f"  {truncate(str(row.get('key', '')), 100)} x{int(row.get('count', 0) or 0)}")
        top_err = diag.get("top_error_reasons", [])
        if isinstance(top_err, list) and top_err:
            lines.append("Top errors:")
            for row in top_err[:6]:
                if not isinstance(row, dict):
                    continue
                lines.append(f"  {truncate(str(row.get('key', '')), 100)} x{int(row.get('count', 0) or 0)}")
        if include_recent:
            recent = diag.get("recent_events", [])
            if isinstance(recent, list) and recent:
                lines.append("Recent:")
                for row in recent[-6:]:
                    if not isinstance(row, dict):
                        continue
                    ts_val = row.get("ts_unix")
                    ts_iso = "?"
                    try:
                        ts_num = float(ts_val or 0.0)
                    except (TypeError, ValueError):
                        ts_num = 0.0
                    if ts_num > 0:
                        ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts_num))
                    lines.append(
                        "  "
                        + f"{ts_iso} {row.get('stage', '?')}.{row.get('outcome', '?')}.{row.get('reason', '?')}"
                        + (f" g={row.get('guild_id')}" if row.get("guild_id") else "")
                        + (f" c={row.get('channel_id')}" if row.get("channel_id") else "")
                        + (f" u={row.get('user_id')}" if row.get("user_id") else "")
                        + (f" src={row.get('source')}" if row.get("source") else "")
                        + (f" mid={row.get('message_id')}" if row.get("message_id") else "")
                    )
        return "\n".join(lines)

    def _get_conversation_state(self, channel_key: str) -> ConversationSessionState:
        state = self.conversation_states.get(channel_key)
        if state is None:
            state = ConversationSessionState()
            self.conversation_states[channel_key] = state
        return state


    @staticmethod
    def _remember_recent(items: list[str], value: str, *, limit: int = 6) -> None:
        cleaned = collapse_spaces(value)
        if not cleaned:
            return
        lowered = cleaned.casefold()
        items[:] = [item for item in items if item.casefold() != lowered]
        items.append(cleaned)
        if len(items) > limit:
            del items[:-limit]

    def _extract_callback_moment(self, text: str) -> str:
        cleaned = collapse_spaces(text)
        if len(cleaned) < 8 or len(cleaned) > 220:
            return ""
        if cleaned.startswith(("!", "/", ".")):
            return ""

        lower = cleaned.casefold()
        notable_markers = (
            "хочу",
            "буду",
            "завтра",
            "потім",
            "сьогодні",
            "вчора",
            "жиза",
            "крінж",
            "імба",
            "лол",
            "bruh",
            "gg",
            "аха",
            "хаха",
        )
        if "?" not in cleaned and "!" not in cleaned and not any(marker in lower for marker in notable_markers):
            return ""

        first_clause = cleaned
        for sep in ("...", "…", ". ", "! ", "? ", "; "):
            idx = first_clause.find(sep)
            if idx >= 0:
                first_clause = first_clause[: idx + (1 if sep[-1] in ".!?" else 0)]
                break

        return truncate(first_clause.strip(), 110)

    def _update_session_state_from_user(
        self,
        channel_key: str,
        user_text: str,
        *,
        user_label: str | None = None,
        modality: str | None = None,
    ) -> None:
        state = self._get_conversation_state(channel_key)
        cleaned = collapse_spaces(user_text)
        if not cleaned:
            return

        state.turn_count += 1
        if state.turn_count >= 14:
            state.familiarity = "regular"
        elif state.turn_count >= 5:
            state.familiarity = "warm"
        else:
            state.familiarity = "new"
        if modality:
            state.last_modality = modality
        if user_label:
            self._remember_recent(state.recent_speakers, user_label, limit=6)
            state.group_vibe = "group" if len(state.recent_speakers) >= 2 else "solo"

        text_cf = cleaned.casefold()
        tokens = tokenize(cleaned)
        upset_hits = {
            "погано",
            "сумно",
            "злий",
            "зла",
            "тривога",
            "stress",
            "stressed",
            "депрес",
            "бісить",
        }
        playful_hits = {
            "bruh",
            "gg",
            "крінж",
            "імба",
            "лол",
            "жиза",
            "мем",
            "кайф",
        }
        tech_hits = {
            "discord",
            "voice",
            "mic",
            "мік",
            "audio",
            "звук",
            "бот",
            "role",
            "server",
            "ping",
            "lag",
            "код",
            "python",
        }
        sleepy_hits = {
            "сплю",
            "сонний",
            "сонна",
            "втом",
            "tired",
            "sleepy",
        }

        if any(k in text_cf for k in sleepy_hits):
            state.mood = "sleepy"
            state.energy = "low"
        elif any(k in text_cf for k in upset_hits):
            state.mood = "supportive"
            state.tease_level = "off"
        elif tokens.intersection(playful_hits):
            state.mood = "playful"
            state.tease_level = "medium"
        elif tokens.intersection(tech_hits):
            state.mood = "focused"
            state.tease_level = "low"
        else:
            state.mood = "neutral"

        if "!" in cleaned or cleaned.isupper():
            state.energy = "high"
        elif "..." in cleaned or "…" in cleaned:
            state.energy = "low"
        elif state.energy not in {"low", "high"}:
            state.energy = "medium"

        if "?" in cleaned:
            state.open_loop = truncate(cleaned, 110)

        topic_words = [w for w in cleaned.split() if len(w) >= 3]
        if topic_words:
            state.topic_hint = truncate(" ".join(topic_words[:8]), 72)
            self._remember_recent(state.recent_topics, truncate(" ".join(topic_words[:4]), 60), limit=6)
        callback_moment = self._extract_callback_moment(cleaned)
        if callback_moment:
            self._remember_recent(state.callback_moments, callback_moment, limit=5)
        state.last_user_text = truncate(cleaned, 220)

    def _looks_like_detail_request(self, user_text: str) -> bool:
        text = collapse_spaces(user_text).casefold()
        if not text:
            return False
        markers = (
            "детал",
            "подроб",
            "розжуй",
            "докладно",
            "поясни",
            "чому",
            "як саме",
            "why",
            "explain",
            "details",
            "step by step",
        )
        return any(marker in text for marker in markers)

    def _reply_similarity(self, left: str, right: str) -> float:
        a = tokenize(collapse_spaces(left))
        b = tokenize(collapse_spaces(right))
        if not a or not b:
            return 0.0
        return len(a.intersection(b)) / float(max(1, min(len(a), len(b))))

    def _limit_question_count(self, text: str, max_questions: int = 1) -> str:
        if text.count("?") <= max_questions:
            return text
        kept = 0
        out: list[str] = []
        for ch in text:
            if ch == "?":
                kept += 1
                out.append("?" if kept <= max_questions else ".")
            else:
                out.append(ch)
        return "".join(out)

    def _vary_repeated_opener(self, channel_key: str, text: str) -> str:
        state = self._get_conversation_state(channel_key)
        cleaned = text.lstrip()
        if not cleaned:
            return text
        parts = cleaned.split(maxsplit=1)
        opener = parts[0].strip(":,.!?()[]{}\"'").casefold()
        if not opener:
            return text
        if state.repeated_openers.get(opener, 0) < 2:
            return text

        replacements = {
            "ну": "слухай,",
            "короч": "йой,",
            "йой": "ну шо,",
            "слухай": "короч,",
            "ок": "та ок,",
            "ага": "пон,",
        }
        repl = replacements.get(opener)
        if not repl:
            return text
        tail = parts[1] if len(parts) > 1 else ""
        prefix = text[: len(text) - len(cleaned)]
        return (prefix + f"{repl} {tail}".strip()).strip()

    def _apply_human_reaction_prefix(self, channel_key: str, user_text: str, text: str, modality: str) -> str:
        state = self._get_conversation_state(channel_key)
        if modality != "voice":
            return text

        cleaned = text.lstrip()
        if not cleaned:
            return text

        first_token = cleaned.split(maxsplit=1)[0].strip(":,.!?()[]{}\"'").casefold()
        if first_token in {
            "ну",
            "йой",
            "ага",
            "ок",
            "короч",
            "слухай",
            "bruh",
            "мм",
            "ех",
        }:
            return text
        if self._looks_like_detail_request(user_text) or len(cleaned) > 240:
            return text

        prefix_options = {
            "supportive": ["Йой, ", "Та блін, "],
            "playful": ["Аха, ", "bruh, ", "Йой, "],
            "focused": ["Ок, ", "Та дивись, "],
            "sleepy": ["Мм, ", "Ех, "],
            "neutral": ["Ага, ", "Ну шо, "],
        }
        options = prefix_options.get(state.mood, prefix_options["neutral"])
        choice = options[state.turn_count % len(options)]
        return f"{choice}{cleaned}" if text == cleaned else f"{text[:len(text)-len(cleaned)]}{choice}{cleaned}"

    def _shape_human_reply(self, channel_key: str, user_text: str, reply: str, modality: str) -> str:
        shaped = reply.strip()
        if not shaped:
            return reply

        shaped = self._limit_question_count(shaped, max_questions=1)
        shaped = self._vary_repeated_opener(channel_key, shaped)
        shaped = self._apply_human_reaction_prefix(channel_key, user_text, shaped, modality)

        if modality == "voice" and "\n" in shaped and not self._looks_like_detail_request(user_text):
            shaped = collapse_spaces(shaped.replace("\n", " "))

        if not self._looks_like_detail_request(user_text):
            state = self._get_conversation_state(channel_key)
            if modality == "voice":
                soft_limit = 210 if state.energy == "high" else 240
                if state.mood == "supportive":
                    soft_limit = min(soft_limit, 220)
            else:
                soft_limit = 340 if state.familiarity == "regular" else 380
            if len(shaped) > soft_limit:
                shaped = truncate(shaped, soft_limit)

        recent = self.recent_assistant_replies.get(channel_key, [])
        if recent and self._reply_similarity(shaped, recent[-1]) >= 0.75:
            shaped = truncate(shaped, max(120, min(len(shaped), 220)))

        return shaped

    def _register_bot_reply_state(self, channel_key: str, reply: str) -> None:
        state = self._get_conversation_state(channel_key)
        cleaned = collapse_spaces(reply)
        if not cleaned:
            return
        state.last_bot_text = truncate(cleaned, 220)
        opener = cleaned.split(maxsplit=1)[0].strip(":,.!?()[]{}\"'").casefold()
        if opener:
            state.repeated_openers[opener] = state.repeated_openers.get(opener, 0) + 1
            if len(state.repeated_openers) > 20:
                state.repeated_openers = dict(list(state.repeated_openers.items())[-20:])

        recent = self.recent_assistant_replies.setdefault(channel_key, [])
        recent.append(cleaned)
        if len(recent) > 6:
            del recent[:-6]

    async def _save_native_user_transcript(
        self,
        guild_id: int,
        channel_id: int,
        user_id: int,
        user_label: str,
        text: str,
        source: str,
        quality: float,
    ) -> None:
        guild_key = str(guild_id)
        channel_key = str(channel_id)
        user_key = str(user_id)
        cleaned = collapse_spaces(text)
        if not cleaned:
            self._record_voice_memory_diag(
                stage="native_user_message",
                outcome="drop",
                reason="empty_transcript",
                guild_id=guild_id,
                channel_id=channel_id,
                user_id=user_id,
                source=source,
            )
            return

        lock = self.channel_locks[channel_key]
        try:
            async with lock:
                self._update_session_state_from_user(channel_key, cleaned, user_label=user_label, modality="voice")
                role_id = await self._resolve_role(guild_key)
                session_id = await self.memory.get_or_create_session(
                    guild_id=guild_key,
                    channel_id=channel_key,
                    mode="voice",
                    role_id=role_id,
                )
                message_id = await self.memory.save_message(
                    session_id=session_id,
                    guild_id=guild_key,
                    channel_id=channel_key,
                    user_id=user_key,
                    author_label=user_label,
                    role="user",
                    modality="voice",
                    content_raw=cleaned,
                    content_clean=cleaned,
                    source=source,
                    quality=quality,
                )
        except Exception as exc:
            logger.exception(
                "Failed to save native user transcript message guild=%s channel=%s user=%s source=%s",
                guild_id,
                channel_id,
                user_id,
                source,
            )
            self._record_voice_memory_diag(
                stage="native_user_message",
                outcome="error",
                reason="save_message_failed",
                guild_id=guild_id,
                channel_id=channel_id,
                user_id=user_id,
                source=source,
                error=exc,
            )
            raise

        self._record_voice_memory_diag(
            stage="native_user_message",
            outcome="saved",
            reason="ok",
            guild_id=guild_id,
            channel_id=channel_id,
            user_id=user_id,
            source=source,
            message_id=message_id,
            quality=f"{float(quality):.3f}",
        )
        try:
            self._enqueue_profile_update(
                guild_key,
                channel_key,
                user_key,
                message_id,
                cleaned,
                user_label=user_label,
                modality="voice",
                source=source,
                quality=quality,
            )
        except Exception as exc:
            logger.exception(
                "Failed to enqueue profile update from native user transcript guild=%s channel=%s user=%s source=%s",
                guild_id,
                channel_id,
                user_id,
                source,
            )
            self._record_voice_memory_diag(
                stage="profile_queue",
                outcome="error",
                reason="enqueue_failed",
                guild_id=guild_id,
                channel_id=channel_id,
                user_id=user_id,
                source=source,
                error=exc,
            )
            raise

        try:
            self._enqueue_summary_update(
                guild_key,
                channel_key,
                user_key,
                modality="voice",
                source=source,
                quality=quality,
            )
        except Exception as exc:
            logger.exception(
                "Failed to enqueue summary update from native user transcript guild=%s channel=%s user=%s source=%s",
                guild_id,
                channel_id,
                user_id,
                source,
            )
            self._record_voice_memory_diag(
                stage="summary_queue",
                outcome="error",
                reason="enqueue_failed",
                guild_id=guild_id,
                channel_id=channel_id,
                user_id=user_id,
                source=source,
                error=exc,
            )
            raise

    async def _save_native_assistant_transcript(
        self,
        guild_id: int,
        channel_id: int,
        text: str,
        source: str,
    ) -> None:
        guild_key = str(guild_id)
        channel_key = str(channel_id)
        cleaned = collapse_spaces(text)
        if not cleaned:
            self._record_voice_memory_diag(
                stage="native_assistant_message",
                outcome="drop",
                reason="empty_transcript",
                guild_id=guild_id,
                channel_id=channel_id,
                source=source,
            )
            return

        lock = self.channel_locks[channel_key]
        try:
            async with lock:
                role_id = await self._resolve_role(guild_key)
                session_id = await self.memory.get_or_create_session(
                    guild_id=guild_key,
                    channel_id=channel_key,
                    mode="voice",
                    role_id=role_id,
                )
                assistant_message_id = await self.memory.save_message(
                    session_id=session_id,
                    guild_id=guild_key,
                    channel_id=channel_key,
                    user_id=str(self.user.id if self.user else 0),
                    author_label="assistant",
                    role="assistant",
                    modality="voice",
                    content_raw=cleaned,
                    content_clean=cleaned,
                    source=source,
                    quality=1.0,
                )
                if bool(getattr(self.settings, "memory_persona_self_facts_enabled", True)):
                    self._enqueue_profile_update(
                        guild_id=guild_key,
                        channel_id=channel_key,
                        user_id=str(self.user.id if self.user else 0),
                        message_id=int(assistant_message_id),
                        user_text=cleaned,
                        user_label="assistant",
                        modality="voice",
                        source=source,
                        quality=1.0,
                        speaker_role="assistant",
                        fact_owner_kind="persona",
                        fact_owner_id=str(getattr(self.settings, "persona_id", "") or "persona"),
                    )
        except Exception as exc:
            logger.exception(
                "Failed to save native assistant transcript message guild=%s channel=%s source=%s",
                guild_id,
                channel_id,
                source,
            )
            self._record_voice_memory_diag(
                stage="native_assistant_message",
                outcome="error",
                reason="save_message_failed",
                guild_id=guild_id,
                channel_id=channel_id,
                source=source,
                error=exc,
            )
            raise
        self._record_voice_memory_diag(
            stage="native_assistant_message",
            outcome="saved",
            reason="ok",
            guild_id=guild_id,
            channel_id=channel_id,
            source=source,
        )

    async def _send_chunks(
        self,
        channel: discord.abc.Messageable,
        text: str,
        reference: discord.Message | None = None,
    ) -> None:
        for index, chunk in enumerate(chunk_text(text, 1900)):
            kwargs: dict[str, Any] = {}
            if index == 0 and reference is not None:
                kwargs["reference"] = reference
            await channel.send(chunk, **kwargs)

    def _format_status_report(self, snapshot: dict[str, Any]) -> str:
        uptime_sec = float(snapshot.get("uptime_sec", 0.0) or 0.0)
        uptime = f"{uptime_sec:.1f}s"
        memory_backend = str(snapshot.get("memory_backend", "?"))
        memory_ping_ok = bool(snapshot.get("memory_ping_ok"))
        memory_ping_error = str(snapshot.get("memory_ping_error") or "")
        queues = snapshot.get("queues", {}) if isinstance(snapshot.get("queues"), dict) else {}
        workers = snapshot.get("workers", {}) if isinstance(snapshot.get("workers"), dict) else {}
        voice_clients = snapshot.get("voice_clients", [])
        bridge = snapshot.get("livekit_bridge", {}) if isinstance(snapshot.get("livekit_bridge"), dict) else {}
        bridge_context = bridge.get("context_sync", {}) if isinstance(bridge.get("context_sync"), dict) else {}
        native = snapshot.get("native_audio", {}) if isinstance(snapshot.get("native_audio"), dict) else {}
        plugins = snapshot.get("plugins", {}) if isinstance(snapshot.get("plugins"), dict) else {}
        voice_ingest = snapshot.get("voice_ingest", {}) if isinstance(snapshot.get("voice_ingest"), dict) else {}
        voice_memory_diag = snapshot.get("voice_memory_diag", {}) if isinstance(snapshot.get("voice_memory_diag"), dict) else {}
        persona_queue_diag = snapshot.get("persona_queue_diag", {}) if isinstance(snapshot.get("persona_queue_diag"), dict) else {}
        persona = snapshot.get("persona_growth", {}) if isinstance(snapshot.get("persona_growth"), dict) else {}

        lines = [
            "Bot Status",
            f"- user: {snapshot.get('user') or '-'}",
            f"- uptime: {uptime}",
            f"- guilds: {snapshot.get('guilds', 0)}",
            f"- memory: {memory_backend} ({'ok' if memory_ping_ok else 'error'})",
        ]
        if memory_ping_error:
            lines.append(f"- memory_error: {truncate(memory_ping_error, 180)}")

        lines.extend(
            [
                (
                    f"- queues: profile={queues.get('profile', '?')} persona={queues.get('persona_event', '?')} "
                    f"summary={queues.get('summary', '?')} voice={queues.get('voice_turn', '?')}"
                ),
                (
                    "- workers: "
                    f"profile={'on' if workers.get('profile') else 'off'} "
                    f"persona_ingest={'on' if workers.get('persona_ingest') else 'off'} "
                    f"summary={'on' if workers.get('summary') else 'off'} "
                    f"voice={'on' if workers.get('voice') else 'off'} "
                    f"flush={'on' if workers.get('voice_flush') else 'off'} "
                    f"reflect={'on' if workers.get('persona_reflection') else 'off'} "
                    f"decay={'on' if workers.get('persona_decay') else 'off'}"
                ),
                (
                    "- voice_ingest: "
                    f"worker={'on' if voice_ingest.get('worker_enabled') else 'off'} "
                    f"fallback={'on' if voice_ingest.get('local_fallback_enabled') else 'off'} "
                    f"bridge_memory_stt={'on' if voice_ingest.get('bridge_memory_stt_enabled') else 'off'}"
                ),
                (
                    "- voice_memory_diag: "
                    f"events={voice_memory_diag.get('total_events', 0)} "
                    f"user_saved={voice_memory_diag.get('saved_user_messages', 0)} "
                    f"stt_saved={voice_memory_diag.get('stt_saved', 0)} "
                    f"low_conf_drop={voice_memory_diag.get('stt_dropped_low_conf', 0)} "
                    f"empty_drop={voice_memory_diag.get('stt_dropped_empty', 0)} "
                    f"queue_full={voice_memory_diag.get('queue_full_drops', 0)} "
                    f"save_err={voice_memory_diag.get('save_message_errors', 0)} "
                    f"enq_err={voice_memory_diag.get('enqueue_errors', 0)}"
                ),
                (
                    "- livekit_context_sync: "
                    f"enabled={'on' if bridge.get('context_sync_enabled') else 'off'} "
                    f"pub={bridge_context.get('published', 0)}/{bridge_context.get('attempts', 0)} "
                    f"skip={bridge_context.get('skipped', 0)} "
                    f"(throttle={bridge_context.get('skipped_throttle', 0)},same={bridge_context.get('skipped_unchanged', 0)}) "
                    f"pub_err={bridge_context.get('publish_errors', 0)} "
                    f"agent_rx={bridge_context.get('agent_updates_received', 0)} "
                    f"agent_apply={bridge_context.get('agent_updates_applied', 0)} "
                    f"agent_ign={bridge_context.get('agent_updates_ignored', 0)} "
                    f"agent_err={bridge_context.get('agent_updates_errors', 0)} "
                    f"ack_invalid={bridge_context.get('agent_ack_invalid', 0)}"
                ),
                (
                    "- persona_growth: "
                    f"configured={'yes' if persona.get('configured') else 'no'} "
                    f"enabled={'yes' if persona.get('enabled') else 'no'} "
                    f"backend_supported={'yes' if persona.get('backend_supported') else 'no'} "
                    f"retrieval={'on' if persona.get('retrieval_enabled') else 'off'} "
                    f"shadow={'on' if persona.get('shadow_mode') else 'off'} "
                    f"cache={persona.get('cache_entries', '?')}"
                ),
                (
                    "- persona_queue: "
                    f"events={persona_queue_diag.get('total_events', 0)} "
                    f"queued={persona_queue_diag.get('queued', 0)} "
                    f"drops={persona_queue_diag.get('drops', 0)} "
                    f"errors={persona_queue_diag.get('errors', 0)} "
                    f"applied={persona_queue_diag.get('worker_applied', 0)} "
                    f"deduped={persona_queue_diag.get('worker_deduped', 0)} "
                    f"fallback={persona_queue_diag.get('fallback_profile_worker', 0)}"
                ),
                f"- plugins: enabled={'yes' if plugins.get('enabled') else 'no'} loaded={'yes' if plugins.get('loaded') else 'no'}",
            ]
        )

        if isinstance(voice_clients, list) and voice_clients:
            lines.append(f"- discord_voice_clients: {len(voice_clients)}")
            for row in voice_clients[:4]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    "  "
                    + f"{row.get('guild_name', '?')} -> {row.get('channel_name', '?')} "
                    + f"(connected={'yes' if row.get('connected') else 'no'}, playing={'yes' if row.get('playing') else 'no'})"
                )
        else:
            lines.append("- discord_voice_clients: 0")

        last_voice_err = (
            voice_memory_diag.get("last_error_event", {})
            if isinstance(voice_memory_diag.get("last_error_event"), dict)
            else {}
        )
        if last_voice_err:
            lines.append(
                "- voice_memory_last_error: "
                + f"{truncate(str(last_voice_err.get('ts_iso', '')), 28)} "
                + f"{truncate(str(last_voice_err.get('stage', '?')), 24)}."
                + f"{truncate(str(last_voice_err.get('reason', '?')), 34)}"
            )

        lines.append(
            f"- livekit_bridge: {'enabled' if bridge.get('enabled') else 'disabled'} "
            f"(sessions={bridge.get('session_count', 0)}, bindings={bridge.get('bindings', 0)})"
        )
        bridge_sessions = bridge.get("sessions", [])
        if isinstance(bridge_sessions, list) and bridge_sessions:
            for row in bridge_sessions[:4]:
                if not isinstance(row, dict):
                    continue
                idle = row.get("idle_sec")
                idle_text = f"{float(idle):.1f}s" if isinstance(idle, (int, float)) else "-"
                ctx_age = row.get("last_context_age_sec")
                ctx_age_text = f"{float(ctx_age):.1f}s" if isinstance(ctx_age, (int, float)) else "-"
                ack_age = row.get("agent_ctx_last_ack_age_sec")
                ack_age_text = f"{float(ack_age):.1f}s" if isinstance(ack_age, (int, float)) else "-"
                lines.append(
                    "  "
                    + f"room={row.get('room_name', '?')} pcm_q={row.get('pcm_queue', '?')} "
                    + f"remote={row.get('remote_streams', '?')} idle={idle_text} "
                    + f"ctx_seq={row.get('context_seq', 0)}({truncate(str(row.get('last_context_reason', '')), 16)}) "
                    + f"ctx_age={ctx_age_text} ctx_bytes={row.get('last_context_payload_bytes', 0)} "
                    + f"agent_ctx_seq={row.get('agent_ctx_last_seq', 0)}({truncate(str(row.get('agent_ctx_last_reason', '')), 16)}) "
                    + f"ack_age={ack_age_text}"
                )
                agent_err = str(row.get("agent_ctx_last_error") or "").strip()
                if agent_err:
                    lines.append("    " + f"agent_ctx_last_error={truncate(agent_err, 90)}")
                ctx_err = str(row.get("last_context_error") or "").strip()
                if ctx_err:
                    lines.append("    " + f"context_publish_error={truncate(ctx_err, 90)}")

        lines.append(
            f"- native_audio: {'enabled' if native.get('enabled') else 'disabled'} "
            f"(sessions={native.get('session_count', 0)})"
        )
        native_sessions = native.get("sessions", [])
        if isinstance(native_sessions, list) and native_sessions:
            for row in native_sessions[:4]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    "  "
                    + f"guild={row.get('guild_id', '?')} ready={'yes' if row.get('ready') else 'no'} "
                    + f"in_q={row.get('input_queue', '?')} out_q={row.get('playback_queue', '?')} "
                    + f"playback={'yes' if row.get('playback_active') else 'no'}"
                )

        return "\n".join(lines)

    def _is_persona_admin_authorized(self, message: discord.Message) -> bool:
        if not bool(getattr(self.settings, "persona_admin_commands_enabled", False)):
            return False
        allow_ids = getattr(self.settings, "persona_allowed_admin_user_ids", set())
        try:
            if int(message.author.id) in allow_ids:
                return True
        except (TypeError, ValueError):
            pass
        if isinstance(message.author, discord.Member):
            with contextlib.suppress(Exception):
                return bool(message.author.guild_permissions.administrator)
        return False

    @staticmethod
    def _parse_user_id_hint(text: str) -> int | None:
        raw = (text or "").strip()
        if not raw:
            return None
        cleaned = raw.replace("<@", "").replace(">", "").replace("!", "").strip()
        if not cleaned.isdigit():
            return None
        try:
            return int(cleaned)
        except ValueError:
            return None

    def _resolve_admin_target_user(self, message: discord.Message, args: list[str]) -> tuple[str, int] | None:
        if message.mentions:
            target = message.mentions[0]
            return (getattr(target, "display_name", target.name), int(target.id))
        if args:
            user_id = self._parse_user_id_hint(args[0])
            if user_id is not None:
                guild = message.guild
                if guild is not None:
                    member = guild.get_member(user_id)
                    if member is not None:
                        return (member.display_name, int(member.id))
                return (f"user:{user_id}", user_id)
        author = message.author
        return (getattr(author, "display_name", author.name), int(author.id))

    def _format_persona_admin_report(self, payload: dict[str, Any]) -> str:
        status = payload.get("status", {}) if isinstance(payload.get("status"), dict) else {}
        if not payload.get("enabled"):
            return (
                "Persona Growth (admin)\n"
                f"- enabled: no\n"
                f"- configured: {'yes' if status.get('configured') else 'no'}\n"
                f"- backend_supported: {'yes' if status.get('backend_supported') else 'no'}\n"
                f"- backend: {status.get('backend', '?')}\n"
            )

        state = payload.get("state", {}) if isinstance(payload.get("state"), dict) else {}
        traits = payload.get("traits", []) if isinstance(payload.get("traits"), list) else []
        reflections = payload.get("reflections", []) if isinstance(payload.get("reflections"), list) else []
        lines = [
            "Persona Growth (admin)",
            f"- persona_id: {status.get('persona_id', '?')}",
            f"- shadow_mode: {'on' if status.get('shadow_mode') else 'off'}",
            f"- retrieval: {'on' if status.get('retrieval_enabled') else 'off'}",
            f"- reflections: {'on' if status.get('reflection_enabled') else 'off'}",
            f"- reflection_apply: {'on' if status.get('reflection_apply_enabled') else 'off'}",
            f"- trait_drift: {'on' if status.get('trait_drift_enabled') else 'off'}",
            f"- decay: {'on' if status.get('decay_enabled') else 'off'}",
            f"- episode_recall_reconfirm: {'on' if status.get('episode_recall_reconfirm_enabled') else 'off'}",
            f"- cache_entries: {status.get('cache_entries', 0)}",
            f"- core_hash: {str(state.get('core_dna_hash', ''))[:12]}",
            f"- policy_version: {state.get('policy_version', '?')}",
            f"- counters: total={state.get('total_messages_seen', 0)} eligible={state.get('eligible_messages_seen', 0)} unique_users={state.get('unique_users_seen', 0)}",
        ]
        if callable(getattr(self, "_voice_memory_diag_snapshot", None)):
            diag = self._voice_memory_diag_snapshot()
            lines.append(
                f"- voice_mem_diag: events={diag.get('total_events', 0)} user_saved={diag.get('saved_user_messages', 0)} "
                f"stt_saved={diag.get('stt_saved', 0)} queue_full={diag.get('queue_full_drops', 0)} "
                f"save_err={diag.get('save_message_errors', 0)} enq_err={diag.get('enqueue_errors', 0)}"
            )
            top_drop = diag.get("top_drop_reasons", [])
            if isinstance(top_drop, list) and top_drop:
                first = top_drop[0] if isinstance(top_drop[0], dict) else {}
                if isinstance(first, dict):
                    lines.append(
                        f"- voice_mem_top_drop: {truncate(str(first.get('key', '')), 70)} x{int(first.get('count', 0) or 0)}"
                    )
            last_err = diag.get("last_error_event", {})
            if isinstance(last_err, dict) and last_err:
                lines.append(
                    "- voice_mem_last_error: "
                    + f"{truncate(str(last_err.get('ts_iso', '')), 28)} "
                    + f"{truncate(str(last_err.get('stage', '?')), 22)}.{truncate(str(last_err.get('reason', '?')), 28)}"
                )
        if callable(getattr(self, "_persona_queue_diag_snapshot", None)):
            pdiag = self._persona_queue_diag_snapshot()
            lines.append(
                f"- persona_queue_diag: events={pdiag.get('total_events', 0)} queued={pdiag.get('queued', 0)} "
                f"drops={pdiag.get('drops', 0)} errors={pdiag.get('errors', 0)} "
                f"applied={pdiag.get('worker_applied', 0)} deduped={pdiag.get('worker_deduped', 0)} "
                f"fallback={pdiag.get('fallback_profile_worker', 0)}"
            )
        if state.get("last_decay_at"):
            lines.append(f"- last_decay_at: {truncate(str(state.get('last_decay_at')), 50)}")
        if status.get("last_decay_status"):
            lines.append(f"- last_decay_status: {status.get('last_decay_status')}")
        overlay_summary = str(state.get("overlay_summary") or "").strip()
        if overlay_summary:
            lines.append(f"- overlay_summary: {truncate(overlay_summary, 180)}")
        if traits:
            lines.append("Traits:")
            for row in traits[:8]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    "  "
                    + f"{row.get('trait_key', '?')}={float(row.get('current_value', 0.0)):.2f} "
                    + f"(anchor={float(row.get('anchor_value', 0.0)):.2f}, "
                    + f"c={float(row.get('confidence', 0.0)):.2f}, {row.get('status', '?')})"
                )
        if reflections:
            lines.append("Recent reflections:")
            for row in reflections[:4]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    "  "
                    + f"#{row.get('reflection_id', '?')} {row.get('trigger_type', '?')} "
                    + f"{row.get('status', '?')} "
                    + f"({truncate(str(row.get('created_at', '')), 40)})"
                )
        return "\n".join(lines)

    def _format_relationship_admin_report(
        self,
        target_label: str,
        target_user_id: int,
        payload: dict[str, Any],
    ) -> str:
        status = payload.get("status", {}) if isinstance(payload.get("status"), dict) else {}
        if not payload.get("enabled"):
            return (
                "Persona Relationship (admin)\n"
                f"- target: {target_label} ({target_user_id})\n"
                f"- enabled: no\n"
                f"- configured: {'yes' if status.get('configured') else 'no'}\n"
                f"- backend_supported: {'yes' if status.get('backend_supported') else 'no'}"
            )

        pref = payload.get("memory_pref", {}) if isinstance(payload.get("memory_pref"), dict) else {}
        rel = payload.get("relationship", {}) if isinstance(payload.get("relationship"), dict) else {}
        lines = [
            "Persona Relationship (admin)",
            f"- target: {target_label} ({target_user_id})",
            f"- memory_mode: {pref.get('memory_mode', 'full')}",
            f"- episodic_callbacks: {'yes' if pref.get('allow_episodic_callbacks', True) else 'no'}",
            f"- personality_influence: {'yes' if pref.get('allow_personality_influence', True) else 'no'}",
        ]
        if not rel:
            lines.append("- relationship: none yet")
            return "\n".join(lines)

        lines.extend(
            [
                f"- status: {rel.get('status', '?')} (consent_scope={rel.get('consent_scope', '?')})",
                f"- familiarity={float(rel.get('familiarity', 0.0)):.2f} trust={float(rel.get('trust', 0.0)):.2f} warmth={float(rel.get('warmth', 0.0)):.2f}",
                f"- banter_license={float(rel.get('banter_license', 0.0)):.2f} support_sensitivity={float(rel.get('support_sensitivity', 0.0)):.2f}",
                f"- confidence={float(rel.get('confidence', 0.0)):.2f} influence_weight={float(rel.get('effective_influence_weight', 0.0)):.2f}",
                f"- interactions: total={int(rel.get('interaction_count', 0))} text={int(rel.get('text_turn_count', 0))} voice={int(rel.get('voice_turn_count', 0))}",
            ]
        )
        if rel.get("relationship_summary"):
            lines.append(f"- summary: {truncate(str(rel.get('relationship_summary')), 180)}")
        if rel.get("inside_joke_summary"):
            lines.append(f"- inside_jokes: {truncate(str(rel.get('inside_joke_summary')), 180)}")
        if rel.get("preferred_style_notes"):
            lines.append(f"- style_notes: {truncate(str(rel.get('preferred_style_notes')), 180)}")
        return "\n".join(lines)

    def _format_episodes_admin_report(
        self,
        payload: dict[str, Any],
        *,
        target_label: str = "",
        target_user_id: int | None = None,
    ) -> str:
        status = payload.get("status", {}) if isinstance(payload.get("status"), dict) else {}
        if not payload.get("enabled"):
            return (
                "Persona Episodes (admin)\n"
                f"- enabled: no\n"
                f"- configured: {'yes' if status.get('configured') else 'no'}\n"
                f"- backend_supported: {'yes' if status.get('backend_supported') else 'no'}"
            )
        episodes = payload.get("episodes", []) if isinstance(payload.get("episodes"), list) else []
        lines = ["Persona Episodes (admin)"]
        if target_user_id is not None:
            lines.append(f"- target: {target_label or f'user:{target_user_id}'} ({target_user_id})")
        if payload.get("filter_channel_id"):
            lines.append(f"- channel_id: {payload.get('filter_channel_id')}")
        lines.append(f"- count: {len(episodes)}")
        if not episodes:
            lines.append("- no episodes found")
            return "\n".join(lines)
        for row in episodes[:12]:
            if not isinstance(row, dict):
                continue
            lines.append(
                "  "
                + f"#{row.get('episode_id', '?')} {row.get('episode_type', '?')} "
                + f"{row.get('status', '?')} "
                + f"(imp={float(row.get('importance', 0.0)):.2f}, "
                + f"vid={float(row.get('vividness', 0.0)):.2f}, "
                + f"c={float(row.get('confidence', 0.0)):.2f}, "
                + f"p={row.get('privacy_level', '?')})"
            )
            title = str(row.get("title") or "").strip()
            if title:
                lines.append("    " + truncate(title, 160))
            callback_line = str(row.get("callback_line") or "").strip()
            if callback_line:
                lines.append("    callback: " + truncate(callback_line, 160))
            snippet = str(row.get("latest_snippet") or "").strip()
            if snippet:
                lines.append("    ev: " + truncate(snippet, 160))
        return "\n".join(lines)

    async def _set_persona_memory_mode_for_user(
        self,
        *,
        message: discord.Message,
        persona_engine: object,
        guild_key: str,
        target_user_id: int,
        mode: str,
    ) -> tuple[bool, str]:
        if not callable(getattr(self.memory, "set_persona_user_memory_pref", None)):
            return (False, "Persona memory preferences are not supported by current memory backend.")
        mode_clean = str(mode or "").strip().lower()
        if mode_clean not in {"full", "minimal", "none"}:
            return (False, "Mode must be one of: full, minimal, none.")
        allow_callbacks = mode_clean == "full"
        allow_influence = mode_clean == "full"
        await self.memory.set_persona_user_memory_pref(
            guild_id=guild_key,
            user_id=str(target_user_id),
            memory_mode=mode_clean,
            allow_episodic_callbacks=allow_callbacks,
            allow_personality_influence=allow_influence,
            allow_sensitive_storage=False,
            updated_by_admin_user_id=str(message.author.id),
        )
        audit_action = "forget_soft" if mode_clean == "none" else "set_memory_mode"
        if callable(getattr(self.memory, "append_persona_audit_log", None)) and hasattr(persona_engine, "persona_id"):
            with contextlib.suppress(Exception):
                await self.memory.append_persona_audit_log(
                    str(getattr(persona_engine, "persona_id", "liza")),
                    actor_type="admin",
                    actor_user_id=str(message.author.id),
                    action=audit_action,
                    entity_type="persona_user_memory_prefs",
                    entity_id=f"{guild_key}:{target_user_id}",
                    after_json={
                        "guild_id": guild_key,
                        "user_id": str(target_user_id),
                        "memory_mode": mode_clean,
                        "allow_episodic_callbacks": allow_callbacks,
                        "allow_personality_influence": allow_influence,
                    },
                    reason=f"persona memory mode set via Discord command by {message.author.id}",
                )
        if callable(getattr(persona_engine, "invalidate_prompt_cache", None)):
            persona_engine.invalidate_prompt_cache()
        return (True, mode_clean)

    def _format_reflect_admin_report(self, result: dict[str, Any]) -> str:
        status = str(result.get("status") or "unknown")
        lines = ["Persona Reflection (admin)", f"- status: {status}"]
        if result.get("reason"):
            lines.append(f"- reason: {truncate(str(result.get('reason')), 220)}")
        if result.get("reflection_id"):
            lines.append(f"- reflection_id: {result.get('reflection_id')}")
        proposer_source = str(result.get("proposer_source") or "").strip()
        if proposer_source:
            lines.append(f"- proposer: {proposer_source}")
        proposer_model = str(result.get("proposer_model") or "").strip()
        if proposer_model:
            lines.append(f"- proposer_model: {truncate(proposer_model, 120)}")
        proposer_prompt = str(result.get("proposer_prompt_version") or "").strip()
        if proposer_prompt:
            lines.append(f"- proposer_prompt: {truncate(proposer_prompt, 120)}")
        proposer_reason = str(result.get("proposer_reason") or "").strip()
        if proposer_reason:
            lines.append(f"- proposer_reason: {truncate(proposer_reason, 180)}")
        if isinstance(result.get("window"), dict):
            window = result["window"]
            lines.append(
                f"- window: msgs={window.get('ingested_count', '?')} users={window.get('unique_users', '?')} "
                f"range={window.get('after_message_id', '?')}..{window.get('max_message_id', '?')}"
            )
        if result.get("accepted_trait_candidates") is not None:
            lines.append(f"- accepted_trait_candidates: {result.get('accepted_trait_candidates')}")
        if result.get("accepted_contested_trait_candidates") is not None:
            lines.append(f"- accepted_contested_trait_candidates: {result.get('accepted_contested_trait_candidates')}")
        if result.get("accepted_episode_promotions") is not None:
            lines.append(f"- accepted_episode_promotions: {result.get('accepted_episode_promotions')}")
        if result.get("applied_trait_updates") is not None:
            lines.append(f"- applied_trait_updates: {result.get('applied_trait_updates')}")
        if result.get("applied_contested_trait_updates") is not None:
            lines.append(f"- applied_contested_trait_updates: {result.get('applied_contested_trait_updates')}")
        if result.get("applied_episode_promotions_count") is not None:
            lines.append(f"- applied_episode_promotions_count: {result.get('applied_episode_promotions_count')}")
        if result.get("overlay_summary_applied") is not None:
            lines.append(f"- overlay_summary_applied: {'yes' if result.get('overlay_summary_applied') else 'no'}")
        warnings = result.get("warnings", [])
        if isinstance(warnings, list) and warnings:
            lines.append("- warnings: " + ", ".join(truncate(str(x), 60) for x in warnings[:4]))
        overlay_summary = str(result.get("overlay_summary_candidate") or "").strip()
        if overlay_summary:
            lines.append(f"- overlay_summary_candidate: {truncate(overlay_summary, 220)}")
        if result.get("duration_ms") is not None:
            lines.append(f"- duration_ms: {result.get('duration_ms')}")
        return "\n".join(lines)

    def _format_decay_admin_report(self, result: dict[str, Any]) -> str:
        status = str(result.get("status") or "unknown")
        lines = ["Persona Decay (admin)", f"- status: {status}"]
        if result.get("reason"):
            lines.append(f"- reason: {truncate(str(result.get('reason')), 220)}")
        if result.get("minutes_remaining") is not None:
            lines.append(f"- minutes_remaining: {result.get('minutes_remaining')}")
        if result.get("trait_updates") is not None:
            lines.append(f"- trait_updates: {result.get('trait_updates')}")
        if result.get("relationship_updates") is not None:
            lines.append(f"- relationship_updates: {result.get('relationship_updates')}")
        if result.get("episode_updates") is not None:
            lines.append(f"- episode_updates: {result.get('episode_updates')}")
        if result.get("episode_archived") is not None:
            lines.append(f"- episode_archived: {result.get('episode_archived')}")
        trait_examples = result.get("trait_examples", [])
        if isinstance(trait_examples, list) and trait_examples:
            lines.append("- trait_examples: " + ", ".join(truncate(str(x), 80) for x in trait_examples[:3]))
        rel_examples = result.get("relationship_examples", [])
        if isinstance(rel_examples, list) and rel_examples:
            lines.append("- relationship_examples: " + ", ".join(truncate(str(x), 90) for x in rel_examples[:2]))
        ep_examples = result.get("episode_examples", [])
        if isinstance(ep_examples, list) and ep_examples:
            lines.append("- episode_examples: " + ", ".join(truncate(str(x), 90) for x in ep_examples[:2]))
        if result.get("duration_ms") is not None:
            lines.append(f"- duration_ms: {result.get('duration_ms')}")
        return "\n".join(lines)

    def _format_persona_trait_why_report(self, payload: dict[str, Any]) -> str:
        status = payload.get("status", {}) if isinstance(payload.get("status"), dict) else {}
        if not payload.get("enabled"):
            return (
                "Persona Trait Evidence (admin)\n"
                f"- enabled: no\n"
                f"- configured: {'yes' if status.get('configured') else 'no'}\n"
                f"- backend_supported: {'yes' if status.get('backend_supported') else 'no'}"
            )
        trait = payload.get("trait", {}) if isinstance(payload.get("trait"), dict) else {}
        requested = str(payload.get("trait_key_requested") or "").strip()
        if not trait:
            available = payload.get("available_trait_keys", [])
            hint = ", ".join(str(x) for x in available[:12]) if isinstance(available, list) else ""
            lines = ["Persona Trait Evidence (admin)", f"- requested: {requested or '(empty)'}", "- trait: not found"]
            if hint:
                lines.append(f"- available: {hint}")
            return "\n".join(lines)

        evidence = payload.get("evidence", []) if isinstance(payload.get("evidence"), list) else []
        lines = [
            "Persona Trait Evidence (admin)",
            f"- trait: {trait.get('trait_key', '?')} ({trait.get('label', '?')})",
            (
                f"- value={float(trait.get('current_value', 0.0)):.3f} "
                f"anchor={float(trait.get('anchor_value', 0.0)):.3f} "
                f"drift={float(trait.get('current_value', 0.0)) - float(trait.get('anchor_value', 0.0)):+.3f}"
            ),
            (
                f"- confidence={float(trait.get('confidence', 0.0)):.2f} "
                f"evidence_count={int(trait.get('evidence_count', 0))} "
                f"status={trait.get('status', '?')} "
                f"protected={trait.get('protected_mode', '?')}"
            ),
        ]
        support_score = float(trait.get("support_score", 0.0) or 0.0)
        contradiction_score = float(trait.get("contradiction_score", 0.0) or 0.0)
        contradiction_ratio = (contradiction_score / support_score) if support_score > 1e-9 else (999.0 if contradiction_score > 0 else 0.0)
        lines.append(
            f"- support={support_score:.3f} contradiction={contradiction_score:.3f} contradiction_ratio={contradiction_ratio:.2f}"
        )
        retrieval_policy = payload.get("retrieval_policy", {}) if isinstance(payload.get("retrieval_policy"), dict) else {}
        if retrieval_policy:
            lines.append(
                "- retrieval_policy: "
                + f"{retrieval_policy.get('policy', '?')} "
                + f"(reason={truncate(str(retrieval_policy.get('reason', '') or '-'), 120)}; "
                + f"prompt_exposure={retrieval_policy.get('prompt_exposure', '?')})"
            )
        notes = str(trait.get("notes") or "").strip()
        if notes:
            lines.append(f"- notes: {truncate(notes, 200)}")
        if not evidence:
            lines.append("- recent_evidence: none")
            return "\n".join(lines)
        support_rows = 0
        conflict_rows = 0
        for row in evidence:
            if not isinstance(row, dict):
                continue
            signal_kind = str(row.get("signal_kind") or "").strip().lower()
            if "conflict" in signal_kind:
                conflict_rows += 1
            else:
                support_rows += 1
        lines.append(f"- evidence_mix: support_like={support_rows} conflict_like={conflict_rows}")
        lines.append("Recent evidence:")
        for row in evidence[:8]:
            if not isinstance(row, dict):
                continue
            direction = float(row.get("direction", 0.0))
            sign = "+" if direction >= 0 else "-"
            signal_kind = str(row.get("signal_kind") or "").strip()
            lines.append(
                "  "
                + f"#{row.get('trait_evidence_id', '?')} {sign}{float(row.get('magnitude', 0.0)):.4f} "
                + f"[{signal_kind or 'signal'}] "
                + f"(c={float(row.get('signal_confidence', 0.0)):.2f}) "
                + f"{truncate(str(row.get('reason_text', '')), 120)}"
            )
            if row.get("reflection_id"):
                lines.append(
                    "    "
                    + f"reflection #{row.get('reflection_id')} {row.get('reflection_status', '?')} "
                    + f"{truncate(str(row.get('reflection_created_at', '')), 40)}"
                )
            refs = row.get("evidence_refs", [])
            if isinstance(refs, list) and refs:
                ref_bits: list[str] = []
                nested_candidate_evidence: dict[str, Any] | None = None
                for ref in refs[:6]:
                    if not isinstance(ref, dict):
                        continue
                    kind = str(ref.get("kind") or "").strip()
                    value = ref.get("value")
                    if not kind:
                        continue
                    if kind in {"message_id", "episode_id", "reflection_id"}:
                        ref_bits.append(f"{kind}={value}")
                    elif kind == "conflict_kind":
                        ref_bits.append(f"conflict={value}")
                    elif kind == "candidate_evidence" and isinstance(value, dict):
                        nested_candidate_evidence = value
                if isinstance(nested_candidate_evidence, dict):
                    for key in ("message_id", "episode_id", "reflection_id"):
                        if key in nested_candidate_evidence and nested_candidate_evidence.get(key) not in (None, ""):
                            ref_bits.append(f"{key}={nested_candidate_evidence.get(key)}")
                    for key in ("message_ids", "episode_ids"):
                        values = nested_candidate_evidence.get(key)
                        if isinstance(values, list) and values:
                            compact = ",".join(str(v) for v in values[:4] if v not in (None, ""))
                            if compact:
                                ref_bits.append(f"{key}=[{compact}]")
                if ref_bits:
                    lines.append("    " + "refs: " + ", ".join(ref_bits))
        return "\n".join(lines)

    async def _try_handle_persona_admin_command(
        self,
        message: discord.Message,
        command_text: str,
    ) -> bool:
        if not command_text:
            return False
        lowered = command_text.casefold()
        if not (
            lowered == "persona"
            or lowered.startswith("persona ")
            or lowered == "relationship"
            or lowered.startswith("relationship ")
            or lowered == "episodes"
            or lowered.startswith("episodes ")
            or lowered == "reflect"
            or lowered.startswith("reflect ")
            or lowered == "decay"
            or lowered.startswith("decay ")
            or lowered == "forget"
            or lowered.startswith("forget ")
        ):
            return False

        if not self._is_persona_admin_authorized(message):
            await message.reply("Persona admin command is disabled or you are not authorized.")
            return True

        persona_engine = getattr(self, "persona_engine", None)
        if persona_engine is None:
            await message.reply("Persona engine is not available.")
            return True

        parts = command_text.split()
        root = parts[0].casefold()
        args = parts[1:]
        guild_key = str(message.guild.id) if message.guild else "dm"

        if root == "persona":
            if args and args[0].casefold() == "why":
                if len(args) < 2:
                    await message.reply("Usage: !persona why <trait_key>")
                    return True
                trait_key = " ".join(args[1:]).strip()
                payload = await persona_engine.admin_trait_why_snapshot(trait_key, limit=8)
                text = self._format_persona_trait_why_report(payload)
                await self._send_chunks(message.channel, text, reference=message)
                return True
            payload = await persona_engine.admin_persona_snapshot()
            await self._send_chunks(message.channel, self._format_persona_admin_report(payload), reference=message)
            return True

        if root == "reflect":
            if args and args[0].casefold() == "status":
                payload = await persona_engine.admin_persona_snapshot()
                text = self._format_persona_admin_report(payload)
                await self._send_chunks(message.channel, text, reference=message)
                return True
            if args and args[0].casefold() == "apply":
                reflection_id: int | None = None
                if len(args) >= 2 and str(args[1]).strip().isdigit():
                    with contextlib.suppress(ValueError):
                        reflection_id = int(str(args[1]).strip())
                result = await persona_engine.apply_reflection(
                    reflection_id=reflection_id,
                    actor_user_id=str(message.author.id),
                )
                await self._send_chunks(message.channel, self._format_reflect_admin_report(result), reference=message)
                return True
            force = True
            reason = "manual_admin_command"
            trigger_type = "manual"
            if args and args[0].casefold() == "scheduled":
                force = False
                reason = "manual_scheduled_check"
                trigger_type = "scheduled"
            result = await persona_engine.run_reflection_once(
                trigger_type=trigger_type,
                trigger_reason=reason,
                force=force,
                dry_run=True,
            )
            await self._send_chunks(message.channel, self._format_reflect_admin_report(result), reference=message)
            return True

        if root == "decay":
            if args and args[0].casefold() == "status":
                payload = await persona_engine.admin_persona_snapshot()
                text = self._format_persona_admin_report(payload)
                await self._send_chunks(message.channel, text, reference=message)
                return True
            force = True
            reason = "manual_admin_command"
            trigger_type = "manual"
            if args and args[0].casefold() == "scheduled":
                force = False
                reason = "manual_scheduled_check"
                trigger_type = "scheduled"
            result = await persona_engine.run_decay_once(
                trigger_type=trigger_type,
                trigger_reason=reason,
                force=force,
                actor_user_id=str(message.author.id),
            )
            await self._send_chunks(message.channel, self._format_decay_admin_report(result), reference=message)
            return True

        if root == "relationship":
            if args and args[0].casefold() in {"mode", "privacy"}:
                if len(args) < 2:
                    await message.reply("Usage: !relationship mode @user <full|minimal|none>")
                    return True
                mode_value = args[-1].casefold()
                target = self._resolve_admin_target_user(message, args[1:-1])
                if target is None:
                    await message.reply("Could not resolve target user.")
                    return True
                target_label, target_user_id = target
                ok, result = await self._set_persona_memory_mode_for_user(
                    message=message,
                    persona_engine=persona_engine,
                    guild_key=guild_key,
                    target_user_id=target_user_id,
                    mode=mode_value,
                )
                if not ok:
                    await message.reply(result)
                    return True
                await message.reply(
                    f"Set persona memory mode for `{target_label}` ({target_user_id}) to `{result}`."
                )
                return True

            target = self._resolve_admin_target_user(message, args)
            if target is None:
                await message.reply("Could not resolve target user.")
                return True
            target_label, target_user_id = target
            payload = await persona_engine.admin_relationship_snapshot(guild_key, str(target_user_id))
            text = self._format_relationship_admin_report(target_label, target_user_id, payload)
            await self._send_chunks(message.channel, text, reference=message)
            return True

        if root == "episodes":
            target_label = ""
            target_user_id: int | None = None
            target_user_key: str | None = None
            limit = 10
            if args:
                if args[-1].isdigit():
                    with contextlib.suppress(ValueError):
                        limit = max(1, min(int(args[-1]), 20))
                    args = args[:-1]
                if args and args[0].casefold() not in {"recent", "all"}:
                    target = self._resolve_admin_target_user(message, args)
                    if target is not None:
                        target_label, target_user_id = target
                        target_user_key = str(target_user_id)
            payload = await persona_engine.admin_episodes_snapshot(
                guild_key,
                user_id=target_user_key,
                channel_id=None,
                limit=limit,
            )
            text = self._format_episodes_admin_report(
                payload,
                target_label=target_label,
                target_user_id=target_user_id,
            )
            await self._send_chunks(message.channel, text, reference=message)
            return True

        if root == "forget":
            target = self._resolve_admin_target_user(message, args)
            if target is None:
                await message.reply("Could not resolve target user.")
                return True
            target_label, target_user_id = target
            ok, result = await self._set_persona_memory_mode_for_user(
                message=message,
                persona_engine=persona_engine,
                guild_key=guild_key,
                target_user_id=target_user_id,
                mode="none",
            )
            if not ok:
                await message.reply(result)
                return True
            await message.reply(
                f"Applied soft forget for `{target_label}` ({target_user_id}): persona memory mode set to `none`."
            )
            return True

        return False

    async def _try_handle_system_command(self, message: discord.Message) -> bool:
        raw = collapse_spaces(message.content)
        if not raw:
            return False
        prefix = self.settings.command_prefix.strip()
        if not prefix or not raw.startswith(prefix):
            return False

        command_text = raw[len(prefix) :].strip()
        command = command_text.lower()
        join_label = f"{prefix}join"
        status_label = f"{prefix}status"

        if command == "voice diag" or command.startswith("voice diag "):
            if not self._is_persona_admin_authorized(message):
                await message.reply("Voice diagnostics command is disabled or you are not authorized.")
                return True
            include_recent = "norecent" not in command.split()
            text = self._format_voice_diag_admin_report(include_recent=include_recent)
            await self._send_chunks(message.channel, text, reference=message)
            return True

        if await self._try_handle_persona_admin_command(message, command_text):
            return True

        if command == "status":
            try:
                snapshot = await self.build_status_snapshot()
                await self._send_chunks(message.channel, self._format_status_report(snapshot), reference=message)
            except Exception as exc:
                logger.exception("Status command failed: %s", exc)
                await message.reply(f"`{status_label}` failed. Check bot logs.")
            return True

        if command != "join":
            return False

        if message.guild is None or not isinstance(message.author, discord.Member):
            await message.reply(f"`{join_label}` works only in a server.")
            return True

        if not self.settings.voice_enabled:
            await message.reply("Voice mode is disabled in config.")
            return True

        voice_state = message.author.voice
        if voice_state is None or voice_state.channel is None:
            await message.reply(f"Join a voice channel and run `{join_label}` again.")
            return True

        target_name = voice_state.channel.name
        try:
            ok = await self._ensure_voice_capture(message.guild, message.author, message.channel.id)
            if ok:
                await message.reply(f"Connected to `{target_name}`.")
            else:
                await message.reply("Could not start voice capture. Check bot logs.")
        except asyncio.TimeoutError:
            logger.warning("Join command timed out for guild=%s channel=%s", message.guild.id, voice_state.channel.id)
            await message.reply("Voice connection timed out. Try again in 2-3 seconds.")
        except Exception as exc:
            logger.exception("Join command failed: %s", exc)
            await message.reply("Could not connect to voice channel.")
        return True

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        if await self._try_handle_system_command(message):
            return
        if not self._should_auto_reply(message):
            return

        user_text = self._strip_bot_mention(message.content)
        if not user_text:
            return

        guild_id = message.guild.id if message.guild else None
        channel_id = message.channel.id
        user_label = getattr(message.author, "display_name", message.author.name)

        if message.guild and isinstance(message.author, discord.Member):
            with contextlib.suppress(Exception):
                user_label = await self._sync_member_identity(message.guild.id, message.author)

        if (
            self.settings.voice_enabled
            and self.settings.voice_auto_join_on_mention
            and message.guild
            and isinstance(message.author, discord.Member)
            and self.user
            and self.user.mentioned_in(message)
        ):
            with contextlib.suppress(Exception):
                started = await self._ensure_voice_capture(message.guild, message.author, message.channel.id)
                if not started:
                    logger.warning(
                        "Voice auto-capture was requested but did not start in guild=%s",
                        message.guild.id,
                    )

        try:
            async with message.channel.typing():
                reply = await self._run_dialogue_turn(
                    guild_id=guild_id,
                    channel_id=channel_id,
                    user_id=message.author.id,
                    user_label=user_label,
                    user_text=user_text,
                    modality="text",
                    source="discord_text",
                    quality=1.0,
                )
            await self._send_chunks(message.channel, reply, reference=message)
        except Exception as exc:
            logger.exception("Text turn failed: %s", exc)
            await message.reply("I failed to answer right now.")

    async def on_native_audio_user_transcript(
        self,
        guild_id: int,
        user_id: int,
        text: str,
        source: str = "gemini_native_input",
        confidence: float = 0.96,
    ) -> None:
        if self.user is not None and user_id == self.user.id:
            self._record_voice_memory_diag(
                stage="native_user_callback",
                outcome="drop",
                reason="assistant_user_id",
                guild_id=guild_id,
                user_id=user_id,
                source=source,
            )
            return
        channel_id = self.voice_text_channels.get(guild_id)
        if channel_id is None:
            self._record_voice_memory_diag(
                stage="native_user_callback",
                outcome="drop",
                reason="no_voice_text_channel_binding",
                guild_id=guild_id,
                user_id=user_id,
                source=source,
            )
            return
        cleaned = collapse_spaces(text)
        if not cleaned:
            self._record_voice_memory_diag(
                stage="native_user_callback",
                outcome="drop",
                reason="empty_transcript",
                guild_id=guild_id,
                channel_id=channel_id,
                user_id=user_id,
                source=source,
            )
            return
        if self._is_recent_assistant_echo(guild_id, cleaned):
            self._record_voice_memory_diag(
                stage="native_user_callback",
                outcome="drop",
                reason="echo_filter",
                guild_id=guild_id,
                channel_id=channel_id,
                user_id=user_id,
                source=source,
            )
            return
        if self._is_duplicate_native_user_transcript(guild_id, user_id, cleaned):
            self._record_voice_memory_diag(
                stage="native_user_callback",
                outcome="drop",
                reason="duplicate_transcript",
                guild_id=guild_id,
                channel_id=channel_id,
                user_id=user_id,
                source=source,
            )
            return

        user_label = f"user:{user_id}"
        guild = self.get_guild(guild_id)
        if guild is not None:
            member = guild.get_member(user_id)
            if member is not None:
                with contextlib.suppress(Exception):
                    user_label = await self._sync_member_identity(guild_id, member)

        try:
            await self._save_native_user_transcript(
                guild_id=guild_id,
                channel_id=channel_id,
                user_id=user_id,
                user_label=user_label,
                text=cleaned,
                source=source,
                quality=confidence,
            )
        except Exception as exc:
            logger.exception(
                "Native audio user transcript callback failed guild=%s channel=%s user=%s source=%s",
                guild_id,
                channel_id,
                user_id,
                source,
            )
            self._record_voice_memory_diag(
                stage="native_user_callback",
                outcome="error",
                reason="save_native_user_transcript_failed",
                guild_id=guild_id,
                channel_id=channel_id,
                user_id=user_id,
                source=source,
                error=exc,
            )
            raise

    async def on_native_audio_assistant_transcript(
        self,
        guild_id: int,
        text: str,
        source: str = "gemini_native_output",
    ) -> None:
        channel_id = self.voice_text_channels.get(guild_id)
        if channel_id is None:
            self._record_voice_memory_diag(
                stage="native_assistant_callback",
                outcome="drop",
                reason="no_voice_text_channel_binding",
                guild_id=guild_id,
                source=source,
            )
            return
        cleaned = collapse_spaces(text)
        if not cleaned:
            self._record_voice_memory_diag(
                stage="native_assistant_callback",
                outcome="drop",
                reason="empty_transcript",
                guild_id=guild_id,
                channel_id=channel_id,
                source=source,
            )
            return
        if self._is_duplicate_native_assistant_transcript(guild_id, cleaned):
            self._record_voice_memory_diag(
                stage="native_assistant_callback",
                outcome="drop",
                reason="duplicate_transcript",
                guild_id=guild_id,
                channel_id=channel_id,
                source=source,
            )
            return
        try:
            await self._save_native_assistant_transcript(
                guild_id=guild_id,
                channel_id=channel_id,
                text=cleaned,
                source=source,
            )
        except Exception as exc:
            logger.exception(
                "Native audio assistant transcript callback failed guild=%s channel=%s source=%s",
                guild_id,
                channel_id,
                source,
            )
            self._record_voice_memory_diag(
                stage="native_assistant_callback",
                outcome="error",
                reason="save_native_assistant_transcript_failed",
                guild_id=guild_id,
                channel_id=channel_id,
                source=source,
                error=exc,
            )
            raise

    async def _run_dialogue_turn(
        self,
        guild_id: int | None,
        channel_id: int,
        user_id: int,
        user_label: str,
        user_text: str,
        modality: str,
        source: str,
        quality: float,
    ) -> str:
        guild_key = str(guild_id) if guild_id is not None else "dm"
        channel_key = str(channel_id)
        user_key = str(user_id)

        lock = self.channel_locks[channel_key]
        async with lock:
            role_id = await self._resolve_role(guild_key)
            session_mode = "voice" if modality == "voice" else "text"
            session_id = await self.memory.get_or_create_session(
                guild_id=guild_key,
                channel_id=channel_key,
                mode=session_mode,
                role_id=role_id,
            )

            clean_user_text = collapse_spaces(user_text)
            self._update_session_state_from_user(channel_key, clean_user_text, user_label=user_label, modality=modality)
            user_message_id = await self.memory.save_message(
                session_id=session_id,
                guild_id=guild_key,
                channel_id=channel_key,
                user_id=user_key,
                author_label=user_label,
                role="user",
                modality=modality,
                content_raw=user_text,
                content_clean=clean_user_text,
                source=source,
                quality=quality,
            )
            logger.info(
                "[msg.user] channel=%s user=%s source=%s text=\"%s\"",
                channel_key,
                user_label,
                source,
                truncate(clean_user_text, 120),
            )

            self._enqueue_profile_update(
                guild_id=guild_key,
                channel_id=channel_key,
                user_id=user_key,
                message_id=user_message_id,
                user_text=clean_user_text,
                user_label=user_label,
                modality=modality,
                source=source,
                quality=quality,
            )
            self._enqueue_summary_update(
                guild_key,
                channel_key,
                user_key,
                modality=modality,
                source=source,
                quality=quality,
            )

            llm_messages = await self._build_context_packet(
                guild_id=guild_key,
                channel_id=channel_key,
                user_id=user_key,
                session_id=session_id,
                role_id=role_id,
                user_text=clean_user_text,
            )

            # --- PLUGIN INTERCEPTION (Fast NLU Fallback) ---
            reply = None
            plugin_used = False
            if hasattr(self, "plugin_manager") and getattr(self, "plugin_manager") is not None:
                plugin, score = self.plugin_manager.find_best_match(clean_user_text)
                if plugin is not None:
                    # Provide context that plugins might need
                    ctx = {
                        "guild_id": guild_key,
                        "channel_id": channel_key,
                        "user_id": user_key,
                        "session_id": session_id,
                        "modality": modality,
                    }
                    try:
                        reply = await plugin.execute(message=None, raw_text=user_text, context=ctx)
                        plugin_used = reply is not None
                    except Exception:
                        plugin_used = True
                        logger.exception("Plugin %s failed to execute", plugin.name)
                        reply = "Ой, сталася помилка при виконанні внутрішньої команди."

            # --- FALLBACK TO LLM ---
            if reply is None:
                reply = await self.llm.chat(llm_messages)

            reply = self._shape_human_reply(channel_key, clean_user_text, reply, modality)
            if self.settings.max_response_chars > 0 and len(reply) > self.settings.max_response_chars:
                reply = truncate(reply, self.settings.max_response_chars)
            self._register_bot_reply_state(channel_key, reply)

            bot_user_id = str(self.user.id if self.user else 0)
            bot_message_id = await self.memory.save_message(
                session_id=session_id,
                guild_id=guild_key,
                channel_id=channel_key,
                user_id=bot_user_id,
                author_label="assistant",
                role="assistant",
                modality="text",
                content_raw=reply,
                content_clean=reply,
                source="plugin" if plugin_used else "gemini",
                quality=1.0,
            )
            if bool(getattr(self.settings, "memory_persona_self_facts_enabled", True)):
                self._enqueue_profile_update(
                    guild_id=guild_key,
                    channel_id=channel_key,
                    user_id=bot_user_id,
                    message_id=int(bot_message_id),
                    user_text=reply,
                    user_label="assistant",
                    modality="text",
                    source="plugin" if plugin_used else "gemini",
                    quality=1.0,
                    speaker_role="assistant",
                    fact_owner_kind="persona",
                    fact_owner_id=str(getattr(self.settings, "persona_id", "") or "persona"),
                )
            logger.info("[msg.bot] channel=%s text=\"%s\"", channel_key, truncate(reply, 120))
            return reply
