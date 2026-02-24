from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..common import PendingProfileUpdate, PendingSummaryUpdate, as_int, collapse_spaces, truncate
from ...prompts.memory import build_summary_update_system_prompt, build_summary_update_user_prompt

logger = logging.getLogger("live_role_bot")


class WorkersMixin:
    def _enqueue_profile_update(
        self,
        guild_id: str,
        channel_id: str,
        user_id: str,
        message_id: int,
        user_text: str,
    ) -> None:
        if not self.settings.memory_enabled:
            return
        item = PendingProfileUpdate(
            guild_id=guild_id,
            channel_id=channel_id,
            user_id=user_id,
            message_id=message_id,
            user_text=user_text,
        )

        if self.profile_queue.full():
            try:
                self.profile_queue.get_nowait()
                self.profile_queue.task_done()
            except asyncio.QueueEmpty:
                pass

        try:
            self.profile_queue.put_nowait(item)
        except asyncio.QueueFull:
            pass

    async def _profile_worker(self) -> None:
        while True:
            item = await self.profile_queue.get()
            try:
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
                    await self.memory.upsert_user_fact(
                        guild_id=item.guild_id,
                        user_id=item.user_id,
                        fact_key=fact.key,
                        fact_value=fact.value,
                        fact_type=fact.fact_type,
                        confidence=fact.confidence,
                        importance=fact.importance,
                        message_id=item.message_id,
                        extractor="gemini_profile_extractor",
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

    def _enqueue_summary_update(self, guild_id: str, channel_id: str, user_id: str) -> None:
        if not self.settings.summary_enabled:
            return
        key = (guild_id, channel_id, user_id)
        if key in self.summary_pending_keys:
            return
        self.summary_pending_keys.add(key)

        item = PendingSummaryUpdate(guild_id=guild_id, channel_id=channel_id, user_id=user_id)
        if self.summary_queue.full():
            try:
                dropped = self.summary_queue.get_nowait()
                self.summary_queue.task_done()
                self.summary_pending_keys.discard((dropped.guild_id, dropped.channel_id, dropped.user_id))
            except asyncio.QueueEmpty:
                pass

        try:
            self.summary_queue.put_nowait(item)
        except asyncio.QueueFull:
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
