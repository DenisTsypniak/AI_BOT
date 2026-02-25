from __future__ import annotations

import contextlib
from typing import Any

import discord

from ..common import as_float, as_int, tokenize, truncate
from ...prompts.dialogue import (
    KNOWN_PARTICIPANT_MEMORY_HEADER,
    TEXT_CONVERSATION_BEHAVIOR,
    TEXT_DEBATE_BEHAVIOR,
    VOICE_SESSION_BEHAVIOR_LINES,
    build_language_rule,
    build_known_participant_fact_line,
    build_relevant_facts_section,
    build_rp_canon_section,
    build_role_profile_lines,
    build_user_dialogue_summary_line,
)


class PromptMixin:
    def _build_session_state_block(self, channel_id: str) -> str:
        state = self.conversation_states.get(channel_id)
        if state is None:
            return ""

        lines = [
            "Hidden session style state (guidance only, do not reveal):",
            f"- mood: {state.mood}",
            f"- energy: {state.energy}",
            f"- tease_level: {state.tease_level}",
        ]
        if state.topic_hint:
            lines.append(f"- current_topic_hint: {state.topic_hint}")
        if state.open_loop:
            lines.append(f"- unresolved_user_thread: {state.open_loop}")
        if state.last_user_text:
            lines.append(f"- latest_user_tone_sample: {truncate(state.last_user_text, 90)}")
        return "\n".join(lines)

    def _build_human_turn_policy_block(self, channel_id: str) -> str:
        lines = [
            "Human-like turn policy:",
            "- Default: short, reactive, natural (1-3 sentences unless user asks for details).",
            "- Usually ask at most one question.",
            "- Start with varied wording; avoid repeating the same opener every turn.",
            "- Sound like you listened: react to the user's emotional tone before giving advice/info.",
            "- Prefer one useful next step over long explanations.",
        ]
        recent = [item for item in self.recent_assistant_replies.get(channel_id, []) if item.strip()]
        if recent:
            lines.append("Recent assistant wording to avoid repeating too closely:")
            for item in recent[-3:]:
                lines.append(f"- {truncate(item, 110)}")
        return "\n".join(lines)

    async def _build_native_audio_system_prompt(
        self,
        guild: discord.Guild,
        voice_channel: discord.abc.Connectable,
    ) -> str:
        guild_key = str(guild.id)
        role_id = await self._resolve_role(guild_key)
        role_profile = await self.memory.get_role_profile(role_id)
        if role_profile is None:
            role_profile = {
                "name": self.settings.role_name,
                "goal": self.settings.role_goal,
                "style": self.settings.role_style,
                "constraints": self.settings.role_constraints,
            }

        lines = [
            self.settings.system_core_prompt,
            build_rp_canon_section(self.rp_canon_prompt),
            *build_role_profile_lines(role_profile),
            *VOICE_SESSION_BEHAVIOR_LINES,
        ]

        members = getattr(voice_channel, "members", [])
        if isinstance(members, list):
            profile_lines: list[str] = []
            added = 0
            for member in members:
                if not isinstance(member, discord.Member) or member.bot:
                    continue
                label = member.display_name
                with contextlib.suppress(Exception):
                    label = await self._sync_member_identity(guild.id, member)
                facts = await self.memory.get_user_facts(guild_key, str(member.id), limit=3)
                fact_text = "; ".join(
                    str(item.get("fact_value") or "").strip()
                    for item in facts
                    if item.get("fact_value")
                )
                if fact_text:
                    profile_lines.append(build_known_participant_fact_line(label, fact_text))
                    added += 1
                if added >= 6:
                    break
            if profile_lines:
                lines.append(KNOWN_PARTICIPANT_MEMORY_HEADER)
                lines.extend(profile_lines)

        return "\n".join(line for line in lines if line)

    async def _build_context_packet(
        self,
        guild_id: str,
        channel_id: str,
        user_id: str,
        session_id: int,
        role_id: str,
        user_text: str,
    ) -> list[dict[str, str]]:
        role_profile = await self.memory.get_role_profile(role_id)
        if role_profile is None:
            role_profile = {
                "role_id": self.settings.default_role_id,
                "name": self.settings.role_name,
                "goal": self.settings.role_goal,
                "style": self.settings.role_style,
                "constraints": self.settings.role_constraints,
            }

        summary_row = await self.memory.get_dialogue_summary(guild_id, user_id, channel_id)
        summary_text = ""
        if self.settings.summary_enabled and summary_row is not None:
            summary_text = str(summary_row.get("summary_text") or "").strip()

        facts = await self._select_relevant_facts(guild_id, user_id, user_text)
        history = await self.memory.get_recent_session_messages(session_id, self.settings.max_recent_messages)

        system_parts = [
            self.settings.system_core_prompt,
            build_rp_canon_section(self.rp_canon_prompt),
            *build_role_profile_lines(role_profile),
            TEXT_CONVERSATION_BEHAVIOR,
            TEXT_DEBATE_BEHAVIOR,
            build_language_rule(self.settings.preferred_response_language),
            self._build_human_turn_policy_block(channel_id),
            self._build_session_state_block(channel_id),
        ]

        if summary_text:
            system_parts.append(build_user_dialogue_summary_line(summary_text))
        if facts:
            system_parts.append(build_relevant_facts_section(facts))

        messages: list[dict[str, str]] = [
            {"role": "system", "content": "\n\n".join(part for part in system_parts if part)}
        ]

        for row in history:
            role = str(row.get("role", "")).strip().lower()
            content = str(row.get("content", "")).strip()
            if not content:
                continue
            if role == "assistant":
                messages.append({"role": "assistant", "content": content})
                continue
            author_label = str(row.get("author_label", "user")).strip() or "user"
            messages.append({"role": "user", "content": f"{author_label}: {content}"})

        return messages

    async def _select_relevant_facts(self, guild_id: str, user_id: str, query_text: str) -> list[dict[str, Any]]:
        if not self.settings.memory_enabled:
            return []

        raw_facts = await self.memory.get_user_facts(guild_id, user_id, limit=48)
        if not raw_facts:
            return []

        query_tokens = tokenize(query_text)
        scored: list[tuple[float, dict[str, Any]]] = []

        for index, fact in enumerate(raw_facts):
            fact_text = str(fact.get("fact_value") or "").strip()
            if not fact_text:
                continue
            confidence = as_float(fact.get("confidence"), 0.0)
            importance = as_float(fact.get("importance"), 0.0)
            evidence = max(1, as_int(fact.get("evidence_count"), 1))
            status = str(fact.get("status") or "candidate")
            pinned = bool(fact.get("pinned"))

            base = confidence * 1.2 + importance * 1.15 + min(6, evidence) * 0.08
            if status == "confirmed":
                base += 0.35
            if pinned:
                base += 1.0

            fact_tokens = tokenize(fact_text)
            overlap = len(query_tokens.intersection(fact_tokens)) if query_tokens else 0
            if query_tokens:
                base += (overlap / max(1.0, float(len(query_tokens)))) * 1.9
                if overlap > 0:
                    base += 0.22

            base -= index * 0.01
            scored.append((base, fact))

        scored.sort(key=lambda row: row[0], reverse=True)

        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for _, fact in scored:
            text = str(fact.get("fact_value") or "").strip()
            key = text.casefold()
            if not text or key in seen:
                continue
            seen.add(key)
            unique.append(fact)
            if len(unique) >= self.settings.memory_fact_top_k:
                break

        return unique
