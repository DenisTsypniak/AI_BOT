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
    def _merge_memory_fact_lists(
        self,
        primary: list[dict[str, Any]],
        fallback: list[dict[str, Any]],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for fact in [*(primary or []), *(fallback or [])]:
            if not isinstance(fact, dict):
                continue
            fact_key = str(fact.get("fact_key") or "").strip().casefold()
            fact_value = str(fact.get("fact_value") or "").strip()
            dedupe_key = fact_key or fact_value.casefold()
            if not dedupe_key or dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            merged.append(fact)
            if len(merged) >= max(1, int(limit)):
                break
        return merged

    @staticmethod
    def _has_identity_memory_fact(facts: list[dict[str, Any]]) -> bool:
        for fact in facts or []:
            if not isinstance(fact, dict):
                continue
            fact_key = str(fact.get("fact_key") or "").strip().casefold()
            fact_type = str(fact.get("fact_type") or "").strip().casefold()
            if fact_type == "identity" or fact_key.startswith("identity:"):
                return True
        return False

    async def _get_global_identity_fallback_fact(
        self,
        guild_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        getter = getattr(self.memory, "get_latest_user_identity_by_user_id", None)
        if not callable(getter):
            return None
        identity = None
        with contextlib.suppress(Exception):
            identity = await getter(user_id, exclude_guild_id=guild_id)
        if not identity:
            return None
        primary_name = str(identity.get("discord_global_name") or "").strip() or str(
            identity.get("discord_username") or ""
        ).strip()
        username = str(identity.get("discord_username") or "").strip()
        if not (primary_name or username):
            return None
        value = primary_name or username
        return {
            "fact_key": "identity:discord_primary_name",
            "fact_value": value,
            "fact_type": "identity",
            "confidence": 0.98,
            "importance": 0.9,
            "status": "confirmed",
            "pinned": False,
            "evidence_count": 1,
        }

    async def _get_summary_with_global_fallback(
        self,
        guild_id: str,
        user_id: str,
        channel_id: str,
    ) -> dict[str, Any] | None:
        summary_row = await self.memory.get_dialogue_summary(guild_id, user_id, channel_id)
        if summary_row is not None and str(summary_row.get("summary_text") or "").strip():
            return summary_row
        getter = getattr(self.memory, "get_latest_dialogue_summary_by_user_id", None)
        if not callable(getter):
            return summary_row
        with contextlib.suppress(Exception):
            global_row = await getter(user_id, exclude_guild_id=guild_id)
            if global_row is not None and str(global_row.get("summary_text") or "").strip():
                return global_row
        return summary_row

    async def _get_user_facts_with_global_fallback(
        self,
        guild_id: str,
        user_id: str,
        *,
        local_limit: int,
        target_count: int,
    ) -> list[dict[str, Any]]:
        local_facts = await self.memory.get_user_facts(guild_id, user_id, limit=max(1, int(local_limit)))
        getter = getattr(self.memory, "get_user_facts_global_by_user_id", None)
        if not callable(getter):
            merged = list(local_facts)
            if not self._has_identity_memory_fact(merged):
                fallback_identity = await self._get_global_identity_fallback_fact(guild_id, user_id)
                if fallback_identity:
                    merged = self._merge_memory_fact_lists(
                        [fallback_identity],
                        merged,
                        limit=max(local_limit, target_count),
                    )
            return merged
        global_facts: list[dict[str, Any]] = []
        with contextlib.suppress(Exception):
            global_facts = await getter(
                user_id,
                limit=max(1, int(max(local_limit, target_count))),
                exclude_guild_id=guild_id,
            )
        merged = self._merge_memory_fact_lists(
            local_facts,
            global_facts,
            limit=max(local_limit, target_count),
        )
        if not self._has_identity_memory_fact(merged):
            fallback_identity = await self._get_global_identity_fallback_fact(guild_id, user_id)
            if fallback_identity:
                merged = self._merge_memory_fact_lists([fallback_identity], merged, limit=max(local_limit, target_count))
        return merged

    def _build_session_state_block(self, channel_id: str) -> str:
        state = self.conversation_states.get(channel_id)
        if state is None:
            return ""

        lines = [
            "Hidden session style state (guidance only, do not reveal):",
            f"- mood: {state.mood}",
            f"- energy: {state.energy}",
            f"- tease_level: {state.tease_level}",
            f"- familiarity: {state.familiarity}",
            f"- group_vibe: {state.group_vibe}",
            f"- turn_count: {state.turn_count}",
            f"- last_modality: {state.last_modality}",
        ]
        if state.topic_hint:
            lines.append(f"- current_topic_hint: {state.topic_hint}")
        if state.open_loop:
            lines.append(f"- unresolved_user_thread: {state.open_loop}")
        if state.last_user_text:
            lines.append(f"- latest_user_tone_sample: {truncate(state.last_user_text, 90)}")
        if state.recent_speakers:
            lines.append(f"- recent_speakers: {', '.join(state.recent_speakers[-4:])}")
        if state.recent_topics:
            lines.append(f"- recent_topics: {', '.join(state.recent_topics[-4:])}")
        if state.callback_moments:
            lines.append("- callback_moments_you_may_reference_naturally:")
            for item in state.callback_moments[-3:]:
                lines.append(f"  - {truncate(item, 100)}")
        return "\n".join(lines)

    def _build_human_turn_policy_block(self, channel_id: str) -> str:
        lines = [
            "Human-like turn policy:",
            "- Default: short, reactive, natural (1-3 sentences unless user asks for details).",
            "- Usually ask at most one question.",
            "- Start with varied wording; avoid repeating the same opener every turn.",
            "- Sound like you listened: react to the user's emotional tone before giving advice/info.",
            "- Prefer one useful next step over long explanations.",
            "- If fitting, briefly callback to one recent session moment/joke/topic (do not force it).",
            "- In group voice vibe, you may address the group casually, but do not overdo it.",
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

        use_rp_only_character = bool(self.rp_canon_prompt.strip())
        lines = [
            build_rp_canon_section(self.rp_canon_prompt),
            *VOICE_SESSION_BEHAVIOR_LINES,
        ]
        if not use_rp_only_character:
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
                facts = await self._get_user_facts_with_global_fallback(
                    guild_key,
                    str(member.id),
                    local_limit=3,
                    target_count=3,
                )
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

        persona_engine = getattr(self, "persona_engine", None)
        if persona_engine is not None:
            with contextlib.suppress(Exception):
                overlay = await persona_engine.build_prompt_overlay(
                    mode="voice",
                    guild_id=guild_key,
                    channel_id=str(getattr(voice_channel, "id", "voice")),
                    user_id="*",
                    query_text="",
                )
                if overlay:
                    lines.append(overlay)

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

        summary_row = await self._get_summary_with_global_fallback(guild_id, user_id, channel_id)
        summary_text = ""
        if self.settings.summary_enabled and summary_row is not None:
            summary_text = str(summary_row.get("summary_text") or "").strip()

        facts = await self._select_relevant_facts(guild_id, user_id, user_text)
        history = await self.memory.get_recent_session_messages(session_id, self.settings.max_recent_messages)

        use_rp_only_character = bool(self.rp_canon_prompt.strip())
        system_parts = [
            build_rp_canon_section(self.rp_canon_prompt),
            TEXT_CONVERSATION_BEHAVIOR,
            TEXT_DEBATE_BEHAVIOR,
            build_language_rule(self.settings.preferred_response_language),
            self._build_human_turn_policy_block(channel_id),
            self._build_session_state_block(channel_id),
        ]
        if not use_rp_only_character:
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
        persona_engine = getattr(self, "persona_engine", None)
        if persona_engine is not None:
            with contextlib.suppress(Exception):
                overlay = await persona_engine.build_prompt_overlay(
                    mode="text",
                    guild_id=guild_id,
                    channel_id=channel_id,
                    user_id=user_id,
                    query_text=user_text,
                )
                if overlay:
                    system_parts.append(overlay)

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

        raw_facts = await self._get_user_facts_with_global_fallback(
            guild_id,
            user_id,
            local_limit=48,
            target_count=max(8, int(getattr(self.settings, "memory_fact_top_k", 4) or 4)),
        )
        if not raw_facts:
            return []

        query_tokens = tokenize(query_text)
        scored: list[tuple[float, dict[str, Any]]] = []

        for index, fact in enumerate(raw_facts):
            fact_text = str(fact.get("fact_value") or "").strip()
            if not fact_text:
                continue
            fact_type = str(fact.get("fact_type") or "").strip().lower()
            fact_key = str(fact.get("fact_key") or "").strip().lower()
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
            if fact_type == "identity" or fact_key.startswith("identity:"):
                base += 0.65
            elif fact_type == "preference":
                base += 0.25

            fact_tokens = tokenize(fact_text)
            overlap = len(query_tokens.intersection(fact_tokens)) if query_tokens else 0
            if query_tokens:
                base += (overlap / max(1.0, float(len(query_tokens)))) * 1.9
                if overlap > 0:
                    base += 0.22
                if ({"звати", "name"} & query_tokens) and (fact_type == "identity" or fact_key.startswith("identity:")):
                    base += 1.25

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
