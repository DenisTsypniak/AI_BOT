from __future__ import annotations

import contextlib
from typing import Any

import discord

from ..common import as_float, as_int, tokenize


class PromptMixin:
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
            (
                "RP CANON (from bot_history.json, highest priority):\n"
                f"{self.rp_canon_prompt}"
            )
            if self.rp_canon_prompt
            else "",
            f"Role name: {role_profile['name']}",
            f"Role goal: {role_profile['goal']}",
            f"Role style: {role_profile['style']}",
            f"Role constraints: {role_profile['constraints']}",
            "Voice mode: respond immediately after user stop. Keep emotional presence.",
            "Voice brevity rule: 1 short sentence. 2 short sentences only when absolutely needed.",
            "Do not give long explanations unless user explicitly asks for details.",
            "Always complete the last sentence naturally; never stop mid-sentence.",
            "Live behavior: keep listening while speaking; if user continues, rebuild response from new details.",
            (
                "Discussion mode: if user disagrees, provide one concise counterpoint and one clarifying question."
            ),
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
                    profile_lines.append(f"- {label}: {fact_text}")
                    added += 1
                if added >= 6:
                    break
            if profile_lines:
                lines.append("Known participant memory:")
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
            (
                "RP CANON (from bot_history.json, highest priority):\n"
                f"{self.rp_canon_prompt}"
            )
            if self.rp_canon_prompt
            else "",
            f"Role name: {role_profile['name']}",
            f"Role goal: {role_profile['goal']}",
            f"Role style: {role_profile['style']}",
            f"Role constraints: {role_profile['constraints']}",
            (
                "Conversation behavior: be present and natural, keep momentum, ask meaningful follow-up questions, "
                "support the user emotionally when needed, and keep discussion intellectually honest."
            ),
            (
                "Debate behavior: when disagreement appears, provide one clear argument and one respectful counterpoint, "
                "then invite the user to respond."
            ),
            f"Always answer in {self.settings.preferred_response_language} unless user explicitly requests another language.",
        ]

        if summary_text:
            system_parts.append(f"User dialogue memory summary: {summary_text}")
        if facts:
            fact_lines = [
                f"- [{fact['fact_type']} | c={fact['confidence']:.2f}] {fact['fact_value']}"
                for fact in facts
            ]
            system_parts.append("User memory facts relevant for this turn:\n" + "\n".join(fact_lines))

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
