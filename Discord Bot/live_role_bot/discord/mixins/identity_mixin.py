from __future__ import annotations

import contextlib
import re
import time

import discord

from ..common import collapse_spaces, tokenize


class IdentityMixin:
    def _should_auto_reply(self, message: discord.Message) -> bool:
        if message.guild is None:
            return True
        if self.user and self.user.mentioned_in(message):
            return True
        if message.channel.id in self.settings.auto_reply_channel_ids:
            return True
        return not self.settings.mention_only

    def _strip_bot_mention(self, text: str) -> str:
        if not self.user:
            return collapse_spaces(text)
        pattern = re.compile(rf"<@!?{self.user.id}>")
        return collapse_spaces(pattern.sub("", text))

    @staticmethod
    def _build_label(
        username: str,
        global_name: str | None,
        guild_nick: str | None,
    ) -> str:
        base = (global_name or username or "unknown").strip() or "unknown"
        nick = (guild_nick or "").strip() or base
        return f"{base}({nick})"

    async def _sync_member_identity(self, guild_id: int, member: discord.Member) -> str:
        username = member.name.strip() if member.name else "unknown"
        global_name = (member.global_name or "").strip() or None
        guild_nick = (member.nick or "").strip() or None
        label = self._build_label(username, global_name, guild_nick)
        await self.memory.upsert_user_identity(
            guild_id=str(guild_id),
            user_id=str(member.id),
            discord_username=username,
            discord_global_name=global_name,
            guild_nick=guild_nick,
            combined_label=label,
        )
        return label

    async def _resolve_user_label(self, guild_id: str, user_id: str) -> str:
        identity = await self.memory.get_user_identity(guild_id, user_id)
        if identity is not None:
            label = str(identity.get("combined_label") or "").strip()
            if label:
                return label

        if guild_id != "dm":
            with contextlib.suppress(Exception):
                guild = self.get_guild(int(guild_id))
                if guild is not None:
                    member = guild.get_member(int(user_id))
                    if member is not None:
                        return self._build_label(
                            member.name.strip() if member.name else "unknown",
                            (member.global_name or "").strip() or None,
                            (member.nick or "").strip() or None,
                        )
        return f"user:{user_id}"

    async def _resolve_role(self, guild_id: str) -> str:
        if guild_id == "dm":
            return self.settings.default_role_id

        await self.memory.ensure_guild_settings(guild_id, self.settings.default_role_id)
        role_id = await self.memory.get_guild_role_id(guild_id)
        if not role_id:
            return self.settings.default_role_id
        return role_id

    def _is_duplicate_native_user_transcript(self, guild_id: int, user_id: int, text: str) -> bool:
        normalized = collapse_spaces(text).casefold()
        if not normalized:
            return True
        now = time.monotonic()
        key = (guild_id, user_id, normalized)
        last = self._native_user_transcripts.get(key, 0.0)
        if now - last < self._native_transcript_dedupe_seconds:
            return True
        self._native_user_transcripts[key] = now
        if len(self._native_user_transcripts) > 1200:
            cutoff = now - self._native_transcript_dedupe_seconds * 4
            stale = [k for k, ts in self._native_user_transcripts.items() if ts < cutoff]
            for item in stale[:800]:
                self._native_user_transcripts.pop(item, None)
        return False

    def _is_duplicate_native_assistant_transcript(self, guild_id: int, text: str) -> bool:
        normalized = collapse_spaces(text).casefold()
        if not normalized:
            return True
        now = time.monotonic()
        key = (guild_id, normalized)
        last = self._native_assistant_transcripts.get(key, 0.0)
        if now - last < self._native_transcript_dedupe_seconds:
            return True
        self._native_assistant_transcripts[key] = now
        if len(self._native_assistant_transcripts) > 800:
            cutoff = now - self._native_transcript_dedupe_seconds * 4
            stale = [k for k, ts in self._native_assistant_transcripts.items() if ts < cutoff]
            for item in stale[:500]:
                self._native_assistant_transcripts.pop(item, None)
        return False

    def _is_recent_assistant_echo(self, guild_id: int, text: str) -> bool:
        normalized = collapse_spaces(text).casefold()
        if not normalized:
            return True

        now = time.monotonic()
        window_sec = max(8.0, self._native_transcript_dedupe_seconds * 2)
        exact_ts = self._native_assistant_transcripts.get((guild_id, normalized), 0.0)
        if exact_ts > 0.0 and now - exact_ts <= window_sec:
            return True

        user_tokens = tokenize(normalized)
        for (gid, assistant_text), ts in self._native_assistant_transcripts.items():
            if gid != guild_id:
                continue
            if now - ts > window_sec:
                continue
            if len(normalized) >= 18 and (
                normalized in assistant_text or assistant_text in normalized
            ):
                return True

            assistant_tokens = tokenize(assistant_text)
            if not user_tokens or not assistant_tokens:
                continue
            overlap = len(user_tokens & assistant_tokens) / float(max(len(user_tokens), len(assistant_tokens)))
            if overlap >= 0.82 and len(user_tokens) >= 3:
                return True
        return False
