from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import discord

from ..common import chunk_text, collapse_spaces, truncate

logger = logging.getLogger("live_role_bot")


class MessageMixin:
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
            return

        lock = self.channel_locks[channel_key]
        async with lock:
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
        self._enqueue_profile_update(guild_key, channel_key, user_key, message_id, cleaned)
        self._enqueue_summary_update(guild_key, channel_key, user_key)

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
            return

        lock = self.channel_locks[channel_key]
        async with lock:
            role_id = await self._resolve_role(guild_key)
            session_id = await self.memory.get_or_create_session(
                guild_id=guild_key,
                channel_id=channel_key,
                mode="voice",
                role_id=role_id,
            )
            await self.memory.save_message(
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

    async def _try_handle_system_command(self, message: discord.Message) -> bool:
        raw = collapse_spaces(message.content)
        if not raw:
            return False
        prefix = self.settings.command_prefix.strip()
        if not prefix or not raw.startswith(prefix):
            return False

        command = raw[len(prefix) :].strip().lower()
        join_label = f"{prefix}join"
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
            return
        channel_id = self.voice_text_channels.get(guild_id)
        if channel_id is None:
            return
        cleaned = collapse_spaces(text)
        if not cleaned:
            return
        if self._is_recent_assistant_echo(guild_id, cleaned):
            return
        if self._is_duplicate_native_user_transcript(guild_id, user_id, cleaned):
            return

        user_label = f"user:{user_id}"
        guild = self.get_guild(guild_id)
        if guild is not None:
            member = guild.get_member(user_id)
            if member is not None:
                with contextlib.suppress(Exception):
                    user_label = await self._sync_member_identity(guild_id, member)

        await self._save_native_user_transcript(
            guild_id=guild_id,
            channel_id=channel_id,
            user_id=user_id,
            user_label=user_label,
            text=cleaned,
            source=source,
            quality=confidence,
        )

    async def on_native_audio_assistant_transcript(
        self,
        guild_id: int,
        text: str,
        source: str = "gemini_native_output",
    ) -> None:
        channel_id = self.voice_text_channels.get(guild_id)
        if channel_id is None:
            return
        cleaned = collapse_spaces(text)
        if not cleaned:
            return
        if self._is_duplicate_native_assistant_transcript(guild_id, cleaned):
            return
        await self._save_native_assistant_transcript(
            guild_id=guild_id,
            channel_id=channel_id,
            text=cleaned,
            source=source,
        )

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

            clean_user_text = collapse_spaces(user_text)[:2000]
            user_message_id = await self.memory.save_message(
                session_id=session_id,
                guild_id=guild_key,
                channel_id=channel_key,
                user_id=user_key,
                author_label=user_label,
                role="user",
                modality=modality,
                content_raw=user_text[:3500],
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
            )
            self._enqueue_summary_update(guild_key, channel_key, user_key)

            llm_messages = await self._build_context_packet(
                guild_id=guild_key,
                channel_id=channel_key,
                user_id=user_key,
                session_id=session_id,
                role_id=role_id,
                user_text=clean_user_text,
            )

            reply = await self.llm.chat(llm_messages)
            if len(reply) > self.settings.max_response_chars:
                reply = truncate(reply, self.settings.max_response_chars)

            bot_user_id = str(self.user.id if self.user else 0)
            await self.memory.save_message(
                session_id=session_id,
                guild_id=guild_key,
                channel_id=channel_key,
                user_id=bot_user_id,
                author_label="assistant",
                role="assistant",
                modality="text",
                content_raw=reply,
                content_clean=reply,
                source="gemini",
                quality=1.0,
            )
            logger.info("[msg.bot] channel=%s text=\"%s\"", channel_key, truncate(reply, 120))
            return reply
