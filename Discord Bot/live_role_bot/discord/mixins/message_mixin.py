from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import discord

from ..common import ConversationSessionState, chunk_text, collapse_spaces, tokenize, truncate

logger = logging.getLogger("live_role_bot")


class MessageMixin:
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
            return

        lock = self.channel_locks[channel_key]
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
                    except Exception as e:
                        plugin_used = True
                        logger.error("Plugin %s failed to execute: %s", plugin.name, e)
                        reply = "Ой, сталася помилка при виконанні внутрішньої команди."

            # --- FALLBACK TO LLM ---
            if reply is None:
                reply = await self.llm.chat(llm_messages)

            reply = self._shape_human_reply(channel_key, clean_user_text, reply, modality)
            if self.settings.max_response_chars > 0 and len(reply) > self.settings.max_response_chars:
                reply = truncate(reply, self.settings.max_response_chars)
            self._register_bot_reply_state(channel_key, reply)

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
                source="plugin" if plugin_used else "gemini",
                quality=1.0,
            )
            logger.info("[msg.bot] channel=%s text=\"%s\"", channel_key, truncate(reply, 120))
            return reply
