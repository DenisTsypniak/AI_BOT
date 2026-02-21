from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from typing import List

import discord
from discord.ext import commands

from config import Settings
from memory import MemoryStore
from ollama_client import OllamaClient
from voice import VoiceManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("discord_ai_bot")


def chunk_text(text: str, limit: int = 1900) -> List[str]:
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) <= limit:
            current += line
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(line) <= limit:
            current = line
            continue
        for i in range(0, len(line), limit):
            chunks.append(line[i : i + limit])
    if current:
        chunks.append(current)
    return chunks


def has_permission(ctx: commands.Context, permission_name: str) -> bool:
    if not ctx.guild or not isinstance(ctx.author, discord.Member):
        return False
    perms = ctx.author.guild_permissions
    return perms.administrator or bool(getattr(perms, permission_name, False))


class DiscordAIBot(commands.Bot):
    def __init__(
        self,
        settings: Settings,
        memory: MemoryStore,
        llm: OllamaClient,
        voice: VoiceManager,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.messages = True
        intents.voice_states = True

        super().__init__(command_prefix=settings.command_prefix, intents=intents)
        self.settings = settings
        self.memory = memory
        self.llm = llm
        self.voice = voice
        self.channel_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def setup_hook(self) -> None:
        await self.memory.init()
        await self.llm.start()

    async def close(self) -> None:
        await self.voice.shutdown_all()
        await self.llm.close()
        await super().close()

    async def on_ready(self) -> None:
        if self.user:
            logger.info("Connected as %s (%s)", self.user, self.user.id)

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return

        await self.process_commands(message)
        if message.content.startswith(self.settings.command_prefix):
            return

        if not self.should_auto_reply(message):
            return

        user_text = self.strip_bot_mention(message.content)
        if not user_text:
            return

        try:
            async with message.channel.typing():
                reply_text = await self.generate_reply(
                    guild_id=message.guild.id if message.guild else None,
                    channel_id=message.channel.id,
                    user_id=message.author.id,
                    user_text=user_text,
                )
            await self.send_chunks(message.channel, reply_text, reference=message)
            await self.maybe_speak(message.guild, message.author, reply_text)
        except Exception as exc:
            logger.exception("Auto-reply failed: %s", exc)
            await message.reply("I could not generate a reply right now.")

    def should_auto_reply(self, message: discord.Message) -> bool:
        if message.guild is None:
            return True
        if self.user and self.user.mentioned_in(message):
            return True
        if message.channel.id in self.settings.auto_reply_channel_ids:
            return True
        return not self.settings.mention_only

    def strip_bot_mention(self, text: str) -> str:
        if not self.user:
            return text.strip()
        pattern = re.compile(rf"<@!?{self.user.id}>")
        return pattern.sub("", text).strip()

    async def send_chunks(
        self,
        destination: discord.abc.Messageable,
        text: str,
        reference: discord.Message | None = None,
    ) -> None:
        chunks = chunk_text(text, 1900)
        for index, part in enumerate(chunks):
            kwargs = {}
            if index == 0 and reference is not None:
                kwargs["reference"] = reference
            await destination.send(part, **kwargs)

    async def _get_effective_prompt(self, guild_id: int | None) -> str:
        if guild_id is None:
            return self.settings.default_system_prompt
        custom = await self.memory.get_guild_prompt(str(guild_id))
        return custom or self.settings.default_system_prompt

    async def generate_reply(
        self,
        guild_id: int | None,
        channel_id: int,
        user_id: int,
        user_text: str,
    ) -> str:
        guild_key = str(guild_id) if guild_id is not None else "dm"
        channel_key = str(channel_id)
        user_key = str(user_id)

        lock = self.channel_locks[channel_id]
        async with lock:
            await self.memory.save_message(guild_key, channel_key, user_key, "user", user_text)
            history = await self.memory.get_recent_messages(channel_key, self.settings.max_history_messages)
            prompt = await self._get_effective_prompt(guild_id)
            llm_messages = [{"role": "system", "content": prompt}, *history]

            reply = await self.llm.chat(llm_messages)
            if len(reply) > self.settings.max_response_chars:
                reply = reply[: self.settings.max_response_chars - 3] + "..."

            await self.memory.save_message(guild_key, channel_key, str(self.user.id if self.user else 0), "assistant", reply)
            return reply

    async def maybe_speak(
        self,
        guild: discord.Guild | None,
        author: discord.abc.User,
        text: str,
    ) -> None:
        if guild is None:
            return
        if not isinstance(author, discord.Member):
            return
        if not author.voice or not author.voice.channel:
            return
        voice_client = guild.voice_client
        if voice_client is None or voice_client.channel != author.voice.channel:
            return

        setting = await self.memory.get_voice_auto_speak(str(guild.id))
        enabled = self.settings.voice_auto_speak_default if setting is None else setting
        if not enabled:
            return

        await self.voice.enqueue(guild, text)


def register_commands(bot: DiscordAIBot) -> None:
    @bot.command(name="help_ai")
    async def help_ai(ctx: commands.Context) -> None:
        lines = [
            "AI command list:",
            f"`{bot.settings.command_prefix}ask <text>` Ask the assistant.",
            f"`{bot.settings.command_prefix}prompt_set <text>` Set guild behavior prompt (Manage Guild).",
            f"`{bot.settings.command_prefix}prompt_show` Show active behavior prompt.",
            f"`{bot.settings.command_prefix}prompt_reset` Reset to default prompt (Manage Guild).",
            f"`{bot.settings.command_prefix}memory_clear` Clear channel memory (Manage Messages).",
            f"`{bot.settings.command_prefix}join` Join your voice channel.",
            f"`{bot.settings.command_prefix}leave` Leave voice channel.",
            f"`{bot.settings.command_prefix}voice_on` / `{bot.settings.command_prefix}voice_off` Toggle auto-voice replies.",
            f"`{bot.settings.command_prefix}say <text>` Speak text in active voice channel.",
            "Mention the bot in chat to get contextual replies.",
        ]
        await ctx.send("\n".join(lines))

    @bot.command(name="ask")
    async def ask(ctx: commands.Context, *, question: str) -> None:
        question = question.strip()
        if not question:
            await ctx.send("Provide text after command.")
            return
        try:
            async with ctx.typing():
                reply = await bot.generate_reply(
                    guild_id=ctx.guild.id if ctx.guild else None,
                    channel_id=ctx.channel.id,
                    user_id=ctx.author.id,
                    user_text=question,
                )
            await bot.send_chunks(ctx.channel, reply, reference=ctx.message)
            await bot.maybe_speak(ctx.guild, ctx.author, reply)
        except Exception as exc:
            logger.exception("Ask command failed: %s", exc)
            await ctx.send("Failed to generate response.")

    @bot.command(name="prompt_set")
    async def prompt_set(ctx: commands.Context, *, prompt: str) -> None:
        if not has_permission(ctx, "manage_guild"):
            await ctx.send("You need `Manage Server` permission for this command.")
            return
        if not ctx.guild:
            await ctx.send("This command works only in a server.")
            return
        prompt = prompt.strip()
        if not prompt:
            await ctx.send("Prompt cannot be empty.")
            return
        await bot.memory.set_guild_prompt(str(ctx.guild.id), prompt)
        await ctx.send("Custom behavior prompt saved for this server.")

    @bot.command(name="prompt_show")
    async def prompt_show(ctx: commands.Context) -> None:
        prompt = await bot._get_effective_prompt(ctx.guild.id if ctx.guild else None)
        await bot.send_chunks(ctx.channel, f"Current prompt:\n{prompt}")

    @bot.command(name="prompt_reset")
    async def prompt_reset(ctx: commands.Context) -> None:
        if not has_permission(ctx, "manage_guild"):
            await ctx.send("You need `Manage Server` permission for this command.")
            return
        if not ctx.guild:
            await ctx.send("This command works only in a server.")
            return
        await bot.memory.reset_guild_prompt(str(ctx.guild.id))
        await ctx.send("Prompt reset to default.")

    @bot.command(name="memory_clear")
    async def memory_clear(ctx: commands.Context) -> None:
        if not has_permission(ctx, "manage_messages"):
            await ctx.send("You need `Manage Messages` permission for this command.")
            return
        await bot.memory.clear_channel_history(str(ctx.channel.id))
        await ctx.send("Channel memory cleared.")

    @bot.command(name="join")
    async def join(ctx: commands.Context) -> None:
        if not ctx.guild or not isinstance(ctx.author, discord.Member):
            await ctx.send("This command works only in a server voice channel.")
            return
        if not ctx.author.voice or not ctx.author.voice.channel:
            await ctx.send("Join a voice channel first.")
            return
        target = ctx.author.voice.channel
        voice_client = ctx.guild.voice_client
        if voice_client:
            if voice_client.channel != target:
                await voice_client.move_to(target)
                await ctx.send(f"Moved to `{target.name}`.")
            else:
                await ctx.send(f"Already in `{target.name}`.")
            return
        await target.connect()
        await ctx.send(f"Joined `{target.name}`.")

    @bot.command(name="leave")
    async def leave(ctx: commands.Context) -> None:
        if not ctx.guild:
            await ctx.send("This command works only in a server.")
            return
        voice_client = ctx.guild.voice_client
        if not voice_client:
            await ctx.send("I am not in a voice channel.")
            return
        await voice_client.disconnect(force=True)
        await bot.voice.shutdown_guild(ctx.guild.id)
        await ctx.send("Disconnected from voice channel.")

    @bot.command(name="voice_on")
    async def voice_on(ctx: commands.Context) -> None:
        if not has_permission(ctx, "manage_guild"):
            await ctx.send("You need `Manage Server` permission for this command.")
            return
        if not ctx.guild:
            await ctx.send("This command works only in a server.")
            return
        await bot.memory.set_voice_auto_speak(str(ctx.guild.id), True)
        await ctx.send("Auto-voice replies enabled for this server.")

    @bot.command(name="voice_off")
    async def voice_off(ctx: commands.Context) -> None:
        if not has_permission(ctx, "manage_guild"):
            await ctx.send("You need `Manage Server` permission for this command.")
            return
        if not ctx.guild:
            await ctx.send("This command works only in a server.")
            return
        await bot.memory.set_voice_auto_speak(str(ctx.guild.id), False)
        await ctx.send("Auto-voice replies disabled for this server.")

    @bot.command(name="say")
    async def say(ctx: commands.Context, *, text: str) -> None:
        if not ctx.guild or not isinstance(ctx.author, discord.Member):
            await ctx.send("This command works only in a server.")
            return
        if not ctx.author.voice or not ctx.author.voice.channel:
            await ctx.send("Join a voice channel first.")
            return

        if ctx.guild.voice_client is None:
            await ctx.author.voice.channel.connect()

        await bot.voice.enqueue(ctx.guild, text)
        await ctx.send("Queued for voice playback.")

    @bot.event
    async def on_command_error(ctx: commands.Context, error: commands.CommandError) -> None:
        if isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(f"Missing argument. Use `{bot.settings.command_prefix}help_ai`.")
            return
        if isinstance(error, commands.CommandNotFound):
            return
        logger.exception("Command error: %s", error)
        await ctx.send("Command failed.")


def main() -> None:
    settings = Settings.from_env()
    settings.validate()

    memory = MemoryStore(settings.sqlite_path)
    llm = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        temperature=settings.ollama_temperature,
        num_ctx=settings.ollama_num_ctx,
    )
    voice = VoiceManager(
        voice_name=settings.tts_voice,
        voice_rate=settings.tts_rate,
        max_tts_chars=settings.max_tts_chars,
    )

    bot = DiscordAIBot(settings=settings, memory=memory, llm=llm, voice=voice)
    register_commands(bot)
    bot.run(settings.discord_token)


if __name__ == "__main__":
    main()
