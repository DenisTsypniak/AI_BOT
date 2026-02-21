from __future__ import annotations

import asyncio
import contextlib
import logging
import tempfile
from pathlib import Path
from uuid import uuid4

import discord
import edge_tts


logger = logging.getLogger(__name__)


class VoiceManager:
    def __init__(self, voice_name: str, voice_rate: str, max_tts_chars: int) -> None:
        self.voice_name = voice_name
        self.voice_rate = voice_rate
        self.max_tts_chars = max_tts_chars
        self._queues: dict[int, asyncio.Queue[str]] = {}
        self._workers: dict[int, asyncio.Task[None]] = {}

    async def enqueue(self, guild: discord.Guild, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        if len(cleaned) > self.max_tts_chars:
            cleaned = cleaned[: self.max_tts_chars - 3] + "..."

        queue = self._queues.setdefault(guild.id, asyncio.Queue())
        await queue.put(cleaned)
        self._ensure_worker(guild)

    def _ensure_worker(self, guild: discord.Guild) -> None:
        task = self._workers.get(guild.id)
        if task and not task.done():
            return
        self._workers[guild.id] = asyncio.create_task(self._worker(guild))

    async def _worker(self, guild: discord.Guild) -> None:
        queue = self._queues[guild.id]
        while True:
            text = await queue.get()
            temp_path: Path | None = None
            try:
                voice_client = guild.voice_client
                if voice_client is None:
                    continue

                loop = asyncio.get_running_loop()
                temp_path = Path(tempfile.gettempdir()) / f"discord_ai_tts_{guild.id}_{uuid4().hex}.mp3"
                tts = edge_tts.Communicate(text=text, voice=self.voice_name, rate=self.voice_rate)
                await tts.save(str(temp_path))

                finished = asyncio.Event()

                def _after_play(error: Exception | None) -> None:
                    if error:
                        logger.error("Voice playback error: %s", error)
                    loop.call_soon_threadsafe(finished.set)

                while voice_client.is_playing() or voice_client.is_paused():
                    await asyncio.sleep(0.1)

                source = discord.FFmpegPCMAudio(str(temp_path))
                voice_client.play(source, after=_after_play)
                await finished.wait()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Voice worker failed: %s", exc)
            finally:
                if temp_path and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        logger.warning("Failed to delete temp file: %s", temp_path)

    async def shutdown_guild(self, guild_id: int) -> None:
        task = self._workers.pop(guild_id, None)
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._queues.pop(guild_id, None)

    async def shutdown_all(self) -> None:
        guild_ids = list(self._workers.keys())
        for guild_id in guild_ids:
            await self.shutdown_guild(guild_id)
