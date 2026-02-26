from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

import discord

try:
    import audioop
except ModuleNotFoundError:
    import audioop_lts as audioop  # type: ignore[import-not-found]

from ..common import PendingVoiceTurn, VoiceTurnBuffer, collapse_spaces
from ..voice_integration import VOICE_RECV_AVAILABLE, VoiceInputSink, voice_recv

logger = logging.getLogger("live_role_bot")


class VoiceMixin:
    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        # Handle manual/admin moves of the bot between voice channels by re-binding the voice pipeline
        # (Discord capture + LiveKit bridge/native audio) to the new channel.
        me = self.user
        if me is None or int(getattr(member, "id", 0) or 0) != int(getattr(me, "id", 0) or 0):
            return

        guild = getattr(member, "guild", None)
        if guild is None:
            return

        before_channel = getattr(before, "channel", None)
        after_channel = getattr(after, "channel", None)
        before_id = int(getattr(before_channel, "id", 0) or 0)
        after_id = int(getattr(after_channel, "id", 0) or 0)
        if before_id == after_id:
            return

        # Ignore initial bot connect (join command path already creates the pipeline).
        if before_channel is None and after_channel is not None:
            return

        # If bot was disconnected, clean up bridge/native sessions for this guild.
        if after_channel is None:
            logger.info("Bot voice state left channel in guild=%s; stopping voice sessions", guild.id)
            with contextlib.suppress(Exception):
                if getattr(self, "livekit_bridge", None) is not None:
                    await self.livekit_bridge.stop_session(guild.id)
            with contextlib.suppress(Exception):
                if getattr(self, "native_audio", None) is not None:
                    await self.native_audio.stop_session(guild.id)
            with contextlib.suppress(Exception):
                self.voice_text_channels.pop(guild.id, None)
            return

        text_channel_id = self.voice_text_channels.get(guild.id)
        if not text_channel_id:
            logger.info(
                "Bot moved to voice channel=%s in guild=%s but no voice text binding exists; waiting for next !join",
                after_id,
                guild.id,
            )
            return

        lock_key = f"voice-rebind:{guild.id}"
        lock = getattr(self, "channel_locks", {}).get(lock_key) if hasattr(self, "channel_locks") else None
        if lock is None:
            # Fallback if channel_locks is not initialized for some reason.
            logger.info(
                "Bot moved voice channel in guild=%s %s->%s; re-binding voice pipeline (no lock)",
                guild.id,
                before_id or "-",
                after_id,
            )
            await self._ensure_voice_capture(guild, member, int(text_channel_id))
            return

        async with lock:
            logger.info(
                "Bot moved voice channel in guild=%s %s->%s; re-binding voice pipeline",
                guild.id,
                before_id or "-",
                after_id,
            )
            await self._ensure_voice_capture(guild, member, int(text_channel_id))

    async def _connect_or_move_voice_client(
        self,
        guild: discord.Guild,
        target: discord.abc.Connectable,
    ) -> discord.VoiceClient | None:
        attempts = 3
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                voice_client = guild.voice_client
                if voice_client is None:
                    connected = await target.connect(
                        cls=voice_recv.VoiceRecvClient,
                        timeout=18.0,
                        reconnect=True,
                    )
                    if isinstance(connected, discord.VoiceClient):
                        return connected
                    return None

                if not isinstance(voice_client, discord.VoiceClient):
                    with contextlib.suppress(Exception):
                        await voice_client.disconnect(force=True)
                    voice_client = None
                    continue

                if voice_client.channel != target:
                    await asyncio.wait_for(voice_client.move_to(target), timeout=12.0)
                return voice_client
            except asyncio.CancelledError:
                raise
            except (asyncio.TimeoutError, TimeoutError) as exc:
                last_error = exc
                logger.warning(
                    "Voice connect attempt %s/%s timed out for guild=%s channel=%s",
                    attempt,
                    attempts,
                    guild.id,
                    getattr(target, "id", "unknown"),
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Voice connect attempt %s/%s failed for guild=%s channel=%s: %s",
                    attempt,
                    attempts,
                    guild.id,
                    getattr(target, "id", "unknown"),
                    exc,
                )
                logger.debug("Voice connect attempt exception details", exc_info=exc)

            current = guild.voice_client
            if current is not None:
                with contextlib.suppress(Exception):
                    if VOICE_RECV_AVAILABLE and voice_recv is not None and isinstance(current, voice_recv.VoiceRecvClient):
                        if current.is_listening():
                            current.stop_listening()
                with contextlib.suppress(Exception):
                    await current.disconnect(force=True)
            await asyncio.sleep(0.35 * attempt)

        if last_error is not None:
            logger.error(
                "Voice connect failed after retries for guild=%s channel=%s: %s",
                guild.id,
                getattr(target, "id", "unknown"),
                last_error,
            )
        return None

    async def _ensure_voice_capture(
        self,
        guild: discord.Guild,
        member: discord.Member,
        text_channel_id: int,
    ) -> bool:
        if not self.settings.voice_enabled or not self.settings.voice_auto_capture:
            return False
        if not VOICE_RECV_AVAILABLE or voice_recv is None:
            return False
        if not member.voice or not member.voice.channel:
            return False

        target = member.voice.channel
        voice_client = await self._connect_or_move_voice_client(guild, target)
        if voice_client is None:
            return False

        if not isinstance(voice_client, voice_recv.VoiceRecvClient):
            return False

        if voice_client.is_listening():
            voice_client.stop_listening()

        self.voice_text_channels[guild.id] = text_channel_id
        bridge_active = False

        if getattr(self, "livekit_bridge", None) is not None:
            try:
                room_name = await self.livekit_bridge.start_session(
                    guild_id=guild.id,
                    voice_channel_id=target.id,
                    text_channel_id=text_channel_id,
                    bot_user_id=self.user.id if self.user else 0,
                )
                logger.info(
                    "LiveKit bridge session started for guild=%s channel=%s room=%s",
                    guild.id,
                    target.id,
                    room_name,
                )
                bridge_active = True
            except Exception:
                logger.exception("LiveKit bridge start failed for guild=%s", guild.id)
                channel = self.get_channel(text_channel_id)
                if isinstance(
                    channel,
                    (
                        discord.TextChannel,
                        discord.Thread,
                        discord.DMChannel,
                        discord.GroupChannel,
                        discord.PartialMessageable,
                    ),
                ):
                    with contextlib.suppress(Exception):
                        await channel.send("LiveKit bridge failed to start. Check bot logs and LiveKit credentials.")
                return False

        if (not bridge_active) and self.settings.gemini_native_audio_enabled:
            if self.native_audio is None:
                logger.error("Native Audio is enabled but manager is unavailable")
                return False
            try:
                prompt = await self._build_native_audio_system_prompt(guild, target)
                await self.native_audio.start_session(
                    guild_id=guild.id,
                    voice_channel_id=target.id,
                    text_channel_id=text_channel_id,
                    bot_user_id=self.user.id if self.user else 0,
                    system_prompt=prompt,
                    preferred_language=self.settings.preferred_response_language,
                    send_transcripts_to_text=self.settings.voice_send_transcripts_to_text,
                )
                logger.info("Gemini Native Audio session started for guild=%s channel=%s", guild.id, target.id)
            except Exception:
                logger.exception("Native Audio start failed for guild=%s", guild.id)
                channel = self.get_channel(text_channel_id)
                if isinstance(
                    channel,
                    (
                        discord.TextChannel,
                        discord.Thread,
                        discord.DMChannel,
                        discord.GroupChannel,
                        discord.PartialMessageable,
                    ),
                ):
                    with contextlib.suppress(Exception):
                        await channel.send(
                            "Native Audio session failed to start. "
                            "Use `GEMINI_LIVE_MODEL=models/gemini-2.5-flash-native-audio-latest`."
                        )
                return False

        sink: Any = VoiceInputSink(self, guild.id, self.user.id if self.user else 0)

        def _after(error: Exception | None) -> None:
            if error:
                logger.error("Voice listening stopped with error: %s", error)

        voice_client.listen(sink, after=_after)
        logger.info("Voice capture active in guild=%s channel=%s", guild.id, target.id)
        return True

    def push_voice_pcm(self, guild_id: int, user_id: int, user_label: str, pcm_48k_stereo: bytes) -> None:
        key = (guild_id, user_id)
        if key not in self._seen_voice_pcm_users:
            self._seen_voice_pcm_users.add(key)
            logger.info("[voice.input] first PCM packet from user=%s guild=%s", user_id, guild_id)

        loop = self._loop
        bridge = getattr(self, "livekit_bridge", None)
        if bridge is not None and bridge.has_session(guild_id):
            bridge.push_pcm(guild_id, user_id, user_label, pcm_48k_stereo)
            if getattr(self, "bridge_voice_memory_transcript_enabled", False):
                if loop is not None and not loop.is_closed():
                    loop.call_soon_threadsafe(
                        self._ingest_voice_pcm,
                        guild_id,
                        user_id,
                        user_label,
                        pcm_48k_stereo,
                        False,  # reply is produced by LiveKit agent, local STT is for memory only
                        "bridge_local_stt",
                    )
            return

        if self.settings.gemini_native_audio_enabled:
            if self.native_audio is not None and self.native_audio.has_session(guild_id):
                self.native_audio.push_pcm(guild_id, user_id, user_label, pcm_48k_stereo)
            return

        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._ingest_voice_pcm, guild_id, user_id, user_label, pcm_48k_stereo, True, "local_stt")

    def _ingest_voice_pcm(
        self,
        guild_id: int,
        user_id: int,
        user_label: str,
        pcm_48k_stereo: bytes,
        reply_enabled: bool = True,
        transcript_source: str = "local_stt",
    ) -> None:
        if not pcm_48k_stereo:
            if callable(getattr(self, "_record_voice_memory_diag", None)):
                self._record_voice_memory_diag(
                    stage="voice_pcm_ingest",
                    outcome="drop",
                    reason="empty_pcm",
                    guild_id=guild_id,
                    user_id=user_id,
                    source=transcript_source,
                )
            return

        now = time.monotonic()
        key = (guild_id, user_id)
        state = self.voice_buffers.setdefault(key, VoiceTurnBuffer())
        state.reply_enabled = bool(reply_enabled)
        state.transcript_source = transcript_source or state.transcript_source

        try:
            rms = int(audioop.rms(pcm_48k_stereo, 2))
        except (audioop.error, TypeError, ValueError):
            rms = 0

        silence_threshold = max(20, self.settings.voice_silence_rms)
        if rms >= silence_threshold:
            if state.started_at <= 0:
                state.started_at = now
            state.last_voice_at = now
            state.user_label = user_label or state.user_label
            state.data.extend(pcm_48k_stereo)

            duration_sec = len(state.data) / (48000 * 4)
            if duration_sec >= self.settings.voice_max_turn_seconds:
                self._finalize_voice_turn(key)
            return

        if state.started_at > 0 and now - state.last_voice_at >= (self.settings.voice_silence_ms / 1000.0):
            self._finalize_voice_turn(key)

    def _finalize_voice_turn(self, key: tuple[int, int]) -> None:
        state = self.voice_buffers.pop(key, None)
        if state is None or not state.data:
            if callable(getattr(self, "_record_voice_memory_diag", None)):
                self._record_voice_memory_diag(
                    stage="voice_turn_finalize",
                    outcome="drop",
                    reason="empty_buffer",
                    guild_id=key[0],
                    user_id=key[1],
                )
            return

        pcm = bytes(state.data)
        duration_ms = int((len(pcm) / (48000 * 4)) * 1000)
        if duration_ms < self.settings.voice_min_turn_ms:
            if callable(getattr(self, "_record_voice_memory_diag", None)):
                self._record_voice_memory_diag(
                    stage="voice_turn_finalize",
                    outcome="drop",
                    reason="too_short",
                    guild_id=key[0],
                    user_id=key[1],
                    duration_ms=duration_ms,
                    min_turn_ms=self.settings.voice_min_turn_ms,
                )
            return

        guild_id, user_id = key
        channel_id = self.voice_text_channels.get(guild_id)
        if channel_id is None:
            if callable(getattr(self, "_record_voice_memory_diag", None)):
                self._record_voice_memory_diag(
                    stage="voice_turn_finalize",
                    outcome="drop",
                    reason="no_voice_text_channel_binding",
                    guild_id=guild_id,
                    user_id=user_id,
                    duration_ms=duration_ms,
                    source=state.transcript_source or "local_stt",
                )
            return

        item = PendingVoiceTurn(
            guild_id=guild_id,
            channel_id=channel_id,
            user_id=user_id,
            user_label=state.user_label,
            pcm_48k_stereo=pcm,
            reply_enabled=bool(state.reply_enabled),
            transcript_source=state.transcript_source or "local_stt",
        )

        if self.voice_turn_queue.full():
            if callable(getattr(self, "_record_voice_memory_diag", None)):
                self._record_voice_memory_diag(
                    stage="voice_turn_queue",
                    outcome="drop",
                    reason="queue_full_drop_oldest",
                    guild_id=guild_id,
                    user_id=user_id,
                    source=state.transcript_source or "local_stt",
                )
            with contextlib.suppress(asyncio.QueueEmpty):
                self.voice_turn_queue.get_nowait()
                self.voice_turn_queue.task_done()

        try:
            self.voice_turn_queue.put_nowait(item)
            if callable(getattr(self, "_record_voice_memory_diag", None)):
                self._record_voice_memory_diag(
                    stage="voice_turn_queue",
                    outcome="queued",
                    reason="ok",
                    guild_id=guild_id,
                    user_id=user_id,
                    channel_id=channel_id,
                    source=state.transcript_source or "local_stt",
                    duration_ms=duration_ms,
                    reply_enabled=bool(state.reply_enabled),
                )
        except asyncio.QueueFull:
            if callable(getattr(self, "_record_voice_memory_diag", None)):
                self._record_voice_memory_diag(
                    stage="voice_turn_queue",
                    outcome="drop",
                    reason="queue_full_put_failed",
                    guild_id=guild_id,
                    user_id=user_id,
                    source=state.transcript_source or "local_stt",
                )

    async def _voice_flush_loop(self) -> None:
        while True:
            await asyncio.sleep(0.18)
            if not self.voice_buffers:
                continue
            now = time.monotonic()
            silence_seconds = self.settings.voice_silence_ms / 1000.0
            stale: list[tuple[int, int]] = []
            for key, state in list(self.voice_buffers.items()):
                if state.started_at <= 0:
                    continue
                if now - state.last_voice_at >= silence_seconds:
                    stale.append(key)
            for key in stale:
                self._finalize_voice_turn(key)

    async def _voice_worker(self) -> None:
        while True:
            item = await self.voice_turn_queue.get()
            try:
                result = await self.local_stt.transcribe(item.pcm_48k_stereo)
                transcript = collapse_spaces(result.text)
                try:
                    await self.memory.save_stt_turn(
                        guild_id=str(item.guild_id),
                        channel_id=str(item.channel_id),
                        user_id=str(item.user_id),
                        duration_ms=result.duration_ms,
                        rms=result.rms,
                        transcript=transcript or None,
                        confidence=result.confidence,
                        model_name=result.model_name,
                        status=result.status,
                        message_id=None,
                    )
                    if callable(getattr(self, "_record_voice_memory_diag", None)):
                        self._record_voice_memory_diag(
                            stage="voice_stt",
                            outcome="save_stt_turn",
                            reason="ok",
                            guild_id=item.guild_id,
                            channel_id=item.channel_id,
                            user_id=item.user_id,
                            source=item.transcript_source or "local_stt",
                            confidence=f"{float(result.confidence):.3f}",
                            status=result.status,
                        )
                except Exception as exc:
                    if callable(getattr(self, "_record_voice_memory_diag", None)):
                        self._record_voice_memory_diag(
                            stage="voice_stt",
                            outcome="error",
                            reason="save_stt_turn_failed",
                            guild_id=item.guild_id,
                            channel_id=item.channel_id,
                            user_id=item.user_id,
                            source=item.transcript_source or "local_stt",
                            error=exc,
                        )
                    raise

                if not transcript:
                    if callable(getattr(self, "_record_voice_memory_diag", None)):
                        if result.status == "empty":
                            self._record_voice_memory_diag(
                                stage="voice_stt",
                                outcome="drop",
                                reason="empty_transcript",
                                guild_id=item.guild_id,
                                channel_id=item.channel_id,
                                user_id=item.user_id,
                                source=item.transcript_source or "local_stt",
                                status=result.status,
                            )
                        elif result.status in {"error", "model_unavailable"}:
                            self._record_voice_memory_diag(
                                stage="voice_stt",
                                outcome="error",
                                reason=f"transcribe_{result.status}",
                                guild_id=item.guild_id,
                                channel_id=item.channel_id,
                                user_id=item.user_id,
                                source=item.transcript_source or "local_stt",
                                status=result.status,
                                model_name=result.model_name,
                            )
                        else:
                            self._record_voice_memory_diag(
                                stage="voice_stt",
                                outcome="drop",
                                reason=f"no_transcript_status_{result.status or 'unknown'}",
                                guild_id=item.guild_id,
                                channel_id=item.channel_id,
                                user_id=item.user_id,
                                source=item.transcript_source or "local_stt",
                                status=result.status,
                            )
                    continue
                if result.confidence < self.settings.transcription_min_confidence:
                    if callable(getattr(self, "_record_voice_memory_diag", None)):
                        self._record_voice_memory_diag(
                            stage="voice_stt",
                            outcome="drop",
                            reason="low_confidence",
                            guild_id=item.guild_id,
                            channel_id=item.channel_id,
                            user_id=item.user_id,
                            source=item.transcript_source or "local_stt",
                            confidence=f"{float(result.confidence):.3f}",
                            threshold=f"{float(self.settings.transcription_min_confidence):.3f}",
                        )
                    continue

                guild = self.get_guild(item.guild_id)
                if guild is not None:
                    member = guild.get_member(item.user_id)
                    if member is not None:
                        with contextlib.suppress(Exception):
                            item.user_label = await self._sync_member_identity(item.guild_id, member)

                if not item.reply_enabled:
                    try:
                        await self._save_native_user_transcript(
                            guild_id=item.guild_id,
                            channel_id=item.channel_id,
                            user_id=item.user_id,
                            user_label=item.user_label,
                            text=transcript,
                            source=item.transcript_source or "bridge_local_stt",
                            quality=result.confidence,
                        )
                    except Exception as exc:
                        if callable(getattr(self, "_record_voice_memory_diag", None)):
                            self._record_voice_memory_diag(
                                stage="voice_stt",
                                outcome="error",
                                reason="save_native_user_transcript_failed",
                                guild_id=item.guild_id,
                                channel_id=item.channel_id,
                                user_id=item.user_id,
                                source=item.transcript_source or "bridge_local_stt",
                                error=exc,
                            )
                        raise
                    continue

                reply = await self._run_dialogue_turn(
                    guild_id=item.guild_id,
                    channel_id=item.channel_id,
                    user_id=item.user_id,
                    user_label=item.user_label,
                    user_text=transcript,
                    modality="voice",
                    source="local_stt",
                    quality=result.confidence,
                )

                channel_obj = self.get_channel(item.channel_id)
                messageable: discord.abc.Messageable | None = None
                if isinstance(
                    channel_obj,
                    (
                        discord.TextChannel,
                        discord.Thread,
                        discord.DMChannel,
                        discord.GroupChannel,
                        discord.PartialMessageable,
                    ),
                ):
                    messageable = channel_obj
                if messageable is not None:
                    await self._send_chunks(messageable, reply)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Voice worker error for guild=%s channel=%s user=%s source=%s",
                    getattr(item, "guild_id", 0),
                    getattr(item, "channel_id", 0),
                    getattr(item, "user_id", 0),
                    getattr(item, "transcript_source", ""),
                )
            finally:
                self.voice_turn_queue.task_done()
