from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import discord

try:
    import audioop
except ModuleNotFoundError:
    import audioop_lts as audioop  # type: ignore[import-not-found]

from ..native_audio.audio import StreamingPCMAudioSource
from .config import LiveKitAgentSettings
from .observability import HealthHeartbeat, LiveKitRuntimeHealth

logger = logging.getLogger("live_role_bot.livekit.bridge")


BridgeMode = Literal["discord_to_livekit", "hybrid_voice"]


@dataclass(slots=True)
class LiveKitBridgeConfig:
    enabled: bool = False
    mode: BridgeMode = "discord_to_livekit"
    control_channel_name: str = "bridge-control"
    room_prefix: str = "discord-bridge"


@dataclass(slots=True)
class BridgeBinding:
    guild_id: int
    discord_voice_channel_id: int
    discord_text_channel_id: int
    livekit_room_name: str
    active: bool = False


@dataclass(slots=True)
class BridgeRegistry:
    bindings: dict[tuple[int, int], BridgeBinding] = field(default_factory=dict)

    def upsert(self, binding: BridgeBinding) -> None:
        self.bindings[(binding.guild_id, binding.discord_voice_channel_id)] = binding

    def get(self, guild_id: int, voice_channel_id: int) -> BridgeBinding | None:
        return self.bindings.get((guild_id, voice_channel_id))


class DiscordLiveKitBridge:
    """
    Phase 2 scaffold.

    This class is an adapter boundary for a future Discord<->LiveKit voice relay.
    Current implementation only keeps binding metadata and emits logs.
    """

    def __init__(self, config: LiveKitBridgeConfig) -> None:
        self.config = config
        self.registry = BridgeRegistry()

    def bind_channel(
        self,
        guild_id: int,
        discord_voice_channel_id: int,
        discord_text_channel_id: int,
        room_name: str,
    ) -> BridgeBinding:
        binding = BridgeBinding(
            guild_id=guild_id,
            discord_voice_channel_id=discord_voice_channel_id,
            discord_text_channel_id=discord_text_channel_id,
            livekit_room_name=room_name,
            active=False,
        )
        self.registry.upsert(binding)
        logger.info(
            "[bridge.plan] bound guild=%s voice=%s text=%s room=%s",
            guild_id,
            discord_voice_channel_id,
            discord_text_channel_id,
            room_name,
        )
        return binding

    async def start_binding(self, guild_id: int, discord_voice_channel_id: int) -> None:
        binding = self.registry.get(guild_id, discord_voice_channel_id)
        if binding is None:
            raise KeyError("Bridge binding not found")
        binding.active = True
        logger.info(
            "[bridge.plan] start requested guild=%s voice=%s room=%s (relay not implemented yet)",
            guild_id,
            discord_voice_channel_id,
            binding.livekit_room_name,
        )

    async def stop_binding(self, guild_id: int, discord_voice_channel_id: int) -> None:
        binding = self.registry.get(guild_id, discord_voice_channel_id)
        if binding is None:
            return
        binding.active = False
        logger.info(
            "[bridge.plan] stop requested guild=%s voice=%s room=%s",
            guild_id,
            discord_voice_channel_id,
            binding.livekit_room_name,
        )


@dataclass(slots=True)
class _BridgeSessionState:
    guild_id: int
    voice_channel_id: int
    text_channel_id: int
    room_name: str
    identity: str
    room: Any
    local_audio_source: Any
    local_audio_track: Any
    local_publication: Any
    agent_dispatch_id: str | None = None
    agent_dispatch_owned: bool = False
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    pcm_queue: asyncio.Queue[bytes] = field(default_factory=lambda: asyncio.Queue(maxsize=220))
    sender_task: asyncio.Task[None] | None = None
    remote_tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    playback_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    discord_source: StreamingPCMAudioSource | None = None
    heartbeat: HealthHeartbeat | None = None
    health: LiveKitRuntimeHealth | None = None
    discord_ingress_started_emitted: bool = False


class LiveKitDiscordBridgeManager:
    """
    Real Discord <-> LiveKit audio relay (Phase 2).

    Direction:
    - Discord inbound PCM (48k stereo) -> published local audio track in LiveKit room
    - Remote LiveKit audio tracks -> Discord voice playback
    """

    def __init__(self, discord_client: discord.Client, settings: LiveKitAgentSettings) -> None:
        self.client = discord_client
        self.settings = settings
        self.registry = BridgeRegistry()
        self._sessions: dict[int, _BridgeSessionState] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

        try:
            from livekit import rtc, api  # type: ignore
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(
                "LiveKit Python SDK is not available in this environment. "
                "Install optional LiveKit dependencies before enabling bridge mode."
            ) from exc
        self._rtc = rtc
        self._api = api

    def has_session(self, guild_id: int) -> bool:
        state = self._sessions.get(guild_id)
        return state is not None and not state.stop_event.is_set()

    def bind_channel(
        self,
        guild_id: int,
        discord_voice_channel_id: int,
        discord_text_channel_id: int,
        room_name: str,
    ) -> BridgeBinding:
        binding = BridgeBinding(
            guild_id=guild_id,
            discord_voice_channel_id=discord_voice_channel_id,
            discord_text_channel_id=discord_text_channel_id,
            livekit_room_name=room_name,
            active=False,
        )
        self.registry.upsert(binding)
        logger.info(
            "[bridge.live] bound guild=%s voice=%s text=%s room=%s",
            guild_id,
            discord_voice_channel_id,
            discord_text_channel_id,
            room_name,
        )
        return binding

    def build_room_name(self, guild_id: int, voice_channel_id: int) -> str:
        return f"{self.settings.room_prefix}-g{guild_id}-v{voice_channel_id}"

    def push_pcm(self, guild_id: int, user_id: int, user_label: str, pcm_48k_stereo: bytes) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._enqueue_pcm, guild_id, user_id, user_label, pcm_48k_stereo)

    def _enqueue_pcm(self, guild_id: int, user_id: int, user_label: str, pcm_48k_stereo: bytes) -> None:
        del user_id, user_label  # Mixed bridge feed currently uses one published track.
        if not pcm_48k_stereo:
            return
        state = self._sessions.get(guild_id)
        if state is None or state.stop_event.is_set():
            return
        if state.pcm_queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                state.pcm_queue.get_nowait()
                state.pcm_queue.task_done()
        with contextlib.suppress(asyncio.QueueFull):
            state.pcm_queue.put_nowait(bytes(pcm_48k_stereo))
        if state.health is not None:
            state.health.mark_activity()

    async def start_session(
        self,
        guild_id: int,
        voice_channel_id: int,
        text_channel_id: int,
        bot_user_id: int,
    ) -> str:
        self._loop = asyncio.get_running_loop()
        await self.stop_session(guild_id)

        room_name = self.build_room_name(guild_id, voice_channel_id)
        binding = self.bind_channel(guild_id, voice_channel_id, text_channel_id, room_name)
        identity = f"discord-bridge-{guild_id}-{bot_user_id}"
        room = self._rtc.Room(loop=self._loop)

        self._attach_room_event_handlers(room, guild_id)
        token = self._build_access_token(room_name, identity)
        options = self._rtc.RoomOptions(auto_subscribe=True)
        await room.connect(self.settings.url, token, options)

        audio_source = self._rtc.AudioSource(sample_rate=48000, num_channels=2, queue_size_ms=1000, loop=self._loop)
        local_track = self._rtc.LocalAudioTrack.create_audio_track("discord-bridge-in", audio_source)
        publish_options = self._rtc.TrackPublishOptions()
        with contextlib.suppress(Exception):
            publish_options.source = self._rtc.TrackSource.SOURCE_MICROPHONE
        with contextlib.suppress(Exception):
            publish_options.stream = "discord-bridge"
        publication = await room.local_participant.publish_track(local_track, publish_options)

        state = _BridgeSessionState(
            guild_id=guild_id,
            voice_channel_id=voice_channel_id,
            text_channel_id=text_channel_id,
            room_name=room_name,
            identity=identity,
            room=room,
            local_audio_source=audio_source,
            local_audio_track=local_track,
            local_publication=publication,
        )
        state.health = LiveKitRuntimeHealth(
            worker_name="discord-bridge",
            agent_name=getattr(self.client.user, "name", "discord-bot"),
            room_name=room_name,
        )
        state.health.rooms_started = 1
        state.heartbeat = HealthHeartbeat(state.health, self.settings.health_log_interval_seconds)
        state.heartbeat.start()

        self._sessions[guild_id] = state
        binding.active = True
        state.sender_task = asyncio.create_task(self._sender_loop(state), name=f"lk-bridge-sender-{guild_id}")

        await self._ensure_agent_dispatched(state)
        await self._emit_status(
            state,
            event="bridge_started",
            payload={
                "guild_id": guild_id,
                "voice_channel_id": voice_channel_id,
                "text_channel_id": text_channel_id,
                "room_name": room_name,
            },
            send_text=True,
        )

        # Attach already subscribed tracks if any were present before callbacks fired.
        await self._attach_existing_remote_audio_tracks(state)
        logger.info(
            "[bridge.live] started guild=%s voice=%s room=%s pub_sid=%s",
            guild_id,
            voice_channel_id,
            room_name,
            getattr(publication, "sid", "?"),
        )
        return room_name

    async def stop_session(self, guild_id: int) -> None:
        state = self._sessions.pop(guild_id, None)
        if state is None:
            return
        state.stop_event.set()

        for task in list(state.remote_tasks.values()):
            task.cancel()
        for task in list(state.remote_tasks.values()):
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(task, timeout=0.75)
        state.remote_tasks.clear()

        if state.sender_task is not None:
            state.sender_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(state.sender_task, timeout=0.75)

        if state.discord_source is not None:
            with contextlib.suppress(Exception):
                state.discord_source.force_stop()
            state.discord_source = None

        if state.heartbeat is not None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(state.heartbeat.stop(), timeout=1.0)
        if state.health is not None:
            state.health.rooms_closed += 1
            state.health.mark_activity()
        await self._emit_status(
            state,
            event="bridge_stopping",
            payload={"room_name": state.room_name},
            send_text=True,
            best_effort=True,
        )

        pub_sid = str(getattr(state.local_publication, "sid", "") or "")
        if pub_sid:
            with contextlib.suppress(Exception):
                await state.room.local_participant.unpublish_track(pub_sid)

        if state.agent_dispatch_id and state.agent_dispatch_owned:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._delete_agent_dispatch(state.room_name, state.agent_dispatch_id), timeout=1.5)

        with contextlib.suppress(Exception):
            await asyncio.wait_for(state.room.disconnect(), timeout=2.5)

        binding = self.registry.get(guild_id, state.voice_channel_id)
        if binding is not None:
            binding.active = False

        logger.info(
            "[bridge.live] stopped guild=%s voice=%s room=%s",
            guild_id,
            state.voice_channel_id,
            state.room_name,
        )

    async def shutdown_all(self) -> None:
        for guild_id in list(self._sessions.keys()):
            await self.stop_session(guild_id)

    def _build_access_token(self, room_name: str, identity: str) -> str:
        grants = self._api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
            # Keep the bridge participant visible so LiveKit Agents RoomIO can resolve
            # participant -> track ownership correctly (avoids "unknown participant" tracks).
            hidden=False,
        )
        token = (
            self._api.AccessToken(self.settings.api_key, self.settings.api_secret)
            .with_identity(identity)
            .with_name(identity)
            .with_grants(grants)
        )
        return token.to_jwt()

    async def _ensure_agent_dispatched(self, state: _BridgeSessionState) -> None:
        agent_name = (self.settings.agent_name or "").strip()
        if not agent_name:
            return
        try:
            lkapi = self._api.LiveKitAPI(
                url=self.settings.url,
                api_key=self.settings.api_key,
                api_secret=self.settings.api_secret,
            )
        except Exception as exc:
            logger.warning("[bridge.live] failed to init LiveKitAPI for dispatch room=%s: %s", state.room_name, exc)
            return

        try:
            existing = await lkapi.agent_dispatch.list_dispatch(state.room_name)
            for item in existing:
                if str(getattr(item, "agent_name", "") or "") == agent_name:
                    state.agent_dispatch_id = str(getattr(item, "id", "") or "") or None
                    state.agent_dispatch_owned = False
                    logger.info(
                        "[bridge.live] agent dispatch already exists room=%s agent=%s dispatch_id=%s",
                        state.room_name,
                        agent_name,
                        state.agent_dispatch_id or "?",
                    )
                    await self._emit_status(
                        state,
                        event="agent_dispatch_existing",
                        payload={"agent_name": agent_name, "dispatch_id": state.agent_dispatch_id or ""},
                        send_text=True,
                        best_effort=True,
                    )
                    return

            req = self._api.CreateAgentDispatchRequest(
                agent_name=agent_name,
                room=state.room_name,
                metadata=f'{{"source":"discord-bridge","guild_id":{state.guild_id},"voice_channel_id":{state.voice_channel_id}}}',
            )
            dispatch = await lkapi.agent_dispatch.create_dispatch(req)
            state.agent_dispatch_id = str(getattr(dispatch, "id", "") or "") or None
            state.agent_dispatch_owned = True
            logger.info(
                "[bridge.live] agent dispatch created room=%s agent=%s dispatch_id=%s",
                state.room_name,
                agent_name,
                state.agent_dispatch_id or "?",
            )
            await self._emit_status(
                state,
                event="agent_dispatch_created",
                payload={"agent_name": agent_name, "dispatch_id": state.agent_dispatch_id or ""},
                send_text=True,
                best_effort=True,
            )
        except Exception as exc:
            logger.warning("[bridge.live] agent dispatch failed room=%s agent=%s: %s", state.room_name, agent_name, exc)
            await self._emit_status(
                state,
                event="agent_dispatch_failed",
                payload={"agent_name": agent_name, "error": str(exc)},
                send_text=True,
                best_effort=True,
            )
        finally:
            with contextlib.suppress(Exception):
                await lkapi.aclose()

    async def _delete_agent_dispatch(self, room_name: str, dispatch_id: str) -> None:
        if not room_name or not dispatch_id:
            return
        lkapi = None
        try:
            lkapi = self._api.LiveKitAPI(
                url=self.settings.url,
                api_key=self.settings.api_key,
                api_secret=self.settings.api_secret,
            )
            await lkapi.agent_dispatch.delete_dispatch(dispatch_id, room_name)
            logger.info("[bridge.live] agent dispatch deleted room=%s dispatch_id=%s", room_name, dispatch_id)
        except Exception as exc:
            logger.debug("[bridge.live] failed to delete agent dispatch room=%s dispatch_id=%s: %s", room_name, dispatch_id, exc)
        finally:
            if lkapi is not None:
                with contextlib.suppress(Exception):
                    await lkapi.aclose()

    def _attach_room_event_handlers(self, room: Any, guild_id: int) -> None:
        def _on_connected() -> None:
            logger.info("[bridge.live] room connected guild=%s", guild_id)
            self._schedule_status_event(guild_id, "room_connected")

        def _on_disconnected(reason: object) -> None:
            logger.info("[bridge.live] room disconnected guild=%s reason=%s", guild_id, reason)
            self._schedule_status_event(guild_id, "room_disconnected", {"reason": str(reason)})

        def _on_reconnecting() -> None:
            logger.warning("[bridge.live] room reconnecting guild=%s", guild_id)
            self._schedule_status_event(guild_id, "room_reconnecting")

        def _on_reconnected() -> None:
            logger.info("[bridge.live] room reconnected guild=%s", guild_id)
            self._schedule_status_event(guild_id, "room_reconnected")

        def _on_track_subscribed(track: Any, publication: Any, participant: Any) -> None:
            loop = self._loop
            if loop is None or loop.is_closed():
                return
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    self._handle_track_subscribed(guild_id, track, publication, participant),
                    name=f"lk-bridge-track-sub-{guild_id}",
                )
            )

        def _on_track_unsubscribed(track: Any, publication: Any, participant: Any) -> None:
            del track, participant
            loop = self._loop
            if loop is None or loop.is_closed():
                return
            pub_sid = str(getattr(publication, "sid", "") or "")
            loop.call_soon_threadsafe(self._cancel_remote_task, guild_id, pub_sid)

        room.on("connected", _on_connected)
        room.on("disconnected", _on_disconnected)
        room.on("reconnecting", _on_reconnecting)
        room.on("reconnected", _on_reconnected)
        room.on("track_subscribed", _on_track_subscribed)
        room.on("track_unsubscribed", _on_track_unsubscribed)

    async def _attach_existing_remote_audio_tracks(self, state: _BridgeSessionState) -> None:
        participants = getattr(state.room, "remote_participants", {}) or {}
        values = participants.values() if isinstance(participants, dict) else []
        for participant in list(values):
            pubs = getattr(participant, "track_publications", {}) or {}
            for publication in list(pubs.values()):
                track = getattr(publication, "track", None)
                if track is None:
                    continue
                await self._handle_track_subscribed(state.guild_id, track, publication, participant)

    async def _handle_track_subscribed(self, guild_id: int, track: Any, publication: Any, participant: Any) -> None:
        state = self._sessions.get(guild_id)
        if state is None or state.stop_event.is_set():
            return
        if participant is None:
            return

        participant_identity = str(getattr(participant, "identity", "") or "")
        if participant_identity and participant_identity == state.identity:
            return

        kind = getattr(track, "kind", None)
        if kind != self._rtc.TrackKind.KIND_AUDIO:
            return

        pub_sid = str(getattr(publication, "sid", "") or getattr(track, "sid", "") or "")
        if not pub_sid or pub_sid in state.remote_tasks:
            return

        task = asyncio.create_task(
            self._consume_remote_audio_track(state, pub_sid, track, participant_identity),
            name=f"lk-bridge-remote-audio-{guild_id}-{pub_sid}",
        )
        state.remote_tasks[pub_sid] = task
        logger.info(
            "[bridge.live] subscribed remote audio guild=%s room=%s participant=%s pub_sid=%s",
            guild_id,
            state.room_name,
            participant_identity or "?",
            pub_sid,
        )
        await self._emit_status(
            state,
            event="assistant_audio_stream_subscribed",
            payload={"participant_identity": participant_identity or "", "publication_sid": pub_sid},
            send_text=False,
            best_effort=True,
        )

    def _cancel_remote_task(self, guild_id: int, pub_sid: str) -> None:
        if not pub_sid:
            return
        state = self._sessions.get(guild_id)
        if state is None:
            return
        task = state.remote_tasks.pop(pub_sid, None)
        if task is not None:
            task.cancel()
        self._schedule_status_event(
            guild_id,
            "assistant_audio_stream_unsubscribed",
            {"publication_sid": pub_sid},
        )

    async def _sender_loop(self, state: _BridgeSessionState) -> None:
        while not state.stop_event.is_set():
            try:
                pcm = await asyncio.wait_for(state.pcm_queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            try:
                frame = self._discord_pcm_to_livekit_frame(pcm)
                if frame is None:
                    continue
                await state.local_audio_source.capture_frame(frame)
                if not state.discord_ingress_started_emitted:
                    state.discord_ingress_started_emitted = True
                    await self._emit_status(
                        state,
                        event="discord_audio_ingress_active",
                        payload={"sample_rate": 48000, "channels": 2},
                        send_text=False,
                        best_effort=True,
                    )
                if state.health is not None:
                    state.health.mark_activity()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Bridge sender loop error (guild=%s): %s", state.guild_id, exc)
            finally:
                state.pcm_queue.task_done()

    def _discord_pcm_to_livekit_frame(self, pcm_48k_stereo: bytes) -> Any | None:
        if not pcm_48k_stereo:
            return None
        if len(pcm_48k_stereo) % 4 != 0:
            trim = len(pcm_48k_stereo) % 4
            pcm_48k_stereo = pcm_48k_stereo[: len(pcm_48k_stereo) - trim]
        if not pcm_48k_stereo:
            return None
        samples_per_channel = len(pcm_48k_stereo) // 4
        return self._rtc.AudioFrame(
            data=pcm_48k_stereo,
            sample_rate=48000,
            num_channels=2,
            samples_per_channel=samples_per_channel,
        )

    async def _consume_remote_audio_track(
        self,
        state: _BridgeSessionState,
        pub_sid: str,
        track: Any,
        participant_identity: str,
    ) -> None:
        stream = None
        try:
            await self._emit_status(
                state,
                event="assistant_audio_stream_started",
                payload={"participant_identity": participant_identity or "", "publication_sid": pub_sid},
                send_text=False,
                best_effort=True,
            )
            stream = self._rtc.AudioStream.from_track(
                track=track,
                loop=self._loop,
                sample_rate=48000,
                num_channels=2,
                frame_size_ms=20,
            )
            async for event in stream:
                if state.stop_event.is_set():
                    break
                frame = getattr(event, "frame", None)
                if frame is None:
                    continue
                pcm = self._livekit_frame_to_discord_pcm(frame)
                if not pcm:
                    continue
                await self._play_discord_pcm(state, pcm)
                if state.health is not None:
                    state.health.mark_activity()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(
                "Bridge remote audio consumer error guild=%s participant=%s pub_sid=%s: %s",
                state.guild_id,
                participant_identity or "?",
                pub_sid,
                exc,
            )
        finally:
            if stream is not None:
                with contextlib.suppress(Exception):
                    await stream.aclose()
            state.remote_tasks.pop(pub_sid, None)
            with contextlib.suppress(Exception):
                await self._emit_status(
                    state,
                    event="assistant_audio_stream_stopped",
                    payload={"participant_identity": participant_identity or "", "publication_sid": pub_sid},
                    send_text=False,
                    best_effort=True,
                )

    def _livekit_frame_to_discord_pcm(self, frame: Any) -> bytes:
        try:
            pcm = frame.data.tobytes()
            sample_rate = int(frame.sample_rate)
            channels = int(frame.num_channels)
        except Exception:
            return b""

        if not pcm:
            return b""

        try:
            if channels == 1:
                pcm = audioop.tostereo(pcm, 2, 1, 1)
                channels = 2
            if channels != 2:
                return b""
            if sample_rate != 48000:
                pcm, _ = audioop.ratecv(pcm, 2, 2, sample_rate, 48000, None)
        except Exception:
            return b""
        return pcm

    async def _ensure_discord_playback_source(self, state: _BridgeSessionState) -> StreamingPCMAudioSource | None:
        guild = self.client.get_guild(state.guild_id)
        if guild is None:
            return None
        vc = guild.voice_client
        if vc is None or getattr(vc, "channel", None) is None:
            return None
        if int(vc.channel.id) != state.voice_channel_id:
            return None

        async with state.playback_lock:
            if state.discord_source is not None:
                if vc.is_playing() or vc.is_paused():
                    return state.discord_source
                state.discord_source = None

            if vc.is_playing() or vc.is_paused():
                with contextlib.suppress(Exception):
                    vc.stop()

            source = StreamingPCMAudioSource()

            def _after(error: Exception | None) -> None:
                if error:
                    logger.error("[bridge.live] Discord playback error guild=%s: %s", state.guild_id, error)

            try:
                vc.play(source, after=_after)
            except Exception as exc:
                logger.debug("[bridge.live] Failed to start Discord playback guild=%s: %s", state.guild_id, exc)
                return None

            state.discord_source = source
            return source

    async def _play_discord_pcm(self, state: _BridgeSessionState, pcm_48k_stereo: bytes) -> None:
        if not pcm_48k_stereo:
            return
        source = await self._ensure_discord_playback_source(state)
        if source is None:
            return
        source.feed(pcm_48k_stereo)

    def _schedule_status_event(
        self,
        guild_id: int,
        event: str,
        payload: dict[str, Any] | None = None,
        *,
        send_text: bool = False,
    ) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        def _run() -> None:
            state = self._sessions.get(guild_id)
            if state is None or state.stop_event.is_set():
                return
            asyncio.create_task(
                self._emit_status(
                    state,
                    event=event,
                    payload=payload or {},
                    send_text=send_text,
                    best_effort=True,
                ),
                name=f"lk-bridge-status-{guild_id}-{event}",
            )

        loop.call_soon_threadsafe(_run)

    async def _emit_status(
        self,
        state: _BridgeSessionState,
        *,
        event: str,
        payload: dict[str, Any],
        send_text: bool,
        best_effort: bool = False,
    ) -> None:
        packet = {
            "event": event,
            "source": "discord-livekit-bridge",
            "guild_id": state.guild_id,
            "voice_channel_id": state.voice_channel_id,
            "text_channel_id": state.text_channel_id,
            "room_name": state.room_name,
            "payload": payload or {},
        }
        if state.health is not None:
            state.health.mark_activity()

        try:
            await state.room.local_participant.publish_data(
                json.dumps(packet, ensure_ascii=False),
                reliable=True,
                topic=self.settings.bridge_control_channel,
            )
        except Exception as exc:
            if not best_effort:
                logger.debug(
                    "[bridge.live] publish_data failed guild=%s room=%s event=%s: %s",
                    state.guild_id,
                    state.room_name,
                    event,
                    exc,
                )

        if send_text:
            with contextlib.suppress(Exception):
                await self._send_text_status(state, event, payload)

    async def _send_text_status(self, state: _BridgeSessionState, event: str, payload: dict[str, Any]) -> None:
        channel = self.client.get_channel(state.text_channel_id)
        if channel is None or not hasattr(channel, "send"):
            return

        if event == "bridge_started":
            text = f"[bridge] LiveKit room started: `{state.room_name}`"
        elif event == "bridge_stopping":
            text = f"[bridge] LiveKit room stopping: `{state.room_name}`"
        elif event == "agent_dispatch_created":
            text = f"[bridge] Agent dispatch created for `{payload.get('agent_name') or self.settings.agent_name}`."
        elif event == "agent_dispatch_existing":
            text = f"[bridge] Agent dispatch already exists for `{payload.get('agent_name') or self.settings.agent_name}`."
        elif event == "agent_dispatch_failed":
            text = f"[bridge] Agent dispatch failed for `{payload.get('agent_name') or self.settings.agent_name}`. Check LiveKit agent logs."
        else:
            return

        await channel.send(text)
