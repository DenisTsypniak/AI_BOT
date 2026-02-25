from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import discord

try:
    import audioop
except ModuleNotFoundError:
    import audioop_lts as audioop  # type: ignore[import-not-found]

from ..native_audio.audio import StreamingPCMAudioSource
from ...discord.common import collapse_spaces
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
    last_ingress_user_id: str = ""
    last_ingress_user_label: str = ""
    last_ingress_unix: float = 0.0
    seen_transcription_segment_ids: set[str] = field(default_factory=set)
    context_sync_task: asyncio.Task[None] | None = None
    context_sync_seq: int = 0
    last_context_sent_at: float = 0.0
    last_context_sent_unix: float = 0.0
    last_context_hash: str = ""
    last_context_reason: str = ""
    last_context_publish_error: str = ""
    last_context_payload_bytes: int = 0
    context_publish_attempts: int = 0
    context_publish_success: int = 0
    context_publish_skipped_throttle: int = 0
    context_publish_skipped_unchanged: int = 0
    context_publish_errors: int = 0
    agent_ctx_updates_received: int = 0
    agent_ctx_updates_applied: int = 0
    agent_ctx_updates_ignored: int = 0
    agent_ctx_updates_errors: int = 0
    agent_ctx_ack_invalid: int = 0
    agent_ctx_last_seq: int = 0
    agent_ctx_last_reason: str = ""
    agent_ctx_last_chars: int = 0
    agent_ctx_last_error: str = ""
    agent_ctx_last_ack_unix: float = 0.0
    agent_ctx_sender_identity: str = ""


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

    def status_snapshot(self) -> dict[str, object]:
        sessions: list[dict[str, object]] = []
        now = time.monotonic()
        ctx_attempts = 0
        ctx_published = 0
        ctx_skipped = 0
        ctx_publish_errors = 0
        agent_rx = 0
        agent_apply = 0
        agent_ignored = 0
        agent_errors = 0
        agent_ack_invalid = 0
        for guild_id, state in self._sessions.items():
            idle_sec = None
            if state.health is not None:
                idle_sec = max(0.0, now - float(state.health.last_activity_at))
            ctx_attempts += int(state.context_publish_attempts)
            ctx_published += int(state.context_publish_success)
            ctx_skipped += int(state.context_publish_skipped_throttle + state.context_publish_skipped_unchanged)
            ctx_publish_errors += int(state.context_publish_errors)
            agent_rx += int(state.agent_ctx_updates_received)
            agent_apply += int(state.agent_ctx_updates_applied)
            agent_ignored += int(state.agent_ctx_updates_ignored)
            agent_errors += int(state.agent_ctx_updates_errors)
            agent_ack_invalid += int(state.agent_ctx_ack_invalid)
            last_ctx_age_sec = (
                max(0.0, time.time() - float(state.last_context_sent_unix))
                if float(state.last_context_sent_unix or 0.0) > 0
                else None
            )
            last_ack_age_sec = (
                max(0.0, time.time() - float(state.agent_ctx_last_ack_unix))
                if float(state.agent_ctx_last_ack_unix or 0.0) > 0
                else None
            )
            sessions.append(
                {
                    "guild_id": guild_id,
                    "voice_channel_id": state.voice_channel_id,
                    "text_channel_id": state.text_channel_id,
                    "room_name": state.room_name,
                    "pcm_queue": int(state.pcm_queue.qsize()),
                    "remote_streams": len(state.remote_tasks),
                    "dispatch_id": state.agent_dispatch_id or "",
                    "dispatch_owned": bool(state.agent_dispatch_owned),
                    "ingress_active": bool(state.discord_ingress_started_emitted),
                    "sender_task_alive": bool(state.sender_task is not None and not state.sender_task.done()),
                    "context_sync_task_alive": bool(
                        state.context_sync_task is not None and not state.context_sync_task.done()
                    ),
                    "context_seq": int(state.context_sync_seq),
                    "last_context_reason": str(state.last_context_reason or ""),
                    "last_context_error": str(state.last_context_publish_error or ""),
                    "last_context_sent_unix": float(state.last_context_sent_unix or 0.0),
                    "last_context_payload_bytes": int(state.last_context_payload_bytes or 0),
                    "context_publish_attempts": int(state.context_publish_attempts or 0),
                    "context_publish_success": int(state.context_publish_success or 0),
                    "context_publish_skipped_throttle": int(state.context_publish_skipped_throttle or 0),
                    "context_publish_skipped_unchanged": int(state.context_publish_skipped_unchanged or 0),
                    "context_publish_errors": int(state.context_publish_errors or 0),
                    "last_context_age_sec": last_ctx_age_sec,
                    "agent_ctx_updates_received": int(state.agent_ctx_updates_received or 0),
                    "agent_ctx_updates_applied": int(state.agent_ctx_updates_applied or 0),
                    "agent_ctx_updates_ignored": int(state.agent_ctx_updates_ignored or 0),
                    "agent_ctx_updates_errors": int(state.agent_ctx_updates_errors or 0),
                    "agent_ctx_ack_invalid": int(state.agent_ctx_ack_invalid or 0),
                    "agent_ctx_last_seq": int(state.agent_ctx_last_seq or 0),
                    "agent_ctx_last_reason": str(state.agent_ctx_last_reason or ""),
                    "agent_ctx_last_chars": int(state.agent_ctx_last_chars or 0),
                    "agent_ctx_last_error": str(state.agent_ctx_last_error or ""),
                    "agent_ctx_last_ack_unix": float(state.agent_ctx_last_ack_unix or 0.0),
                    "agent_ctx_sender_identity": str(state.agent_ctx_sender_identity or ""),
                    "agent_ctx_last_ack_age_sec": last_ack_age_sec,
                    "idle_sec": idle_sec,
                }
            )

        return {
            "enabled": True,
            "room_prefix": self.settings.room_prefix,
            "control_channel": self.settings.bridge_control_channel,
            "context_topic": self.settings.bridge_context_topic,
            "context_sync_enabled": bool(self.settings.bridge_context_sync_enabled),
            "context_sync": {
                "attempts": ctx_attempts,
                "published": ctx_published,
                "skipped": ctx_skipped,
                "skipped_throttle": sum(int(getattr(s, "context_publish_skipped_throttle", 0) or 0) for s in self._sessions.values()),
                "skipped_unchanged": sum(int(getattr(s, "context_publish_skipped_unchanged", 0) or 0) for s in self._sessions.values()),
                "publish_errors": ctx_publish_errors,
                "agent_updates_received": agent_rx,
                "agent_updates_applied": agent_apply,
                "agent_updates_ignored": agent_ignored,
                "agent_updates_errors": agent_errors,
                "agent_ack_invalid": agent_ack_invalid,
            },
            "bindings": len(self.registry.bindings),
            "sessions": sessions,
            "session_count": len(sessions),
        }

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
        if not pcm_48k_stereo:
            return
        state = self._sessions.get(guild_id)
        if state is None or state.stop_event.is_set():
            return
        state.last_ingress_user_id = str(user_id or "")
        state.last_ingress_user_label = self._trim_label(user_label, 48)
        state.last_ingress_unix = float(time.time())
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
        if self._context_sync_enabled():
            state.context_sync_task = asyncio.create_task(
                self._context_sync_loop(state),
                name=f"lk-bridge-context-sync-{guild_id}",
            )

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
            send_text=False,
        )
        await self._publish_context_snapshot(state, reason="bridge_started", force=True, best_effort=True)

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
        if state.context_sync_task is not None:
            state.context_sync_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(state.context_sync_task, timeout=0.75)

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
            send_text=False,
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
                        send_text=False,
                        best_effort=True,
                    )
                    await self._publish_context_snapshot(state, reason="agent_dispatch_existing", force=True, best_effort=True)
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
                send_text=False,
                best_effort=True,
            )
            await self._publish_context_snapshot(state, reason="agent_dispatch_created", force=True, best_effort=True)
        except Exception as exc:
            logger.warning("[bridge.live] agent dispatch failed room=%s agent=%s: %s", state.room_name, agent_name, exc)
            await self._emit_status(
                state,
                event="agent_dispatch_failed",
                payload={"agent_name": agent_name, "error": str(exc)},
                send_text=False,
                best_effort=True,
            )
            await self._publish_context_snapshot(state, reason="agent_dispatch_failed", force=True, best_effort=True)
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

    def _context_sync_enabled(self) -> bool:
        return bool(getattr(self.settings, "bridge_context_sync_enabled", True))

    def _context_min_interval_seconds(self) -> float:
        return max(0.1, float(int(getattr(self.settings, "bridge_context_min_interval_ms", 1200))) / 1000.0)

    def _context_force_interval_seconds(self) -> float:
        return max(1.0, float(int(getattr(self.settings, "bridge_context_force_interval_seconds", 12))))

    @staticmethod
    def _trim_label(value: object, limit: int = 80) -> str:
        text = collapse_spaces(str(value or ""))
        if len(text) <= limit:
            return text
        if limit <= 3:
            return text[:limit]
        return text[: limit - 3].rstrip() + "..."

    def _build_context_snapshot(self, state: _BridgeSessionState) -> dict[str, Any]:
        guild = self.client.get_guild(state.guild_id)
        voice_channel = guild.get_channel(state.voice_channel_id) if guild is not None else None
        text_channel = self.client.get_channel(state.text_channel_id)
        voice_members = list(getattr(voice_channel, "members", []) or [])

        participants: list[dict[str, Any]] = []
        humans = 0
        bots = 0
        for member in voice_members:
            is_bot = bool(getattr(member, "bot", False))
            if is_bot:
                bots += 1
            else:
                humans += 1
            participants.append(
                {
                    "user_id": str(getattr(member, "id", "") or ""),
                    "display_name": self._trim_label(getattr(member, "display_name", getattr(member, "name", "")), 48),
                    "bot": is_bot,
                    "is_bot": is_bot,
                    "self_mute": bool(getattr(getattr(member, "voice", None), "self_mute", False)),
                    "self_deaf": bool(getattr(getattr(member, "voice", None), "self_deaf", False)),
                    "mute": bool(getattr(getattr(member, "voice", None), "mute", False)),
                    "deaf": bool(getattr(getattr(member, "voice", None), "deaf", False)),
                }
            )
        participants.sort(key=lambda p: (bool(p.get("bot")), str(p.get("display_name") or "").casefold()))

        active_speaker_hints: list[str] = []
        with contextlib.suppress(Exception):
            active = list(getattr(state.room, "active_speakers", []) or [])
            for item in active[:8]:
                identity = self._trim_label(getattr(item, "identity", ""), 48)
                if identity:
                    active_speaker_hints.append(identity)

        remote_participants = getattr(state.room, "remote_participants", {}) or {}
        remote_values = remote_participants.values() if isinstance(remote_participants, dict) else []
        livekit_participants = []
        for participant in list(remote_values):
            identity = str(getattr(participant, "identity", "") or "")
            if identity == state.identity:
                continue
            livekit_participants.append(
                {
                    "identity": self._trim_label(identity, 64),
                    "sid": str(getattr(participant, "sid", "") or ""),
                }
            )

        return {
            "version": 1,
            "guild": {
                "id": str(state.guild_id),
                "name": self._trim_label(getattr(guild, "name", ""), 80),
            },
            "voice_channel": {
                "id": str(state.voice_channel_id),
                "name": self._trim_label(getattr(voice_channel, "name", ""), 80),
            },
            "text_channel": {
                "id": str(state.text_channel_id),
                "name": self._trim_label(getattr(text_channel, "name", ""), 80),
            },
            "participants": {
                "count": len(participants),
                "humans": humans,
                "bots": bots,
                "members": participants[:24],
                "active_speaker_hints": active_speaker_hints[:8],
            },
            "bridge_runtime": {
                "remote_streams": len(state.remote_tasks),
                "ingress_active": bool(state.discord_ingress_started_emitted),
                "dispatch_id": state.agent_dispatch_id or "",
                "dispatch_owned": bool(state.agent_dispatch_owned),
                "last_ingress_user_id": str(state.last_ingress_user_id or ""),
                "last_ingress_user_label": self._trim_label(state.last_ingress_user_label, 48),
                "last_ingress_unix": float(state.last_ingress_unix or 0.0),
            },
            "livekit_room": {
                "name": state.room_name,
                "remote_participants": livekit_participants[:24],
            },
            "context_sync": {
                "seq": int(state.context_sync_seq or 0),
                "last_reason": str(state.last_context_reason or ""),
                "last_sent_unix": float(state.last_context_sent_unix or 0.0),
                "last_error": str(state.last_context_publish_error or ""),
            },
        }

    async def _publish_context_snapshot(
        self,
        state: _BridgeSessionState,
        *,
        reason: str,
        force: bool = False,
        best_effort: bool = True,
    ) -> None:
        if not self._context_sync_enabled():
            return
        if state.stop_event.is_set():
            return
        state.context_publish_attempts += 1
        now = time.monotonic()
        now_unix = time.time()
        min_interval = self._context_min_interval_seconds()
        if (not force) and state.last_context_sent_at > 0 and (now - state.last_context_sent_at) < min_interval:
            state.context_publish_skipped_throttle += 1
            return

        snapshot = self._build_context_snapshot(state)
        payload_json = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        payload_hash = hashlib.sha1(payload_json.encode("utf-8", errors="ignore")).hexdigest()
        force_interval = self._context_force_interval_seconds()
        unchanged = bool(payload_hash and payload_hash == state.last_context_hash)
        if unchanged and (not force) and state.last_context_sent_at > 0 and (now - state.last_context_sent_at) < force_interval:
            state.context_publish_skipped_unchanged += 1
            return

        state.context_sync_seq += 1
        updated_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_unix))
        packet = {
            "event": "bridge_context_snapshot",
            "source": "discord-livekit-bridge",
            "version": 1,
            "seq": int(state.context_sync_seq),
            "guild_id": state.guild_id,
            "voice_channel_id": state.voice_channel_id,
            "text_channel_id": state.text_channel_id,
            "room_name": state.room_name,
            "reason": reason,
            "updated_at_unix": round(now_unix, 3),
            "updated_at_iso": updated_at_iso,
            "payload": snapshot,
        }
        try:
            packet_text = json.dumps(packet, ensure_ascii=False)
            await state.room.local_participant.publish_data(
                packet_text,
                reliable=True,
                topic=str(getattr(self.settings, "bridge_context_topic", "bridge-context") or "bridge-context"),
            )
            state.last_context_sent_at = now
            state.last_context_sent_unix = now_unix
            state.last_context_hash = payload_hash
            state.last_context_reason = reason
            state.last_context_publish_error = ""
            state.last_context_payload_bytes = len(packet_text.encode("utf-8", errors="ignore"))
            state.context_publish_success += 1
            if state.health is not None:
                state.health.mark_activity()
        except Exception as exc:
            state.last_context_publish_error = str(exc)
            state.context_publish_errors += 1
            if best_effort:
                if state.context_publish_errors == 1 or (state.context_publish_errors % 10) == 0:
                    logger.warning(
                        "[bridge.live] context publish failed guild=%s room=%s reason=%s errors=%s: %s",
                        state.guild_id,
                        state.room_name,
                        reason,
                        state.context_publish_errors,
                        exc,
                    )
            if not best_effort:
                logger.debug(
                    "[bridge.live] context publish_data failed guild=%s room=%s reason=%s: %s",
                    state.guild_id,
                    state.room_name,
                    reason,
                    exc,
                )

    async def _context_sync_loop(self, state: _BridgeSessionState) -> None:
        if not self._context_sync_enabled():
            return
        while not state.stop_event.is_set():
            try:
                await asyncio.sleep(max(1.0, self._context_force_interval_seconds() / 2.0))
                if state.stop_event.is_set():
                    break
                await self._publish_context_snapshot(
                    state,
                    reason="periodic",
                    force=False,
                    best_effort=True,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("[bridge.live] context sync loop error guild=%s: %s", state.guild_id, exc)

    def _schedule_context_snapshot(self, guild_id: int, *, reason: str, force: bool = False) -> None:
        if not self._context_sync_enabled():
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        def _run() -> None:
            state = self._sessions.get(guild_id)
            if state is None or state.stop_event.is_set():
                return
            asyncio.create_task(
                self._publish_context_snapshot(state, reason=reason, force=force, best_effort=True),
                name=f"lk-bridge-context-{guild_id}",
            )

        loop.call_soon_threadsafe(_run)

    async def _handle_data_packet_received(self, guild_id: int, packet: Any) -> None:
        state = self._sessions.get(guild_id)
        if state is None or state.stop_event.is_set():
            return
        try:
            topic = str(getattr(packet, "topic", "") or "").strip()
            expected_topic = str(getattr(self.settings, "bridge_context_topic", "bridge-context") or "bridge-context")
            if topic != expected_topic:
                return
            participant = getattr(packet, "participant", None)
            participant_identity = str(getattr(participant, "identity", "") or "")
            if participant_identity and participant_identity == state.identity:
                # Ignore our own bridge-published packets if SDK echoes them back.
                return
            if not participant_identity:
                state.agent_ctx_ack_invalid += 1
                return
            remote_participants = getattr(state.room, "remote_participants", {}) or {}
            known_remote_identities: set[str] = set()
            if isinstance(remote_participants, dict):
                known_remote_identities = {
                    str(getattr(p, "identity", "") or "")
                    for p in remote_participants.values()
                }
                if known_remote_identities and participant_identity not in known_remote_identities:
                    # Best-effort anti-spoof guard: require ack sender to exist among current remote participants.
                    state.agent_ctx_ack_invalid += 1
                    return
            if state.agent_ctx_sender_identity and participant_identity != state.agent_ctx_sender_identity:
                # Lock the ACK sender identity after the first accepted packet, but allow safe rebind if the old
                # sender disappeared (agent restart/reconnect) or ACKs have been stale for a while.
                old_sender_still_present = (
                    state.agent_ctx_sender_identity in known_remote_identities if known_remote_identities else False
                )
                ack_age_sec = (
                    (time.time() - float(state.agent_ctx_last_ack_unix))
                    if float(state.agent_ctx_last_ack_unix or 0.0) > 0.0
                    else None
                )
                if old_sender_still_present and (ack_age_sec is None or ack_age_sec < 20.0):
                    state.agent_ctx_ack_invalid += 1
                    return
                logger.info(
                    "[bridge.live] context ACK sender rebound guild=%s room=%s old=%s new=%s ack_age=%.1fs",
                    guild_id,
                    state.room_name,
                    state.agent_ctx_sender_identity or "?",
                    participant_identity or "?",
                    float(ack_age_sec or 0.0),
                )
                state.agent_ctx_sender_identity = participant_identity
            raw = getattr(packet, "data", b"")
            if isinstance(raw, bytes):
                if len(raw) <= 0 or len(raw) > 65536:
                    state.agent_ctx_ack_invalid += 1
                    return
                payload_text = raw.decode("utf-8", errors="ignore")
            else:
                payload_text = str(raw or "")
            if not payload_text:
                state.agent_ctx_ack_invalid += 1
                return
            payload_obj = json.loads(payload_text)
            if not isinstance(payload_obj, dict):
                state.agent_ctx_ack_invalid += 1
                return
            event = str(payload_obj.get("event") or "")
            if event != "agent_context_applied":
                return
            source_name = str(payload_obj.get("source") or "")
            if source_name != "livekit-agent-runtime":
                state.agent_ctx_ack_invalid += 1
                return
            packet_room_name = str(payload_obj.get("room_name") or "")
            if packet_room_name and packet_room_name != state.room_name:
                state.agent_ctx_ack_invalid += 1
                return
            if not state.agent_ctx_sender_identity:
                state.agent_ctx_sender_identity = participant_identity
            try:
                state.agent_ctx_updates_received = max(
                    int(state.agent_ctx_updates_received or 0),
                    int(payload_obj.get("received", state.agent_ctx_updates_received) or 0),
                )
            except Exception:
                pass
            try:
                state.agent_ctx_updates_applied = max(
                    int(state.agent_ctx_updates_applied or 0),
                    int(payload_obj.get("applied", state.agent_ctx_updates_applied) or 0),
                )
            except Exception:
                pass
            try:
                state.agent_ctx_updates_ignored = max(
                    int(state.agent_ctx_updates_ignored or 0),
                    int(payload_obj.get("ignored", state.agent_ctx_updates_ignored) or 0),
                )
            except Exception:
                pass
            try:
                state.agent_ctx_updates_errors = max(
                    int(state.agent_ctx_updates_errors or 0),
                    int(payload_obj.get("errors", state.agent_ctx_updates_errors) or 0),
                )
            except Exception:
                pass
            try:
                seq = int(payload_obj.get("seq", 0) or 0)
            except Exception:
                seq = 0
            if seq > 0:
                state.agent_ctx_last_seq = seq
            state.agent_ctx_last_reason = str(payload_obj.get("reason") or "")
            state.agent_ctx_last_error = str(payload_obj.get("last_error") or "")
            try:
                state.agent_ctx_last_chars = int(payload_obj.get("chars", 0) or 0)
            except Exception:
                state.agent_ctx_last_chars = 0
            try:
                state.agent_ctx_last_ack_unix = float(payload_obj.get("applied_at_unix", time.time()) or time.time())
            except Exception:
                state.agent_ctx_last_ack_unix = time.time()
            if state.health is not None:
                state.health.mark_activity()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state.agent_ctx_ack_invalid += 1
            logger.debug("[bridge.live] data packet parse error guild=%s: %s", guild_id, exc)

    def _attach_room_event_handlers(self, room: Any, guild_id: int) -> None:
        def _on_connected() -> None:
            logger.info("[bridge.live] room connected guild=%s", guild_id)
            self._schedule_status_event(guild_id, "room_connected")
            self._schedule_context_snapshot(guild_id, reason="room_connected", force=True)

        def _on_disconnected(reason: object) -> None:
            logger.info("[bridge.live] room disconnected guild=%s reason=%s", guild_id, reason)
            self._schedule_status_event(guild_id, "room_disconnected", {"reason": str(reason)})

        def _on_reconnecting() -> None:
            logger.warning("[bridge.live] room reconnecting guild=%s", guild_id)
            self._schedule_status_event(guild_id, "room_reconnecting")

        def _on_reconnected() -> None:
            logger.info("[bridge.live] room reconnected guild=%s", guild_id)
            self._schedule_status_event(guild_id, "room_reconnected")
            self._schedule_context_snapshot(guild_id, reason="room_reconnected", force=True)

        def _on_participant_connected(participant: Any) -> None:
            del participant
            self._schedule_context_snapshot(guild_id, reason="participant_connected")

        def _on_participant_disconnected(participant: Any) -> None:
            del participant
            self._schedule_context_snapshot(guild_id, reason="participant_disconnected")

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

        def _on_transcription_received(segments: Any, participant: Any, publication: Any) -> None:
            del publication
            loop = self._loop
            if loop is None or loop.is_closed():
                return
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    self._handle_transcription_received(guild_id, segments, participant),
                    name=f"lk-bridge-transcription-{guild_id}",
                )
            )

        def _on_data_received(packet: Any) -> None:
            loop = self._loop
            if loop is None or loop.is_closed():
                return
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    self._handle_data_packet_received(guild_id, packet),
                    name=f"lk-bridge-data-{guild_id}",
                )
            )

        room.on("connected", _on_connected)
        room.on("disconnected", _on_disconnected)
        room.on("reconnecting", _on_reconnecting)
        room.on("reconnected", _on_reconnected)
        room.on("track_subscribed", _on_track_subscribed)
        room.on("track_unsubscribed", _on_track_unsubscribed)
        room.on("transcription_received", _on_transcription_received)
        room.on("data_received", _on_data_received)
        room.on("participant_connected", _on_participant_connected)
        room.on("participant_disconnected", _on_participant_disconnected)

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
        self._schedule_context_snapshot(guild_id, reason="track_subscribed")

    async def _handle_transcription_received(self, guild_id: int, segments: Any, participant: Any) -> None:
        state = self._sessions.get(guild_id)
        if state is None or state.stop_event.is_set():
            return

        participant_identity = str(getattr(participant, "identity", "") or "")
        if participant_identity and participant_identity == state.identity:
            # This is the bridge participant itself (mixed Discord input). We prefer
            # per-user Discord-side STT for memory so speaker attribution is preserved.
            return

        final_chunks: list[str] = []
        for seg in list(segments or []):
            try:
                if not bool(getattr(seg, "final", False)):
                    continue
                seg_id = str(getattr(seg, "id", "") or "")
                if seg_id and seg_id in state.seen_transcription_segment_ids:
                    continue
                text = collapse_spaces(str(getattr(seg, "text", "") or ""))
                if not text:
                    continue
                if seg_id:
                    state.seen_transcription_segment_ids.add(seg_id)
                final_chunks.append(text)
            except Exception:
                continue

        if len(state.seen_transcription_segment_ids) > 800:
            state.seen_transcription_segment_ids.clear()

        joined = collapse_spaces(" ".join(final_chunks))
        if not joined:
            return

        # Persist assistant transcript in memory so summaries/context see both sides of the voice dialogue.
        cb = getattr(self.client, "on_native_audio_assistant_transcript", None)
        if callable(cb):
            with contextlib.suppress(Exception):
                await cb(
                    guild_id=guild_id,
                    text=joined,
                    source=f"livekit_transcription:{participant_identity or 'remote'}",
                )
        if state.health is not None:
            state.health.mark_activity()

    def _cancel_remote_task(self, guild_id: int, pub_sid: str) -> None:
        if not pub_sid:
            return
        state = self._sessions.get(guild_id)
        if state is None:
            return
        task = state.remote_tasks.pop(pub_sid, None)
        if task is not None:
            task.cancel()
        self._schedule_context_snapshot(guild_id, reason="track_unsubscribed")
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
                    await self._publish_context_snapshot(state, reason="discord_audio_ingress_active", force=True, best_effort=True)
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
