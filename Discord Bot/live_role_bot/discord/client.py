from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import defaultdict

import discord

from ..config import Settings
from ..memory.extractor import MemoryExtractor
from ..memory.store import MemoryStore
from ..persona import PersonaGrowthEngine
from ..services.gemini_client import GeminiClient
from ..services.livekit.bridge import LiveKitDiscordBridgeManager
from ..services.livekit.config import LiveKitAgentSettings
from ..services.native_audio import GeminiNativeAudioManager
from ..services.local_stt import LocalSTT
from .common import (
    ConversationSessionState,
    PendingProfileUpdate,
    PendingSummaryUpdate,
    PendingVoiceTurn,
    VoiceTurnBuffer,
    load_rp_canon,
)
from .mixins.dialogue_mixin import DialogueMixin
from .mixins.identity_mixin import IdentityMixin
from .mixins.workers_voice_mixin import WorkersVoiceMixin
from .voice_integration import VOICE_RECV_AVAILABLE, install_voice_recv_decode_guard, voice_recv
try:
    from ..plugins.manager import PluginManager
except ModuleNotFoundError:
    PluginManager = None  # type: ignore[assignment]

logger = logging.getLogger("live_role_bot")


class LiveRoleDiscordBot(
    DialogueMixin,
    IdentityMixin,
    WorkersVoiceMixin,
    discord.Client,
):
    def __init__(
        self,
        settings: Settings,
        memory: MemoryStore,
        llm: GeminiClient,
        memory_extractor: MemoryExtractor,
        local_stt: LocalSTT,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = settings.discord_message_content_intent
        intents.members = settings.discord_members_intent
        intents.voice_states = True

        super().__init__(intents=intents)

        self.settings = settings
        self.memory = memory
        self.llm = llm
        self.memory_extractor = memory_extractor
        self.local_stt = local_stt
        self.rp_canon_prompt = self._load_bot_history_prompt()
        self.persona_engine = PersonaGrowthEngine(
            settings=settings,
            memory=memory,
            core_dna_path=settings.bot_history_json_path,
            llm=llm,
        )

        self.plugin_manager: PluginManager | None = None
        if self.settings.plugins_enabled and PluginManager is not None:
            self.plugin_manager = PluginManager(match_threshold=self.settings.plugins_match_threshold)
        elif self.settings.plugins_enabled and PluginManager is None:
            logger.warning("Plugins are enabled in config, but live_role_bot.plugins is missing. Plugins disabled.")

        self.channel_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

        self.profile_queue: asyncio.Queue[PendingProfileUpdate] = asyncio.Queue(maxsize=260)
        self.persona_event_queue: asyncio.Queue[PendingProfileUpdate] = asyncio.Queue(
            maxsize=max(20, int(getattr(settings, "persona_event_queue_maxsize", 260)))
        )
        self.summary_queue: asyncio.Queue[PendingSummaryUpdate] = asyncio.Queue(maxsize=220)
        self.voice_turn_queue: asyncio.Queue[PendingVoiceTurn] = asyncio.Queue(maxsize=140)

        self.profile_worker_task: asyncio.Task[None] | None = None
        self.persona_ingest_task: asyncio.Task[None] | None = None
        self.summary_worker_task: asyncio.Task[None] | None = None
        self.voice_worker_task: asyncio.Task[None] | None = None
        self.voice_flush_task: asyncio.Task[None] | None = None
        self.persona_reflection_task: asyncio.Task[None] | None = None
        self.persona_decay_task: asyncio.Task[None] | None = None

        self.summary_pending_keys: set[tuple[str, str, str]] = set()
        self.voice_buffers: dict[tuple[int, int], VoiceTurnBuffer] = {}
        self.voice_text_channels: dict[int, int] = {}
        self._seen_voice_pcm_users: set[tuple[int, int]] = set()
        self._native_user_transcripts: dict[tuple[int, int, str], float] = {}
        self._native_assistant_transcripts: dict[tuple[int, str], float] = {}
        self._native_transcript_dedupe_seconds = 6.0
        self.conversation_states: dict[str, ConversationSessionState] = {}
        self.recent_assistant_replies: dict[str, list[str]] = {}

        self.native_audio: GeminiNativeAudioManager | None = None
        if self.settings.gemini_native_audio_enabled:
            try:
                self.native_audio = GeminiNativeAudioManager(self, settings)
            except Exception as exc:
                logger.error("Failed to initialize Gemini Native Audio manager: %s", exc)
                self.native_audio = None
        self.livekit_bridge: LiveKitDiscordBridgeManager | None = None
        with contextlib.suppress(Exception):
            lk_settings = LiveKitAgentSettings.from_env(settings)
            if lk_settings.bridge_enabled:
                try:
                    lk_settings.validate_bridge()
                    self.livekit_bridge = LiveKitDiscordBridgeManager(self, lk_settings)
                    logger.info(
                        "LiveKit bridge enabled (room_prefix=%s url=%s)",
                        lk_settings.room_prefix,
                        lk_settings.url,
                    )
                except Exception as exc:
                    logger.error("Failed to initialize LiveKit bridge manager: %s", exc)
        self.local_voice_fallback_enabled = self.settings.local_stt_enabled and not self.settings.gemini_native_audio_enabled
        local_stt_runtime_enabled = bool(getattr(self.local_stt, "enabled", False))
        self.bridge_voice_memory_transcript_enabled = bool(
            local_stt_runtime_enabled
            and self.settings.memory_enabled
            and self.settings.voice_bridge_memory_stt_enabled
            and self.livekit_bridge is not None
        )
        self.local_voice_ingest_enabled = bool(
            self.local_voice_fallback_enabled or self.bridge_voice_memory_transcript_enabled
        )
        install_voice_recv_decode_guard()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._started_at_monotonic = time.monotonic()

    def _load_bot_history_prompt(self) -> str:
        path = self.settings.bot_history_json_path
        prompt = load_rp_canon(path)
        if prompt:
            logger.info("Loaded RP canon from %s (%s chars)", path, len(prompt))
            return prompt
        if not path.exists():
            raise RuntimeError(f"bot_history.json not found at {path}. This bot requires bot_history.json as the character source.")
        raise RuntimeError(
            f"Failed to load usable RP canon from {path}. "
            "Ensure JSON is valid, enabled=true, and contains non-empty system_prompt/style_persona_prompt."
        )

    async def setup_hook(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self.memory.init()
        await self.llm.start()
        await self.memory_extractor.start()
        await self.persona_engine.start()

        if self.plugin_manager is not None:
            self.plugin_manager.load_plugins()

        await self.memory.ensure_role_profile(
            role_id=self.settings.default_role_id,
            name=self.settings.role_name,
            goal=self.settings.role_goal,
            style=self.settings.role_style,
            constraints=self.settings.role_constraints,
        )

        self.profile_worker_task = asyncio.create_task(self._profile_worker(), name="profile-worker")
        if bool(getattr(self.settings, "persona_queue_isolation_enabled", False)) and bool(
            getattr(self.persona_engine, "enabled", False)
        ):
            self.persona_ingest_task = asyncio.create_task(
                self._persona_ingest_worker(),
                name="persona-ingest-worker",
            )
        self.summary_worker_task = asyncio.create_task(self._summary_worker(), name="summary-worker")
        self.persona_reflection_task = asyncio.create_task(
            self._persona_reflection_loop(),
            name="persona-reflection-worker",
        )
        self.persona_decay_task = asyncio.create_task(
            self._persona_decay_loop(),
            name="persona-decay-worker",
        )
        if self.local_voice_ingest_enabled:
            self.voice_worker_task = asyncio.create_task(self._voice_worker(), name="voice-worker")
            self.voice_flush_task = asyncio.create_task(self._voice_flush_loop(), name="voice-flush")

    async def close(self) -> None:
        await self._run_shutdown_step("disconnect_voice_clients", self._disconnect_voice_clients(), timeout=6.0)
        if self.livekit_bridge is not None:
            await self._run_shutdown_step("livekit_bridge.shutdown_all", self.livekit_bridge.shutdown_all(), timeout=3.0)
        if self.native_audio is not None:
            await self._run_shutdown_step("native_audio.shutdown_all", self.native_audio.shutdown_all(), timeout=3.0)

        await self._cancel_task(self.profile_worker_task)
        await self._cancel_task(self.persona_ingest_task)
        await self._cancel_task(self.summary_worker_task)
        await self._cancel_task(self.voice_worker_task)
        await self._cancel_task(self.voice_flush_task)
        await self._cancel_task(self.persona_reflection_task)
        await self._cancel_task(self.persona_decay_task)

        await self._run_shutdown_step("persona_engine.close", self.persona_engine.close(), timeout=6.0)
        await self._run_shutdown_step("memory_extractor.close", self.memory_extractor.close(), timeout=6.0)
        await self._run_shutdown_step("llm.close", self.llm.close(), timeout=6.0)
        memory_close = getattr(self.memory, "close", None)
        if callable(memory_close):
            await self._run_shutdown_step("memory.close", memory_close(), timeout=6.0)
        await self._run_shutdown_step("discord.Client.close", super().close(), timeout=6.0)

    async def _run_shutdown_step(self, label: str, coro: object, *, timeout: float) -> None:
        try:
            await asyncio.wait_for(coro, timeout=timeout)  # type: ignore[arg-type]
        except asyncio.TimeoutError:
            logger.warning("Shutdown step timed out: %s", label)
        except Exception as exc:
            logger.warning("Shutdown step failed: %s (%s)", label, exc)

    async def _cancel_task(self, task: asyncio.Task[None] | None) -> None:
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _disconnect_voice_clients(self) -> None:
        for guild in list(self.guilds):
            vc = guild.voice_client
            if vc is None:
                continue
            if VOICE_RECV_AVAILABLE and voice_recv is not None:
                with contextlib.suppress(Exception):
                    if isinstance(vc, voice_recv.VoiceRecvClient) and vc.is_listening():
                        vc.stop_listening()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(vc.disconnect(force=True), timeout=4.0)

    async def on_ready(self) -> None:
        if self.user:
            logger.info("Connected as %s (%s)", self.user, self.user.id)

    @staticmethod
    def _task_alive(task: asyncio.Task[None] | None) -> bool:
        return bool(task is not None and not task.done())

    async def build_status_snapshot(self) -> dict[str, object]:
        snapshot: dict[str, object] = {
            "uptime_sec": max(0.0, time.monotonic() - self._started_at_monotonic),
            "user": str(self.user) if self.user else "",
            "guilds": len(self.guilds),
            "memory_backend": str(getattr(self.memory, "backend_name", type(self.memory).__name__)),
            "memory_ping_ok": False,
            "memory_ping_error": "",
            "queues": {
                "profile": int(self.profile_queue.qsize()),
                "persona_event": int(self.persona_event_queue.qsize()),
                "summary": int(self.summary_queue.qsize()),
                "voice_turn": int(self.voice_turn_queue.qsize()),
            },
            "workers": {
                "profile": self._task_alive(self.profile_worker_task),
                "persona_ingest": self._task_alive(self.persona_ingest_task),
                "summary": self._task_alive(self.summary_worker_task),
                "voice": self._task_alive(self.voice_worker_task),
                "voice_flush": self._task_alive(self.voice_flush_task),
                "persona_reflection": self._task_alive(self.persona_reflection_task),
                "persona_decay": self._task_alive(self.persona_decay_task),
            },
            "voice_clients": [],
            "livekit_bridge": {"enabled": False},
            "native_audio": {"enabled": False},
            "plugins": {
                "enabled": bool(self.settings.plugins_enabled),
                "loaded": bool(self.plugin_manager is not None),
            },
            "voice_ingest": {
                "local_fallback_enabled": bool(self.local_voice_fallback_enabled),
                "bridge_memory_stt_enabled": bool(self.bridge_voice_memory_transcript_enabled),
                "worker_enabled": bool(self.local_voice_ingest_enabled),
            },
            "voice_memory_diag": (
                self._voice_memory_diag_snapshot() if callable(getattr(self, "_voice_memory_diag_snapshot", None)) else {}
            ),
            "persona_queue_diag": (
                self._persona_queue_diag_snapshot() if callable(getattr(self, "_persona_queue_diag_snapshot", None)) else {}
            ),
            "persona_growth": self.persona_engine.status_snapshot(),
        }

        ping_fn = getattr(self.memory, "ping", None)
        if callable(ping_fn):
            try:
                await asyncio.wait_for(ping_fn(), timeout=2.5)
                snapshot["memory_ping_ok"] = True
            except Exception as exc:
                snapshot["memory_ping_error"] = str(exc)

        voice_clients: list[dict[str, object]] = []
        for guild in self.guilds:
            vc = guild.voice_client
            if vc is None or getattr(vc, "channel", None) is None:
                continue
            channel = getattr(vc, "channel", None)
            voice_clients.append(
                {
                    "guild_id": int(guild.id),
                    "guild_name": str(guild.name),
                    "channel_id": int(getattr(channel, "id", 0) or 0),
                    "channel_name": str(getattr(channel, "name", "?")),
                    "connected": bool(getattr(vc, "is_connected", lambda: False)()),
                    "playing": bool(getattr(vc, "is_playing", lambda: False)()),
                }
            )
        snapshot["voice_clients"] = voice_clients

        if self.livekit_bridge is not None:
            with contextlib.suppress(Exception):
                snapshot["livekit_bridge"] = self.livekit_bridge.status_snapshot()

        if self.native_audio is not None:
            with contextlib.suppress(Exception):
                snapshot["native_audio"] = self.native_audio.status_snapshot()

        return snapshot
