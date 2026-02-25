from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import site
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


def _bootstrap_local_venv() -> None:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    py_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    paths = [
        os.path.join(root, ".venv", "Lib", "site-packages"),
        os.path.join(root, ".venv", "lib", py_dir, "site-packages"),
    ]
    for path in paths:
        if os.path.isdir(path) and path not in sys.path:
            site.addsitedir(path)


_bootstrap_local_venv()

from ...config import Settings
from ...discord.common import load_rp_canon
from ...prompts.voice import build_native_audio_system_instruction
from .bridge import DiscordLiveKitBridge, LiveKitBridgeConfig
from .config import LiveKitAgentSettings
from .observability import HealthHeartbeat, LiveKitRuntimeHealth

logger = logging.getLogger("live_role_bot.livekit")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("livekit").setLevel(logging.INFO)


@dataclass(slots=True)
class _PromptContext:
    system_core_prompt: str
    preferred_language: str
    role_name: str
    role_goal: str
    role_style: str
    role_constraints: str
    rp_canon_prompt: str


@dataclass(slots=True)
class _RuntimeDeps:
    lk_settings: "LiveKitAgentSettings"
    instructions: str
    google: Any
    silero: Any | None
    health: "LiveKitRuntimeHealth"


def _build_prompt_context(base_settings: Settings, lk_settings: LiveKitAgentSettings) -> _PromptContext:
    rp_canon = load_rp_canon(lk_settings.bot_history_json_path)
    return _PromptContext(
        system_core_prompt=base_settings.system_core_prompt,
        preferred_language=base_settings.preferred_response_language,
        role_name=base_settings.role_name,
        role_goal=base_settings.role_goal,
        role_style=base_settings.role_style,
        role_constraints=base_settings.role_constraints,
        rp_canon_prompt=rp_canon,
    )


def _build_livekit_instructions(ctx: _PromptContext) -> str:
    lines = [
        ctx.system_core_prompt.strip(),
        f"Role name: {ctx.role_name}",
        f"Role goal: {ctx.role_goal}",
        f"Role style: {ctx.role_style}",
        f"Role constraints: {ctx.role_constraints}",
        f"Primary language: {ctx.preferred_language}.",
        "Context: You are speaking in a realtime voice room. Keep replies speech-friendly and natural.",
        "Voice turn policy: default to 1-3 short sentences unless user explicitly asks for details.",
        "Voice turn policy: react first, then give one useful point, then optionally one short question.",
        "Voice turn policy: vary openers and avoid repeating the same phrasing across consecutive turns.",
        "Voice turn policy: if audio is unclear, ask for a repeat in a natural human way.",
    ]
    if ctx.rp_canon_prompt:
        lines.append("RP CANON (highest priority):\n" + ctx.rp_canon_prompt)

    base = "\n".join(part for part in lines if part).strip()
    return build_native_audio_system_instruction(base, ctx.preferred_language)


def _import_livekit_modules() -> tuple[Any, Any, Any]:
    try:
        from livekit import agents  # type: ignore
        from livekit.agents import Agent, AgentServer, AgentSession, room_io  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "LiveKit Agents dependencies are missing. Install with: pip install -r requirements-livekit.txt"
        ) from exc

    return agents, (Agent, AgentServer, AgentSession, room_io), None


def _import_livekit_plugins(use_silero_vad: bool) -> tuple[Any, Any | None]:
    try:
        from livekit.plugins import google  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "LiveKit Google plugin is unavailable. Install with: pip install -r requirements-livekit.txt"
        ) from exc

    silero = None
    if use_silero_vad:
        with contextlib.suppress(Exception):
            from livekit.plugins import silero as _silero  # type: ignore

            silero = _silero
        if silero is None:
            logger.warning("LIVEKIT_USE_SILERO_VAD=true, but silero plugin is unavailable. Using Gemini built-in VAD.")

    return google, silero


def _build_google_realtime_model(google: Any, lk_settings: LiveKitAgentSettings, instructions: str) -> Any:
    model_kwargs = {
        "model": lk_settings.google_realtime_model,
        "voice": lk_settings.voice,
        "temperature": lk_settings.temperature,
        "instructions": instructions,
    }
    if lk_settings.google_api_key:
        # Pass explicitly to avoid child-process env propagation quirks in LiveKit job runners.
        model_kwargs["api_key"] = lk_settings.google_api_key
    # LiveKit plugin versions expose Gemini realtime model under different namespaces.
    beta = getattr(google, "beta", None)
    if beta is not None and hasattr(beta, "realtime"):
        realtime_ns = getattr(beta, "realtime")
        if hasattr(realtime_ns, "RealtimeModel"):
            return realtime_ns.RealtimeModel(**model_kwargs)
    realtime_ns = getattr(google, "realtime", None)
    if realtime_ns is not None and hasattr(realtime_ns, "RealtimeModel"):
        return realtime_ns.RealtimeModel(**model_kwargs)
    raise RuntimeError("Unsupported livekit.plugins.google API: RealtimeModel not found")


def _validate_paths(lk_settings: LiveKitAgentSettings) -> None:
    path = lk_settings.bot_history_json_path
    if not path.exists():
        logger.warning("LiveKit role prompt file not found: %s (will use env role prompts only)", path)


@lru_cache(maxsize=1)
def _load_runtime_deps() -> _RuntimeDeps:
    # In LiveKit `dev` mode on Windows, worker jobs run in a spawned process.
    # Reconfigure logging here so child-process session logs are visible.
    configure_logging()
    base_settings = Settings.from_env()  # Discord token is not required for the LiveKit worker
    lk_settings = LiveKitAgentSettings.from_env(base_settings)
    lk_settings.validate()
    lk_settings.export_env()
    _validate_paths(lk_settings)

    prompt_ctx = _build_prompt_context(base_settings, lk_settings)
    instructions = _build_livekit_instructions(prompt_ctx)

    google, silero = _import_livekit_plugins(lk_settings.use_silero_vad)
    health = LiveKitRuntimeHealth(worker_name=lk_settings.worker_name, agent_name=lk_settings.agent_name)
    return _RuntimeDeps(
        lk_settings=lk_settings,
        instructions=instructions,
        google=google,
        silero=silero,
        health=health,
    )


class _LizaVoiceAgent:  # runtime base class will be created dynamically per process
    pass


async def _handle_session(ctx: Any) -> None:
    configure_logging()
    deps = _load_runtime_deps()
    lk_settings = deps.lk_settings
    health = deps.health

    _, agent_types, _ = _import_livekit_modules()
    Agent, _, AgentSession, room_io = agent_types

    # Build a small concrete Agent subclass at runtime with current instructions.
    class LizaVoiceAgent(Agent):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__(instructions=deps.instructions)

    room_name = "-"
    session: Any = None
    hb: HealthHeartbeat | None = None
    try:
        with contextlib.suppress(Exception):
            room_name = str(getattr(getattr(ctx, "room", None), "name", "-") or "-")
        logger.info("[livekit.room] job received room=%s", room_name)

        await ctx.connect()
        with contextlib.suppress(Exception):
            room_name = str(getattr(ctx.room, "name", "-") or "-")
        health.room_name = room_name
        health.rooms_started += 1
        health.mark_activity()
        logger.info("[livekit.room] connected room=%s", room_name)

        hb = HealthHeartbeat(health, lk_settings.health_log_interval_seconds)
        hb.start()

        session_kwargs: dict[str, Any] = {
            "llm": _build_google_realtime_model(deps.google, lk_settings, deps.instructions)
        }
        if deps.silero is not None:
            with contextlib.suppress(Exception):
                session_kwargs["vad"] = deps.silero.VAD.load()

        session = AgentSession(**session_kwargs)
        room_input_kwargs: dict[str, Any] = {}
        if lk_settings.auto_subscribe_audio_only:
            with contextlib.suppress(Exception):
                room_input_kwargs["audio_enabled"] = True
                room_input_kwargs["video_enabled"] = False

        session_closed = asyncio.Event()
        job_shutdown = asyncio.Event()

        def _on_session_close(*_: Any) -> None:
            session_closed.set()

        async def _on_job_shutdown(*_: Any) -> None:
            job_shutdown.set()

        with contextlib.suppress(Exception):
            session.on("close", _on_session_close)
        with contextlib.suppress(Exception):
            ctx.add_shutdown_callback(_on_job_shutdown)

        try:
            wait_job_task: asyncio.Task[bool] | None = None
            wait_close_task: asyncio.Task[bool] | None = None
            await session.start(
                agent=LizaVoiceAgent(),
                room=ctx.room,
                room_input_options=room_io.RoomInputOptions(**room_input_kwargs),
            )
            health.mark_activity()
            logger.info("[livekit.room] session started room=%s", room_name)
            wait_job_task = asyncio.create_task(job_shutdown.wait(), name=f"lk-job-shutdown-{room_name}")
            wait_close_task = asyncio.create_task(session_closed.wait(), name=f"lk-session-close-{room_name}")
            try:
                done, pending = await asyncio.wait(
                    {wait_job_task, wait_close_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                reason = "job_shutdown" if wait_job_task in done else "session_closed"
                logger.info("[livekit.room] session wait completed room=%s reason=%s", room_name, reason)
            finally:
                if wait_job_task is not None:
                    with contextlib.suppress(Exception):
                        wait_job_task.cancel()
                if wait_close_task is not None:
                    with contextlib.suppress(Exception):
                        wait_close_task.cancel()
        finally:
            with contextlib.suppress(Exception):
                if session is not None:
                    await session.aclose()
            if hb is not None:
                await hb.stop()
            health.rooms_closed += 1
            health.mark_activity()
            logger.info("[livekit.room] session closed room=%s", room_name)
    except Exception:
        logger.exception("[livekit.room] session error room=%s", room_name)
        raise


def main() -> None:
    configure_logging()
    deps = _load_runtime_deps()
    lk_settings = deps.lk_settings
    instructions = deps.instructions

    logger.info(
        "[livekit.start] worker=%s agent=%s url=%s model=%s voice=%s bridge_enabled=%s",
        lk_settings.worker_name,
        lk_settings.agent_name,
        lk_settings.url,
        lk_settings.google_realtime_model,
        lk_settings.voice,
        lk_settings.bridge_enabled,
    )
    logger.info("[livekit.start] instructions chars=%s", len(instructions))

    if lk_settings.bridge_enabled:
        bridge = DiscordLiveKitBridge(
            LiveKitBridgeConfig(
                enabled=True,
                room_prefix=lk_settings.room_prefix,
                control_channel_name=lk_settings.bridge_control_channel,
            )
        )
        logger.info(
            "[bridge.plan] enabled scaffold control_channel=%s room_prefix=%s",
            bridge.config.control_channel_name,
            bridge.config.room_prefix,
        )

    agents, agent_types, _ = _import_livekit_modules()
    _, AgentServer, _, _ = agent_types

    server = AgentServer()

    server.rtc_session(agent_name=lk_settings.agent_name)(_handle_session)  # type: ignore[misc]

    agents.cli.run_app(server)
