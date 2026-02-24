from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from pathlib import Path
import site
import sys


def bootstrap_local_venv() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    paths = [
        os.path.join(root, ".venv", "Lib", "site-packages"),
        os.path.join(root, ".venv", "lib", py_dir, "site-packages"),
    ]
    for path in paths:
        if os.path.isdir(path) and path not in sys.path:
            site.addsitedir(path)


bootstrap_local_venv()

from .config import Settings
from .discord.client import LiveRoleDiscordBot
from .memory.extractor import MemoryExtractor
from .memory.store import MemoryStore
from .services.gemini_client import GeminiClient
from .services.local_stt import LocalSTT

logger = logging.getLogger("live_role_bot")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)
    logging.getLogger("discord.ext.voice_recv.gateway").setLevel(logging.WARNING)
    logging.getLogger("discord.ext.voice_recv.opus").setLevel(logging.ERROR)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)


def _is_tty(stream: object) -> bool:
    with contextlib.suppress(Exception):
        return bool(getattr(stream, "isatty")())
    return False


def _ensure_terminal_only() -> None:
    # Prevent accidental detached/background launches (pythonw, scheduler, service, etc).
    if os.getenv("LIVE_ROLE_BOT_ALLOW_NONINTERACTIVE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    if not (_is_tty(sys.stdin) and _is_tty(sys.stdout)):
        raise RuntimeError("This bot must be started from an interactive terminal.")


def _is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False

    if os.name == "nt":
        # Windows process aliveness check without external deps.
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        SYNCHRONIZE = 0x00100000
        STILL_ACTIVE = 259

        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE,
            False,
            pid,
        )
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            if not ok:
                return False
            return int(exit_code.value) == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _acquire_instance_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    if lock_path.exists():
        stale_pid = 0
        with contextlib.suppress(Exception):
            stale_pid = int(lock_path.read_text(encoding="utf-8").strip() or "0")
        if stale_pid > 0 and _is_process_alive(stale_pid):
            raise RuntimeError(f"Bot is already running (pid={stale_pid}). Stop it before starting a new one.")
        with contextlib.suppress(Exception):
            lock_path.unlink()

    lock_path.write_text(str(os.getpid()), encoding="utf-8")


def _release_instance_lock(lock_path: Path) -> None:
    with contextlib.suppress(Exception):
        if lock_path.exists():
            lock_path.unlink()


def build_bot(settings: Settings) -> LiveRoleDiscordBot:
    memory = MemoryStore(settings.sqlite_path)
    llm = GeminiClient(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        timeout_seconds=settings.gemini_timeout_seconds,
        temperature=settings.gemini_temperature,
        max_output_tokens=settings.gemini_max_output_tokens,
        base_url=settings.gemini_base_url,
    )
    local_stt_enabled = settings.local_stt_enabled and not settings.gemini_native_audio_enabled
    local_stt = LocalSTT(
        enabled=local_stt_enabled,
        model=settings.local_stt_model,
        fallback_model=settings.local_stt_fallback_model,
        device=settings.local_stt_device,
        compute_type=settings.local_stt_compute_type,
        language=settings.local_stt_language,
        max_audio_seconds=settings.local_stt_max_audio_seconds,
    )
    extractor = MemoryExtractor(
        enabled=settings.memory_enabled,
        llm=llm,
        candidate_limit=settings.memory_candidate_fact_limit,
    )
    return LiveRoleDiscordBot(
        settings=settings,
        memory=memory,
        llm=llm,
        memory_extractor=extractor,
        local_stt=local_stt,
    )


async def _run_bot(settings: Settings) -> None:
    bot = build_bot(settings)
    parent_pid = os.getppid()

    async def _parent_watchdog() -> None:
        if parent_pid <= 0:
            return
        while not bot.is_closed():
            await asyncio.sleep(1.5)
            if _is_process_alive(parent_pid):
                continue
            logger.warning("Parent terminal process ended (pid=%s). Shutting down bot.", parent_pid)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(bot.close(), timeout=10.0)
            return

    watchdog_task = asyncio.create_task(_parent_watchdog(), name="parent-watchdog")
    try:
        async with bot:
            await bot.start(settings.discord_token)
    finally:
        watchdog_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watchdog_task
        if not bot.is_closed():
            with contextlib.suppress(Exception):
                await asyncio.wait_for(bot.close(), timeout=10.0)


def main() -> None:
    configure_logging()
    _ensure_terminal_only()
    settings = Settings.from_env()
    settings.validate()
    lock_path = settings.sqlite_path.parent / "live_role_bot.pid"
    _acquire_instance_lock(lock_path)
    try:
        asyncio.run(_run_bot(settings))
    except KeyboardInterrupt:
        logger.info("Shutdown requested, exiting.")
    finally:
        _release_instance_lock(lock_path)
