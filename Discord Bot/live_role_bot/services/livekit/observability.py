from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("live_role_bot.livekit")


@dataclass(slots=True)
class LiveKitRuntimeHealth:
    worker_name: str
    agent_name: str
    room_name: str | None = None
    last_activity_at: float = field(default_factory=time.monotonic)
    rooms_started: int = 0
    rooms_closed: int = 0

    def mark_activity(self) -> None:
        self.last_activity_at = time.monotonic()


class HealthHeartbeat:
    def __init__(self, state: LiveKitRuntimeHealth, interval_seconds: int) -> None:
        self.state = state
        self.interval_seconds = max(5, int(interval_seconds))
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="livekit-health-heartbeat")

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.interval_seconds)
            idle = time.monotonic() - self.state.last_activity_at
            logger.debug(
                "[livekit.health] worker=%s agent=%s room=%s rooms_started=%s rooms_closed=%s idle_sec=%.1f",
                self.state.worker_name,
                self.state.agent_name,
                self.state.room_name or "-",
                self.state.rooms_started,
                self.state.rooms_closed,
                idle,
            )
