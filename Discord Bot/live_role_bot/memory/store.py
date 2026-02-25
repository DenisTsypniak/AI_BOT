from __future__ import annotations

import aiosqlite

from .storage.facts import MemoryFactsMixin
from .storage.identity import MemoryIdentityMixin
from .storage.messages import MemoryMessagesMixin
from .storage.roles import MemoryRolesMixin
from .storage.schema import MemorySchemaMixin
from .storage.stt import MemorySttMixin
from .storage.summaries import MemorySummariesMixin


class MemoryStore(
    MemorySchemaMixin,
    MemoryIdentityMixin,
    MemoryRolesMixin,
    MemoryMessagesMixin,
    MemorySummariesMixin,
    MemorySttMixin,
    MemoryFactsMixin,
):
    """Persistent live-dialogue memory store with session history, summaries and fact evidence."""

    backend_name = "sqlite"

    async def ping(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("SELECT 1")
