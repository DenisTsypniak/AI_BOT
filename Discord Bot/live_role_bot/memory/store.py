from __future__ import annotations

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

