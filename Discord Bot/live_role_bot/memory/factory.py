from __future__ import annotations

import os
from pathlib import Path

from .store import MemoryStore


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def _resolve_backend() -> str:
    backend = _env("MEMORY_BACKEND", "sqlite").lower()
    if backend in {"sqlite", "postgres"}:
        return backend
    raise ValueError("MEMORY_BACKEND must be 'sqlite' or 'postgres'")


def build_memory_store(sqlite_path: Path) -> MemoryStore:
    backend = _resolve_backend()
    if backend == "sqlite":
        return MemoryStore(sqlite_path)

    postgres_dsn = _env("MEMORY_POSTGRES_DSN")
    if not postgres_dsn:
        raise ValueError("MEMORY_POSTGRES_DSN is required when MEMORY_BACKEND=postgres")

    # Phase 3 scaffold: keep app wiring backend-ready before implementing Postgres storage mixins.
    raise NotImplementedError(
        "Postgres memory backend is not implemented yet. "
        "Set MEMORY_BACKEND=sqlite for now."
    )

