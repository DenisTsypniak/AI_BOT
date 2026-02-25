from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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


def build_memory_store(sqlite_path: Path) -> Any:
    backend = _resolve_backend()
    if backend == "sqlite":
        return MemoryStore(sqlite_path)

    postgres_dsn = _env("MEMORY_POSTGRES_DSN")
    if not postgres_dsn:
        raise ValueError("MEMORY_POSTGRES_DSN is required when MEMORY_BACKEND=postgres")

    from .postgres_store import PostgresMemoryStore

    return PostgresMemoryStore(postgres_dsn)
