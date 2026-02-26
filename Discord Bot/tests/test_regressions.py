from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path

import pytest
from starlette.routing import Match


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db_viewer.app import _normalize_paging, app  # noqa: E402
from live_role_bot.memory.storage.schema import MemorySchemaMixin  # noqa: E402


def test_normalize_paging_clamps_invalid_values() -> None:
    assert _normalize_paging(-5, -10, default_limit=50, max_limit=200) == (50, 0)
    assert _normalize_paging(99999, 3, default_limit=50, max_limit=200) == (200, 3)


def test_messages_csv_route_is_not_captured_by_message_detail_route() -> None:
    csv_route = next(r for r in app.routes if getattr(r, "path", "") == "/messages/csv")
    detail_route = next(r for r in app.routes if getattr(r, "path", "") == "/messages/{message_id:int}")

    scope = {"type": "http", "path": "/messages/csv", "method": "GET", "headers": []}
    csv_match, _ = csv_route.matches(scope)
    detail_match, _ = detail_route.matches(scope)

    assert csv_match == Match.FULL
    assert detail_match == Match.NONE


def test_schema_mismatch_raises_without_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMORY_SQLITE_RESET_ON_SCHEMA_MISMATCH", raising=False)
    db_path = tmp_path / "memory.db"

    asyncio.run(MemorySchemaMixin(db_path).init())
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA user_version = 999")
        conn.commit()

    with pytest.raises(RuntimeError, match="schema version mismatch"):
        asyncio.run(MemorySchemaMixin(db_path).init())


def test_schema_mismatch_can_reset_with_explicit_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "memory.db"
    asyncio.run(MemorySchemaMixin(db_path).init())

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA user_version = 999")
        conn.commit()

    monkeypatch.setenv("MEMORY_SQLITE_RESET_ON_SCHEMA_MISMATCH", "1")
    asyncio.run(MemorySchemaMixin(db_path).init())

    with sqlite3.connect(db_path) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == MemorySchemaMixin.SCHEMA_VERSION
