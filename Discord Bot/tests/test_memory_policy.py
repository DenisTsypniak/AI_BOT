from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live_role_bot.memory.storage.utils import merge_memory_fact_state  # noqa: E402
from live_role_bot.memory.store import MemoryStore  # noqa: E402


class _PromptSettings:
    def __init__(self, *, allow_cross_server_summary: bool = False) -> None:
        self.memory_cross_server_dialogue_summary_fallback_enabled = allow_cross_server_summary


class _FakeMemoryForSummary:
    def __init__(self) -> None:
        self.local_calls = 0
        self.global_calls = 0

    async def get_dialogue_summary(self, guild_id: str, user_id: str, channel_id: str) -> dict[str, object] | None:
        self.local_calls += 1
        return None

    async def get_latest_dialogue_summary_by_user_id(
        self,
        user_id: str,
        *,
        exclude_guild_id: str | None = None,
    ) -> dict[str, object] | None:
        self.global_calls += 1
        return {
            "guild_id": "other-guild",
            "channel_id": "other-channel",
            "summary_text": "Cross-server scene summary",
            "source_user_messages": 9,
            "last_message_id": 123,
            "updated_at": "2026-02-25T12:00:00Z",
        }


class _FakeMemoryForFacts:
    async def get_user_facts(self, guild_id: str, user_id: str, limit: int) -> list[dict[str, object]]:
        return [
            {
                "fact_key": "identity:name",
                "fact_value": "Denys",
                "fact_type": "identity",
                "confidence": 0.95,
                "importance": 0.8,
                "status": "confirmed",
                "pinned": False,
                "evidence_count": 3,
            }
        ]

    async def get_user_facts_global_by_user_id(
        self,
        user_id: str,
        limit: int,
        *,
        exclude_guild_id: str | None = None,
    ) -> list[dict[str, object]]:
        raise RuntimeError("boom")


def _build_prompt_mixin_subject(memory: object, *, allow_cross_server_summary: bool = False):
    pytest.importorskip("discord")
    from live_role_bot.discord.mixins.prompt_mixin import PromptMixin  # noqa: WPS433,E402

    class _Subject(PromptMixin):
        def __init__(self, memory_obj: object) -> None:
            self.memory = memory_obj
            self.settings = _PromptSettings(allow_cross_server_summary=allow_cross_server_summary)

    return _Subject(memory)


def test_merge_memory_fact_state_keeps_strong_existing_value_on_weak_conflict() -> None:
    merged = merge_memory_fact_state(
        prior_value="Любить Terraria",
        prior_fact_type="preference",
        prior_confidence=0.94,
        prior_importance=0.80,
        prior_evidence_count=3,
        prior_status="confirmed",
        pinned=False,
        incoming_value="Любить Minecraft",
        incoming_fact_type="preference",
        incoming_confidence=0.31,
        incoming_importance=0.55,
        value_max_chars=280,
    )

    assert merged.fact_value == "Любить Terraria"
    assert merged.value_conflict is True
    assert merged.value_replaced is False
    assert 0.55 <= merged.confidence < 0.94


def test_merge_memory_fact_state_replaces_weak_existing_value_on_strong_conflict() -> None:
    merged = merge_memory_fact_state(
        prior_value="Звати Олег",
        prior_fact_type="identity",
        prior_confidence=0.36,
        prior_importance=0.80,
        prior_evidence_count=1,
        prior_status="candidate",
        pinned=False,
        incoming_value="Звати Денис",
        incoming_fact_type="identity",
        incoming_confidence=0.93,
        incoming_importance=0.90,
        value_max_chars=280,
    )

    assert merged.fact_value == "Звати Денис"
    assert merged.value_conflict is True
    assert merged.value_replaced is True
    assert merged.confidence >= 0.90


def test_prompt_summary_does_not_use_cross_server_fallback_by_default() -> None:
    subject = _build_prompt_mixin_subject(_FakeMemoryForSummary(), allow_cross_server_summary=False)

    summary = asyncio.run(subject._get_summary_with_global_fallback("g1", "u1", "c1"))

    assert summary is None
    assert subject.memory.local_calls == 1
    assert subject.memory.global_calls == 0


def test_prompt_summary_can_opt_in_cross_server_fallback() -> None:
    subject = _build_prompt_mixin_subject(_FakeMemoryForSummary(), allow_cross_server_summary=True)

    summary = asyncio.run(subject._get_summary_with_global_fallback("g1", "u1", "c1"))

    assert summary is not None
    assert summary["summary_text"] == "Cross-server scene summary"
    assert summary["memory_scope"] == "cross_server_dialogue_summary_fallback"
    assert subject.memory.global_calls == 1


def test_prompt_global_facts_fallback_logs_and_returns_local_memory(caplog: pytest.LogCaptureFixture) -> None:
    subject = _build_prompt_mixin_subject(_FakeMemoryForFacts())

    with caplog.at_level(logging.ERROR, logger="live_role_bot"):
        facts = asyncio.run(
            subject._get_user_facts_with_global_fallback(
                "guild-a",
                "user-a",
                local_limit=4,
                target_count=4,
            )
        )

    assert facts
    assert facts[0]["fact_value"] == "Denys"
    assert facts[0]["memory_scope"] == "guild_local_memory"
    assert "Global facts fallback retrieval failed" in caplog.text


def test_sqlite_fact_conflict_does_not_overwrite_strong_value(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path)

    async def scenario() -> None:
        await store.init()
        await store.upsert_user_fact("g1", "u1", "pref:game", "Terraria", "preference", 0.95, 0.7, None, "test")
        await store.upsert_user_fact("g1", "u1", "pref:game", "Minecraft", "preference", 0.28, 0.6, None, "test")

        local_facts = await store.get_user_facts("g1", "u1", limit=10)
        global_facts = await store.get_user_facts_global_by_user_id("u1", limit=10)

        assert local_facts[0]["fact_value"] == "Terraria"
        assert global_facts[0]["fact_value"] == "Terraria"
        assert int(local_facts[0]["evidence_count"]) == 2
        assert int(global_facts[0]["evidence_count"]) == 2

    asyncio.run(scenario())


def test_sqlite_global_facts_exclude_guild_uses_cross_guild_evidence(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path)

    async def scenario() -> None:
        await store.init()
        await store.upsert_user_fact("guild-b", "u1", "pref:genre", "RPG", "preference", 0.87, 0.7, None, "test")
        await store.upsert_user_fact("guild-a", "u1", "pref:genre", "RPG", "preference", 0.88, 0.7, None, "test")
        await store.upsert_user_fact("guild-a", "u1", "pref:pet", "cat", "fact", 0.81, 0.6, None, "test")

        facts = await store.get_user_facts_global_by_user_id("u1", limit=10, exclude_guild_id="guild-a")
        keys = {str(item["fact_key"]) for item in facts}

        assert "pref:genre" in keys
        assert "pref:pet" not in keys

    asyncio.run(scenario())

