from __future__ import annotations

import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live_role_bot.memory.extractor import MemoryExtractor  # noqa: E402
from live_role_bot.memory.store import MemoryStore  # noqa: E402


class _FakeLLM:
    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload if payload is not None else {"facts": []}
        self.calls: list[dict[str, object]] = []

    async def json_chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self.payload


def test_user_fact_extractor_heuristics_capture_name_and_age() -> None:
    extractor = MemoryExtractor(enabled=True, llm=_FakeLLM({"facts": []}), candidate_limit=6)

    result = asyncio.run(extractor.extract_user_facts("Мене звати Микола, мені 29 років.", "Ukrainian"))

    assert result is not None
    facts = {f.key: f for f in result.facts}
    assert "identity:name" in facts
    assert "identity:age" in facts
    assert "Микола" in facts["identity:name"].value
    assert facts["identity:age"].value == "29"


def test_persona_self_fact_extractor_uses_self_prompt_and_parses_claims() -> None:
    fake_llm = _FakeLLM(
        {
            "facts": [
                {
                    "key": "travel history germany",
                    "value": "I traveled to Germany two years ago",
                    "type": "episodic",
                    "confidence": 0.74,
                    "importance": 0.82,
                }
            ]
        }
    )
    extractor = MemoryExtractor(enabled=True, llm=fake_llm, candidate_limit=6)

    result = asyncio.run(
        extractor.extract_persona_self_facts(
            "Вчора я поїла сендвіч з рибою, а два роки тому їздила в Німеччину.",
            "Ukrainian",
        )
    )

    assert result is not None
    assert any("Germany" in fact.value for fact in result.facts)
    assert fake_llm.calls
    first_call_messages = fake_llm.calls[0]["messages"]
    assert isinstance(first_call_messages, list)
    assert "assistant/persona" in str(first_call_messages[1]["content"]).lower()


def test_fact_extractor_parses_about_target_directness_and_evidence_quote() -> None:
    fake_llm = _FakeLLM(
        {
            "facts": [
                {
                    "key": "night shifts",
                    "value": "works night shifts",
                    "type": "context",
                    "confidence": 0.69,
                    "importance": 0.71,
                    "about_target": "speaker",
                    "directness": "implied",
                    "evidence_quote": "знову нічна зміна",
                }
            ]
        }
    )
    extractor = MemoryExtractor(enabled=True, llm=fake_llm, candidate_limit=6)

    result = asyncio.run(
        extractor.extract_user_facts(
            "Та блін, знову нічна зміна.",
            "Ukrainian",
            dialogue_context=[
                {"role": "user", "author_label": "Denys", "content": "Та блін, знову нічна зміна."},
            ],
        )
    )

    assert result is not None
    assert result.facts
    fact = result.facts[0]
    assert fact.about_target == "self"
    assert fact.directness == "implicit"
    assert "нічна зміна" in fact.evidence_quote
    assert "Recent dialogue window" in str(fake_llm.calls[0]["messages"][1]["content"])


def test_extractor_returns_diagnostics_when_llm_json_invalid_and_heuristics_used() -> None:
    class _InvalidJsonLLM:
        async def json_chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            return None

    extractor = MemoryExtractor(enabled=True, llm=_InvalidJsonLLM(), candidate_limit=6)
    result = asyncio.run(extractor.extract_user_facts("Мене звати Микола.", "Ukrainian"))

    assert result is not None
    assert result.diagnostics is not None
    assert result.diagnostics.llm_attempted is True
    assert result.diagnostics.json_valid is False
    assert result.diagnostics.fallback_used is True
    assert any(f.key == "identity:name" for f in result.facts)


def test_sqlite_global_biography_summary_roundtrip_for_user_and_persona(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")

    async def scenario() -> None:
        await store.init()
        await store.upsert_global_biography_summary(
            subject_kind="user",
            subject_id="u1",
            summary_text="Микола, 29 років, любить RPG та короткі відповіді.",
            source_fact_count=3,
            last_source_guild_id="g1",
        )
        await store.upsert_global_biography_summary(
            subject_kind="persona",
            subject_id="liza",
            summary_text="Любить імпровізацію, пам'ятає свої вигадані історії як частину біографії.",
            source_fact_count=5,
            last_source_guild_id="g2",
        )

        user_row = await store.get_global_user_biography_summary("u1")
        persona_row = await store.get_persona_biography_summary("liza")

        assert user_row is not None
        assert persona_row is not None
        assert user_row["subject_kind"] == "user"
        assert persona_row["subject_kind"] == "persona"
        assert int(user_row["source_fact_count"]) == 3
        assert "біографії" in str(persona_row["summary_text"])

    asyncio.run(scenario())


def test_sqlite_memory_extractor_audit_run_roundtrip(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")

    async def scenario() -> None:
        await store.init()
        run_id = await store.record_memory_extractor_run(
            guild_id="g1",
            channel_id="c1",
            speaker_user_id="u1",
            fact_owner_kind="user",
            fact_owner_id="u1",
            speaker_role="user",
            modality="text",
            source="discord",
            backend_name="ollama",
            model_name="qwen2.5:7b-instruct",
            dry_run=True,
            llm_attempted=True,
            llm_ok=True,
            json_valid=True,
            fallback_used=False,
            latency_ms=123,
            candidate_count=2,
            accepted_count=1,
            saved_count=0,
            filtered_count=1,
            error_text="",
            candidates=[
                {
                    "fact_key": "identity:name",
                    "fact_value": "Микола",
                    "fact_type": "identity",
                    "about_target": "self",
                    "directness": "explicit",
                    "evidence_quote": "Мене звати Микола",
                    "confidence": 0.96,
                    "importance": 0.8,
                    "moderation_action": "accept",
                    "moderation_reason": "accepted",
                    "selected_for_apply": True,
                    "saved_to_memory": False,
                },
                {
                    "fact_key": "identity:age",
                    "fact_value": "29",
                    "fact_type": "identity",
                    "about_target": "self",
                    "directness": "implicit",
                    "evidence_quote": "вже другий рік до 30",
                    "confidence": 0.55,
                    "importance": 0.7,
                    "moderation_action": "reject",
                    "moderation_reason": "confidence_below_threshold",
                    "selected_for_apply": False,
                    "saved_to_memory": False,
                },
            ],
        )
        assert run_id > 0

        import sqlite3

        with sqlite3.connect(store.db_path) as conn:
            run_row = conn.execute(
                "SELECT dry_run, backend_name, model_name, candidate_count, accepted_count, saved_count FROM memory_extractor_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            candidate_rows = conn.execute(
                "SELECT moderation_action, saved_to_memory FROM memory_extractor_candidates WHERE run_id = ? ORDER BY candidate_row_id ASC",
                (run_id,),
            ).fetchall()
        assert run_row is not None
        assert int(run_row[0]) == 1
        assert str(run_row[1]) == "ollama"
        assert str(run_row[2]) == "qwen2.5:7b-instruct"
        assert int(run_row[3]) == 2
        assert int(run_row[4]) == 1
        assert int(run_row[5]) == 0
        assert candidate_rows == [("accept", 0), ("reject", 0)]

    asyncio.run(scenario())


def test_sqlite_implicit_fact_candidate_promotes_after_repeat(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")

    async def scenario() -> None:
        await store.init()
        await store.upsert_user_fact(
            guild_id="g1",
            user_id="u1",
            fact_key="context:night_shifts",
            fact_value="Працює в нічні зміни",
            fact_type="context",
            confidence=0.74,
            importance=0.72,
            message_id=None,
            extractor="test",
            about_target="self",
            directness="implicit",
            evidence_quote="знову нічна зміна",
        )
        first = await store.get_user_facts("g1", "u1", limit=10)
        assert first and first[0]["status"] == "candidate"
        assert first[0]["directness"] == "implicit"

        await store.upsert_user_fact(
            guild_id="g1",
            user_id="u1",
            fact_key="context:night_shifts",
            fact_value="Працює в нічні зміни",
            fact_type="context",
            confidence=0.75,
            importance=0.72,
            message_id=None,
            extractor="test",
            about_target="self",
            directness="implicit",
            evidence_quote="в мене знову зміна вночі",
        )
        second_local = await store.get_user_facts("g1", "u1", limit=10)
        second_global = await store.get_user_facts_global_by_user_id("u1", limit=10)

        assert second_local and second_global
        assert second_local[0]["status"] == "confirmed"
        assert second_global[0]["status"] == "confirmed"
        assert second_local[0]["about_target"] == "self"
        assert second_global[0]["directness"] == "implicit"
        assert int(second_local[0]["evidence_count"]) == 2

    asyncio.run(scenario())


def test_persona_self_facts_can_be_stored_in_synthetic_memory_subject(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    persona_subject = "persona::liza"

    async def scenario() -> None:
        await store.init()
        await store.upsert_user_fact(
            guild_id="g42",
            user_id=persona_subject,
            fact_key="episodic:ate_fish_sandwich",
            fact_value="Вчора їла сендвіч з рибою",
            fact_type="episodic",
            confidence=0.76,
            importance=0.81,
            message_id=None,
            extractor="test",
        )
        facts = await store.get_user_facts_global_by_user_id(persona_subject, limit=10)
        assert facts
        assert facts[0]["fact_value"] == "Вчора їла сендвіч з рибою"

    asyncio.run(scenario())
