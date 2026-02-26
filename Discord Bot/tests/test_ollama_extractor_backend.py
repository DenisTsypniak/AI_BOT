from __future__ import annotations

import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live_role_bot.services.ollama_extractor_backend import OllamaExtractorBackend  # noqa: E402


def test_ollama_extractor_backend_builds_structured_output_request_and_parses_json() -> None:
    backend = OllamaExtractorBackend(
        base_url="http://127.0.0.1:11434",
        model="qwen2.5:7b-instruct",
        timeout_seconds=30,
        temperature=0.2,
    )
    captured: dict[str, object] = {}

    async def fake_request(payload, *, retries=3):  # type: ignore[no-untyped-def]
        captured["payload"] = payload
        return {"message": {"content": "{\"facts\": []}"}}

    backend._request = fake_request  # type: ignore[method-assign]

    result = asyncio.run(
        backend.json_chat(
            messages=[
                {"role": "system", "content": "Return JSON only"},
                {"role": "user", "content": "Текст для аналізу"},
            ],
            schema_hint='{"type":"object","properties":{"facts":{"type":"array"}}}',
            temperature=0.12,
            max_output_tokens=256,
        )
    )

    assert result == {"facts": []}
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "qwen2.5:7b-instruct"
    assert payload["stream"] is False
    assert isinstance(payload["format"], dict)
    assert payload["options"]["temperature"] == 0.12
    assert payload["options"]["num_predict"] == 256

