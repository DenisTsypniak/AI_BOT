from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Any

import aiohttp


class OllamaExtractorBackend:
    backend_name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: int = 45,
        temperature: float = 0.1,
    ) -> None:
        self.base_url = (base_url or "http://127.0.0.1:11434").strip().rstrip("/")
        self.model = (model or "").strip()
        if not self.model:
            raise ValueError("Ollama extractor model cannot be empty")
        self.timeout = aiohttp.ClientTimeout(total=max(5, int(timeout_seconds)))
        self.temperature = float(temperature)
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    def _endpoint(self) -> str:
        return f"{self.base_url}/api/chat"

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        cleaned = (text or "").strip()
        cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        if not cleaned.startswith("{"):
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                cleaned = cleaned[start : end + 1].strip()
        return cleaned

    @staticmethod
    def _looks_like_json_schema(parsed: dict[str, Any]) -> bool:
        # Ollama `format` expects a JSON Schema object, not an example JSON payload.
        schema_markers = {
            "type",
            "properties",
            "required",
            "items",
            "oneOf",
            "anyOf",
            "allOf",
            "$schema",
            "$defs",
            "definitions",
            "enum",
        }
        return any(key in parsed for key in schema_markers)

    @staticmethod
    def _schema_format(schema_hint: str) -> dict[str, Any] | str:
        try:
            parsed = json.loads(str(schema_hint or "").strip())
        except Exception:
            return "json"
        if isinstance(parsed, dict) and OllamaExtractorBackend._looks_like_json_schema(parsed):
            return parsed
        return "json"

    async def _request(self, payload: dict[str, Any], *, retries: int = 3) -> dict[str, Any]:
        if self._session is None or self._session.closed:
            await self.start()
        assert self._session is not None

        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                async with self._session.post(self._endpoint(), json=payload) as response:
                    text = await response.text()
                    if response.status == 200:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict):
                            return parsed
                        raise RuntimeError("Ollama returned non-object JSON response")
                    retriable = response.status in {408, 409, 429, 500, 502, 503, 504}
                    if not retriable:
                        raise RuntimeError(f"Ollama error {response.status}: {text}")
                    # Some prompt schema hints are "example JSON objects", not strict JSON Schema.
                    # If Ollama rejects `format`, transparently retry with plain JSON mode.
                    if (
                        response.status == 500
                        and "invalid json schema in format" in text.lower()
                        and payload.get("format") != "json"
                    ):
                        payload = dict(payload)
                        payload["format"] = "json"
                        last_error = RuntimeError(
                            "Ollama rejected structured `format` schema; retrying with format='json'"
                        )
                        continue
                    last_error = RuntimeError(f"Ollama retriable error {response.status}: {text}")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
            if attempt < retries:
                await asyncio.sleep(min(4.0, 0.35 * attempt + random.random() * 0.25))

        if last_error is not None:
            raise RuntimeError(f"Ollama extractor request failed after retries: {last_error}")
        raise RuntimeError("Ollama extractor request failed without explicit error")

    @staticmethod
    def _extract_message_text(data: dict[str, Any]) -> str:
        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
        response_text = data.get("response")
        if isinstance(response_text, str) and response_text.strip():
            return response_text
        raise RuntimeError("Ollama extractor returned empty message content")

    async def json_chat(
        self,
        messages: list[dict[str, str]],
        schema_hint: str,
        temperature: float = 0.1,
        max_output_tokens: int = 900,
    ) -> dict[str, Any] | None:
        mapped_messages: list[dict[str, str]] = []
        for msg in messages:
            role = str(msg.get("role", "")).strip().lower() or "user"
            if role not in {"system", "user", "assistant"}:
                role = "user"
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            mapped_messages.append({"role": role, "content": content})

        if not mapped_messages:
            return None

        options: dict[str, Any] = {
            "temperature": float(temperature),
        }
        if int(max_output_tokens) > 0:
            options["num_predict"] = int(max_output_tokens)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": mapped_messages,
            "stream": False,
            "think": False,
            "format": self._schema_format(schema_hint),
            "options": options,
        }
        data = await self._request(payload)
        raw = self._extract_message_text(data)
        cleaned = self._strip_json_fences(raw)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed
