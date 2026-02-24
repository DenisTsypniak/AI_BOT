from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Any, Dict, List

import aiohttp


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: int,
        temperature: float,
        max_output_tokens: int,
        base_url: str = "https://generativelanguage.googleapis.com",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.temperature = temperature
        self.max_output_tokens: int | None = int(max_output_tokens) if int(max_output_tokens) > 0 else None
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    def _endpoint(self) -> str:
        return f"{self.base_url}/v1beta/models/{self.model}:generateContent?key={self.api_key}"

    @staticmethod
    def _map_messages(messages: List[Dict[str, str]]) -> Dict[str, Any]:
        system_lines: List[str] = []
        contents: List[Dict[str, Any]] = []

        for message in messages:
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if role == "system":
                system_lines.append(content)
                continue
            mapped_role = "model" if role == "assistant" else "user"
            contents.append({"role": mapped_role, "parts": [{"text": content}]})

        payload: Dict[str, Any] = {"contents": contents}
        if system_lines:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_lines)}],
            }
        return payload

    async def _request(self, payload: Dict[str, Any], retries: int = 3) -> Dict[str, Any]:
        if self._session is None or self._session.closed:
            await self.start()
        assert self._session is not None

        url = self._endpoint()
        last_error: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                async with self._session.post(url, json=payload) as response:
                    text = await response.text()
                    if response.status == 200:
                        return json.loads(text)

                    retriable = response.status in {408, 409, 429, 500, 502, 503, 504}
                    if not retriable:
                        raise RuntimeError(f"Gemini error {response.status}: {text}")
                    last_error = RuntimeError(f"Gemini retriable error {response.status}: {text}")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc

            if attempt < retries:
                await asyncio.sleep(min(4.0, 0.35 * attempt + random.random() * 0.2))

        if last_error is not None:
            raise RuntimeError(f"Gemini request failed after retries: {last_error}")
        raise RuntimeError("Gemini request failed without explicit error")

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            prompt_feedback = data.get("promptFeedback") or {}
            block_reason = prompt_feedback.get("blockReason")
            if block_reason:
                raise RuntimeError(f"Gemini blocked response: {block_reason}")
            raise RuntimeError("Gemini returned no candidates")

        first = candidates[0]
        content = first.get("content") or {}
        parts = content.get("parts") or []
        chunks: List[str] = []

        for part in parts:
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())

        joined = "\n".join(chunks).strip()
        if joined:
            return joined

        finish_reason = first.get("finishReason")
        if finish_reason:
            raise RuntimeError(f"Gemini empty response (finishReason={finish_reason})")
        raise RuntimeError("Gemini empty response")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        payload = self._map_messages(messages)
        generation_config: Dict[str, Any] = {
            "temperature": self.temperature if temperature is None else temperature,
        }
        selected_tokens = self.max_output_tokens if max_output_tokens is None else max_output_tokens
        if selected_tokens is not None and int(selected_tokens) > 0:
            generation_config["maxOutputTokens"] = int(selected_tokens)
        payload["generationConfig"] = generation_config
        data = await self._request(payload)
        return self._extract_text(data)

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        return cleaned

    async def json_chat(
        self,
        messages: List[Dict[str, str]],
        schema_hint: str,
        temperature: float = 0.1,
        max_output_tokens: int = 900,
    ) -> Dict[str, Any] | None:
        strict_messages = list(messages)
        strict_messages.append(
            {
                "role": "system",
                "content": (
                    "Return only valid JSON object with no markdown and no additional commentary. "
                    f"Schema hint: {schema_hint}"
                ),
            }
        )
        raw = await self.chat(
            strict_messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        cleaned = self._strip_json_fences(raw)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed
