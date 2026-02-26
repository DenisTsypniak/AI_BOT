from __future__ import annotations

import re
from typing import Any

from .ollama_extractor_backend import OllamaExtractorBackend


class OllamaChatClient(OllamaExtractorBackend):
    """Ollama chat client compatible with GeminiClient's `chat/json_chat` interface."""

    backend_name = "ollama"

    @staticmethod
    def _sanitize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        mapped_messages: list[dict[str, str]] = []
        for msg in messages:
            role = str(msg.get("role", "")).strip().lower() or "user"
            if role not in {"system", "user", "assistant"}:
                role = "user"
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            mapped_messages.append({"role": role, "content": content})
        return mapped_messages

    @staticmethod
    def _strip_reasoning_blocks(text: str) -> str:
        cleaned = str(text or "").strip()
        # Some reasoning-capable models may emit hidden-thought tags.
        cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        return cleaned

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: int = 45,
        temperature: float = 0.1,
        max_output_tokens: int = 0,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )
        self.max_output_tokens = max(0, int(max_output_tokens or 0))

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        mapped_messages = self._sanitize_messages(messages)
        if not mapped_messages:
            return ""

        options: dict[str, Any] = {
            "temperature": float(self.temperature if temperature is None else temperature),
        }
        selected_tokens = self.max_output_tokens if max_output_tokens is None else max_output_tokens
        if selected_tokens is not None:
            try:
                selected_tokens = int(selected_tokens)
            except (TypeError, ValueError):
                selected_tokens = None
        if isinstance(selected_tokens, int) and selected_tokens > 0:
            options["num_predict"] = selected_tokens

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": mapped_messages,
            "stream": False,
            "think": False,
            "options": options,
        }
        data = await self._request(payload)
        raw = self._extract_message_text(data)
        return self._strip_reasoning_blocks(raw)
