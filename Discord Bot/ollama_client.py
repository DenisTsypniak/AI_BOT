from __future__ import annotations

from typing import Dict, List

import aiohttp


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int,
        temperature: float,
        num_ctx: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.temperature = temperature
        self.num_ctx = num_ctx
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def chat(self, messages: List[Dict[str, str]]) -> str:
        if self._session is None or self._session.closed:
            await self.start()
        assert self._session is not None

        payload = {
            "model": self.model,
            "stream": False,
            "messages": messages,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }

        async with self._session.post(f"{self.base_url}/api/chat", json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Ollama error {response.status}: {error_text}")
            data = await response.json()

        message = data.get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            raise RuntimeError("Ollama returned empty content.")
        return content
