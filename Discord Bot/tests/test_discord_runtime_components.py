from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("discord")

import live_role_bot.discord.client as client_mod  # noqa: E402


class _FakeVoiceClient:
    def __init__(self, *, raises: bool = False) -> None:
        self.disconnect_calls = 0
        self._raises = raises

    async def disconnect(self, *, force: bool = False) -> None:
        self.disconnect_calls += 1
        if self._raises:
            raise RuntimeError("disconnect failed")


def test_disconnect_voice_clients_gracefully_handles_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_mod, "VOICE_RECV_AVAILABLE", False)
    monkeypatch.setattr(client_mod, "voice_recv", None)

    vc_ok = _FakeVoiceClient()
    vc_fail = _FakeVoiceClient(raises=True)
    fake_bot = SimpleNamespace(
        guilds=[
            SimpleNamespace(voice_client=vc_ok),
            SimpleNamespace(voice_client=vc_fail),
            SimpleNamespace(voice_client=None),
        ]
    )

    asyncio.run(client_mod.LiveRoleDiscordBot._disconnect_voice_clients(fake_bot))

    assert vc_ok.disconnect_calls == 1
    assert vc_fail.disconnect_calls == 1
