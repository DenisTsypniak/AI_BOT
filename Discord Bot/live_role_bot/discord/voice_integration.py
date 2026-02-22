from __future__ import annotations

import contextlib
import logging
from typing import Any

import discord

logger = logging.getLogger("live_role_bot")

try:
    from discord.ext import voice_recv

    VOICE_RECV_AVAILABLE = True
except Exception:
    voice_recv = None
    VOICE_RECV_AVAILABLE = False


_VOICE_RECV_DECODE_GUARD_INSTALLED = False


def install_voice_recv_decode_guard() -> None:
    global _VOICE_RECV_DECODE_GUARD_INSTALLED
    if _VOICE_RECV_DECODE_GUARD_INSTALLED:
        return
    if not VOICE_RECV_AVAILABLE:
        return

    try:
        from discord.opus import Decoder, OpusError
        from discord.ext.voice_recv import opus as voice_recv_opus
    except Exception:
        return

    original_decode_packet = getattr(voice_recv_opus.PacketDecoder, "_decode_packet", None)
    if original_decode_packet is None:
        return

    def _guarded_decode_packet(self: Any, packet: Any) -> tuple[Any, bytes]:
        try:
            return original_decode_packet(self, packet)
        except OpusError as exc:
            if "corrupted stream" not in str(exc).lower():
                raise
            with contextlib.suppress(Exception):
                setattr(self, "_decoder", None if self.sink.wants_opus() else Decoder())
            return packet, b""
        except Exception:
            with contextlib.suppress(Exception):
                setattr(self, "_decoder", None if self.sink.wants_opus() else Decoder())
            return packet, b""

    setattr(voice_recv_opus.PacketDecoder, "_decode_packet", _guarded_decode_packet)
    _VOICE_RECV_DECODE_GUARD_INSTALLED = True
    logger.info("voice_recv decode guard enabled (corrupted opus packets will be ignored)")


class _VoiceInputSinkFallback:
    def __init__(self, bot: Any, guild_id: int, bot_user_id: int) -> None:
        self.bot = bot
        self.guild_id = guild_id
        self.bot_user_id = bot_user_id

    def wants_opus(self) -> bool:
        return False

    def write(self, user: discord.abc.User | None, data: Any) -> None:
        if user is None:
            return
        if getattr(user, "bot", False):
            return
        if user.id == self.bot_user_id:
            return
        pcm = getattr(data, "pcm", None)
        if not isinstance(pcm, (bytes, bytearray)) or not pcm:
            return
        label = getattr(user, "display_name", user.name)
        self.bot.push_voice_pcm(self.guild_id, user.id, label, bytes(pcm))

    def cleanup(self) -> None:
        return


if VOICE_RECV_AVAILABLE:
    from discord.ext.voice_recv.sinks import AudioSink as _RuntimeAudioSink

    class _VoiceInputSinkRuntime(_RuntimeAudioSink):
        def __init__(self, bot: Any, guild_id: int, bot_user_id: int) -> None:
            super().__init__(None)
            self.bot = bot
            self.guild_id = guild_id
            self.bot_user_id = bot_user_id

        def wants_opus(self) -> bool:
            return False

        def write(self, user: discord.abc.User | None, data: Any) -> None:
            if user is None:
                return
            if getattr(user, "bot", False):
                return
            if user.id == self.bot_user_id:
                return
            pcm = getattr(data, "pcm", None)
            if not isinstance(pcm, (bytes, bytearray)) or not pcm:
                return
            label = getattr(user, "display_name", user.name)
            self.bot.push_voice_pcm(self.guild_id, user.id, label, bytes(pcm))

        def cleanup(self) -> None:
            return

    VoiceInputSink = _VoiceInputSinkRuntime
else:
    VoiceInputSink = _VoiceInputSinkFallback

