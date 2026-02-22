from __future__ import annotations

from .base import _NativeAudioBase
from .events import _NativeAudioEventsMixin
from .io import _NativeAudioIOMixin
from .playback import _NativeAudioPlaybackMixin
from .session import _NativeAudioSessionMixin


class GeminiNativeAudioManager(
    _NativeAudioBase,
    _NativeAudioSessionMixin,
    _NativeAudioIOMixin,
    _NativeAudioPlaybackMixin,
    _NativeAudioEventsMixin,
):
    """Orchestrates Gemini Live Native Audio sessions for Discord voice channels."""

