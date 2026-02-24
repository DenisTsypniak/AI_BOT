from __future__ import annotations

from typing import Any

from google.genai import types
from ...prompts.voice import build_native_audio_system_instruction


def _build_system_instruction(prompt: str, preferred_language: str) -> str:
    return build_native_audio_system_instruction(prompt, preferred_language)


def _build_live_connect_config(
    prompt: str,
    preferred_language: str,
    voice_name: str,
    temperature: float,
    max_output_tokens: int | None,
) -> types.LiveConnectConfig:
    kwargs: dict[str, Any] = {
        "response_modalities": ["AUDIO"],
        "temperature": temperature,
        "speech_config": types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name,
                )
            )
        ),
        "system_instruction": types.Content(
            role="user",
            parts=[types.Part(text=_build_system_instruction(prompt, preferred_language))],
        ),
        "input_audio_transcription": types.AudioTranscriptionConfig(),
        "output_audio_transcription": types.AudioTranscriptionConfig(),
        "realtime_input_config": types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=True,
            ),
            activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
        ),
    }
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    return types.LiveConnectConfig(**kwargs)
