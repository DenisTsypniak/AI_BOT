from __future__ import annotations

from typing import Any

from google.genai import types


def _build_system_instruction(prompt: str, preferred_language: str) -> str:
    language_line = f"Always speak in {preferred_language}."
    pacing_line = (
        "Reply immediately when user finishes speaking. "
        "Keep each voice reply very short: 1 sentence, max 2 sentences only if necessary."
    )
    brevity_line = (
        "No long explanations unless user explicitly asks for details. "
        "No lists, no long intros, no recap."
    )
    completion_line = "Always finish your final sentence naturally. Never end a reply mid-sentence."
    live_line = (
        "Stay fully live: keep listening while speaking. "
        "If user continues speaking, immediately adapt and rebuild answer from updated input."
    )
    silence_line = (
        "If there is silence, keyboard clicks, breathing, random background noise, coughs, or unclear fragments, "
        "stay silent and keep listening."
    )
    echo_line = "Never answer your own echoed voice from speakers or microphone bleed."
    base = prompt.strip()
    if not base:
        return (
            f"{language_line}\n{pacing_line}\n{brevity_line}\n"
            f"{completion_line}\n{live_line}\n{silence_line}\n{echo_line}"
        ).strip()
    return (
        f"{base}\n\n{language_line}\n{pacing_line}\n{brevity_line}\n"
        f"{completion_line}\n{live_line}\n{silence_line}\n{echo_line}"
    ).strip()


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
