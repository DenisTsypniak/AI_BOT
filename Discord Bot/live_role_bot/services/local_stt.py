from __future__ import annotations

import asyncio
import contextlib
import logging
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import audioop
except ModuleNotFoundError:
    import audioop_lts as audioop  # type: ignore[import-not-found]


logger = logging.getLogger("live_role_bot")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class STTResult:
    text: str
    confidence: float
    duration_ms: int
    rms: int
    model_name: str
    status: str


class LocalSTT:
    def __init__(
        self,
        enabled: bool,
        model: str,
        fallback_model: str,
        device: str,
        compute_type: str,
        language: str,
        max_audio_seconds: int,
    ) -> None:
        self.enabled = enabled
        self.model_name = model.strip() or "medium"
        self.fallback_model_name = fallback_model.strip()
        self.device = device.strip() or "auto"
        self.compute_type = compute_type.strip() or "int8"
        self.language = language.strip() or None
        self.max_audio_seconds = max(4, int(max_audio_seconds))

        self._model: Any | None = None
        self._fallback_model: Any | None = None
        self._load_failed = False
        self._fallback_load_failed = False

    def _load_model_sync(self) -> Any | None:
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None

        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception:
            self._load_failed = True
            logger.exception("Local STT failed to import faster_whisper")
            return None

        try:
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception:
            self._load_failed = True
            logger.exception(
                "Local STT failed to load primary model model=%s device=%s compute=%s",
                self.model_name,
                self.device,
                self.compute_type,
            )
            return None

        return self._model

    def _load_fallback_model_sync(self) -> Any | None:
        if not self.fallback_model_name:
            return None
        if self._fallback_model is not None:
            return self._fallback_model
        if self._fallback_load_failed:
            return None

        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception:
            self._fallback_load_failed = True
            logger.exception("Local STT failed to import faster_whisper for fallback model")
            return None

        try:
            self._fallback_model = WhisperModel(
                self.fallback_model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception:
            self._fallback_load_failed = True
            logger.exception(
                "Local STT failed to load fallback model model=%s device=%s compute=%s",
                self.fallback_model_name,
                self.device,
                self.compute_type,
            )
            return None

        return self._fallback_model

    @staticmethod
    def _decode_segments(segments: Any) -> tuple[str, float]:
        chunks: list[str] = []
        logprobs: list[float] = []
        no_speech: list[float] = []

        for segment in segments:
            text = str(getattr(segment, "text", "")).strip()
            if text:
                chunks.append(" ".join(text.split()))

            avg_logprob = getattr(segment, "avg_logprob", None)
            if isinstance(avg_logprob, (float, int)):
                logprobs.append(float(avg_logprob))

            no_speech_prob = getattr(segment, "no_speech_prob", None)
            if isinstance(no_speech_prob, (float, int)):
                no_speech.append(float(no_speech_prob))

        transcript = " ".join(chunk for chunk in chunks if chunk).strip()
        if not transcript:
            return "", 0.0

        if logprobs:
            avg_logprob = sum(logprobs) / len(logprobs)
            confidence = _clamp((avg_logprob + 1.4) / 1.4, 0.0, 1.0)
        else:
            confidence = 0.45

        if no_speech:
            avg_no_speech = _clamp(sum(no_speech) / len(no_speech), 0.0, 1.0)
            confidence *= _clamp(1.0 - avg_no_speech * 0.7, 0.25, 1.0)

        if len(transcript) < 6:
            confidence *= 0.82

        return transcript, _clamp(confidence, 0.0, 1.0)

    @staticmethod
    def _pcm_duration_ms(pcm_48k_stereo: bytes) -> int:
        return int((len(pcm_48k_stereo) / (48000 * 4)) * 1000)

    @staticmethod
    def _pcm_rms(pcm_48k_stereo: bytes) -> int:
        try:
            return int(audioop.rms(pcm_48k_stereo, 2))
        except Exception:
            return 0

    def _prepare_wav(self, pcm_48k_stereo: bytes) -> Path | None:
        if not pcm_48k_stereo:
            return None

        max_bytes = 48000 * 4 * self.max_audio_seconds
        if len(pcm_48k_stereo) > max_bytes:
            pcm_48k_stereo = pcm_48k_stereo[-max_bytes:]

        mono = audioop.tomono(pcm_48k_stereo, 2, 0.5, 0.5)
        mono_16k, _ = audioop.ratecv(mono, 2, 1, 48000, 16000, None)

        with tempfile.NamedTemporaryFile(prefix="discord_stt_", suffix=".wav", delete=False) as tmp:
            raw_path = tmp.name
        path = Path(raw_path)

        with wave.open(raw_path, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(mono_16k)

        return path

    def _run_transcribe_pass(self, model_obj: Any, wav_path: Path) -> tuple[str, float]:
        segments, _ = model_obj.transcribe(
            str(wav_path),
            language=self.language,
            beam_size=4,
            best_of=4,
            temperature=0.0,
            vad_filter=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.7,
            compression_ratio_threshold=2.6,
            log_prob_threshold=-1.4,
        )
        return self._decode_segments(segments)

    def _transcribe_sync(self, pcm_48k_stereo: bytes) -> STTResult:
        duration_ms = self._pcm_duration_ms(pcm_48k_stereo)
        rms = self._pcm_rms(pcm_48k_stereo)

        if not self.enabled:
            return STTResult(
                text="",
                confidence=0.0,
                duration_ms=duration_ms,
                rms=rms,
                model_name=self.model_name,
                status="disabled",
            )

        if duration_ms < 180:
            return STTResult(
                text="",
                confidence=0.0,
                duration_ms=duration_ms,
                rms=rms,
                model_name=self.model_name,
                status="too_short",
            )

        primary = self._load_model_sync()
        if primary is None:
            return STTResult(
                text="",
                confidence=0.0,
                duration_ms=duration_ms,
                rms=rms,
                model_name=self.model_name,
                status="model_unavailable",
            )

        wav_path = self._prepare_wav(pcm_48k_stereo)
        if wav_path is None:
            return STTResult(
                text="",
                confidence=0.0,
                duration_ms=duration_ms,
                rms=rms,
                model_name=self.model_name,
                status="empty_audio",
            )

        try:
            text, confidence = self._run_transcribe_pass(primary, wav_path)
            model_name_used = self.model_name

            needs_fallback = (not text) or confidence < 0.34
            if needs_fallback:
                fallback = self._load_fallback_model_sync()
                if fallback is not None:
                    fallback_text, fallback_conf = self._run_transcribe_pass(fallback, wav_path)
                    if fallback_text and (fallback_conf >= confidence or not text):
                        text = fallback_text
                        confidence = fallback_conf
                        model_name_used = self.fallback_model_name

            if not text:
                return STTResult(
                    text="",
                    confidence=0.0,
                    duration_ms=duration_ms,
                    rms=rms,
                    model_name=model_name_used,
                    status="empty",
                )

            status = "ok" if confidence >= 0.34 else "low_confidence"
            return STTResult(
                text=text,
                confidence=_clamp(confidence, 0.0, 1.0),
                duration_ms=duration_ms,
                rms=rms,
                model_name=model_name_used,
                status=status,
            )
        except Exception:
            logger.exception(
                "Local STT transcribe failed status=error model=%s duration_ms=%s rms=%s",
                self.model_name,
                duration_ms,
                rms,
            )
            return STTResult(
                text="",
                confidence=0.0,
                duration_ms=duration_ms,
                rms=rms,
                model_name=self.model_name,
                status="error",
            )
        finally:
            with contextlib.suppress(OSError):
                wav_path.unlink()

    async def transcribe(self, pcm_48k_stereo: bytes) -> STTResult:
        return await asyncio.to_thread(self._transcribe_sync, pcm_48k_stereo)
