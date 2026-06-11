"""Local speech-to-text via Whisper-MLX.

Feature 3: replace Google cloud STT with fully-offline, Apple-Silicon-accelerated
transcription. Model is loaded once and cached for the session lifetime.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

# Whisper expects float32 at 16 kHz
_WHISPER_SAMPLE_RATE = 16_000


@dataclass
class TranscriptResult:
    text: str
    no_speech_prob: float   # 0–1; high = likely silence/noise
    avg_logprob: float      # quality signal for reference-segment ranking (Feature 6)
    latency_s: float        # inference time in seconds


class Transcriber:
    """Wraps mlx-whisper; model is loaded lazily on first use and then cached."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self._model_id = model
        self._model = None   # loaded on first transcribe() call

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import mlx_whisper
        # mlx_whisper.load_models returns a dict of model components
        self._mlx_whisper = mlx_whisper
        # Run a silent warmup decode to JIT-compile the MLX graph
        dummy = np.zeros(_WHISPER_SAMPLE_RATE, dtype=np.float32)  # 1 s silence
        self._mlx_whisper.transcribe(
            dummy,
            path_or_hf_repo=self._model_id,
            language="en",
        )
        self._model = True  # sentinel: model is warm
        print(f"[STT] Whisper model warmed: {self._model_id}")

    def transcribe(self, wav: np.ndarray, language: Optional[str] = "en") -> TranscriptResult:
        """Transcribe a float32 waveform sampled at 16 kHz.

        Args:
            wav: float32 numpy array at 16 kHz.  May be empty (returns empty text).
            language: BCP-47 language code, or None to auto-detect.

        Returns:
            TranscriptResult with text, no_speech_prob, avg_logprob, latency_s.
        """
        if wav is None or len(wav) == 0:
            return TranscriptResult(text="", no_speech_prob=1.0, avg_logprob=-10.0, latency_s=0.0)

        self._ensure_loaded()

        wav = wav.astype(np.float32)
        t0 = time.perf_counter()

        kwargs = dict(path_or_hf_repo=self._model_id, verbose=False)
        if language:
            kwargs["language"] = language

        result = self._mlx_whisper.transcribe(wav, **kwargs)

        latency_s = time.perf_counter() - t0

        # Extract quality signals from the first segment if available
        segments = result.get("segments", [])
        if segments:
            no_speech_prob = float(segments[0].get("no_speech_prob", 0.0))
            avg_logprob = float(segments[0].get("avg_logprob", 0.0))
        else:
            no_speech_prob = 1.0
            avg_logprob = -10.0

        text = result.get("text", "").strip()
        return TranscriptResult(
            text=text,
            no_speech_prob=no_speech_prob,
            avg_logprob=avg_logprob,
            latency_s=latency_s,
        )

    def is_speech(self, result: TranscriptResult, threshold: float = 0.6) -> bool:
        """Return True if the result contains actual speech (not silence/noise)."""
        return result.no_speech_prob < threshold and len(result.text) > 0
