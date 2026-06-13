"""Rolling reference audio accumulator for progressive voice cloning.

Feature 6: maintain a quality-ranked buffer of participant utterances.  As
audio accumulates across tiers (3 s → 9 s → 20 s+), the reference passed to
F5-TTS sharpens from a rough approximation into an uncannily exact clone.

Feature 7: migration_alpha(tier) maps each sharpening tier to a blend factor
for VoiceCloner.synthesize(), ramping from neutral→self as reference accumulates.

Public API:
    ReferenceBuffer.push(wav, transcript_result) -> None
    ReferenceBuffer.best_reference() -> tuple[np.ndarray, str] | tuple[None, None]
    ReferenceBuffer.voiced_seconds -> float
    ReferenceBuffer.tier -> int  (0 = no ref, 1-3 = progressive tiers)
    ReferenceBuffer.migration_alpha(tier) -> float
    ReferenceBuffer.reset() -> None
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from talk2me.stt.whisper import TranscriptResult

# F5-TTS native sample rate that reference audio is ultimately used at
_F5_SAMPLE_RATE = 24_000

# Quality thresholds
_MIN_RMS = 0.005          # discard near-silent segments
_MAX_PEAK = 0.95          # discard clipped segments
_MIN_AVG_LOGPROB = -1.5   # discard poorly-transcribed segments

# Tier voiced-duration thresholds (seconds of good audio)
_TIER_THRESHOLDS = (3.0, 9.0, 20.0)

# Cap on total reference audio fed to F5-TTS (longer hurts quality)
_MAX_REFERENCE_SECONDS = 30.0

# Feature 7: default migration alpha per tier (0=neutral→self=1.0)
_DEFAULT_MIGRATION_ALPHAS = (0.2, 0.5, 0.8, 1.0)  # tier 0, 1, 2, 3


@dataclass
class _Segment:
    wav: np.ndarray          # float32, at capture sample rate
    text: str
    avg_logprob: float
    sample_rate: int
    duration_s: float


class ReferenceBuffer:
    """Accumulates participant utterances and surfaces the best reference clip.

    Segments are scored by avg_logprob (Whisper confidence) and RMS, then the
    best-scoring ones up to _MAX_REFERENCE_SECONDS are concatenated for F5-TTS.

    The tier property signals how much audio has accumulated:
        0 — no usable reference yet
        1 — ≥ 3 s voiced  (rough approximation)
        2 — ≥ 9 s voiced  (recognisable)
        3 — ≥ 20 s voiced (uncannily exact)
    """

    def __init__(
        self,
        tier_thresholds: Tuple[float, float, float] = _TIER_THRESHOLDS,
        max_reference_seconds: float = _MAX_REFERENCE_SECONDS,
        min_avg_logprob: float = _MIN_AVG_LOGPROB,
        min_rms: float = _MIN_RMS,
        max_peak: float = _MAX_PEAK,
        migration_alphas: Tuple[float, ...] = _DEFAULT_MIGRATION_ALPHAS,
    ):
        self._tier_thresholds = tier_thresholds
        self._max_ref_s = max_reference_seconds
        self._min_logprob = min_avg_logprob
        self._min_rms = min_rms
        self._max_peak = max_peak
        self._migration_alphas = migration_alphas

        self._segments: list[_Segment] = []
        self._voiced_seconds: float = 0.0
        self._cached_ref: Optional[Tuple[np.ndarray, str]] = None  # invalidated on push

    # ── public interface ──────────────────────────────────────────────────────

    def push(
        self,
        wav: np.ndarray,
        transcript_result: TranscriptResult,
        sample_rate: int = 16_000,
    ) -> bool:
        """Add a new participant utterance.

        Returns True if the segment passed quality checks and was accepted.
        """
        if wav is None or len(wav) == 0:
            return False

        wav = wav.astype(np.float32)
        rms = float(np.sqrt(np.mean(wav ** 2)))
        peak = float(np.max(np.abs(wav)))
        duration_s = len(wav) / sample_rate

        if rms < self._min_rms:
            return False
        if peak > self._max_peak:
            return False
        if transcript_result.avg_logprob < self._min_logprob:
            return False
        if not transcript_result.text.strip():
            return False

        seg = _Segment(
            wav=wav,
            text=transcript_result.text.strip(),
            avg_logprob=transcript_result.avg_logprob,
            sample_rate=sample_rate,
            duration_s=duration_s,
        )
        self._segments.append(seg)
        self._voiced_seconds += duration_s
        self._cached_ref = None  # invalidate cache
        return True

    def best_reference(self) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Return (wav_at_16kHz, text) of the best accumulated reference.

        Segments are ranked by avg_logprob descending, then concatenated up to
        _MAX_REFERENCE_SECONDS.  Returns (None, None) if no usable audio yet.

        Note: returned wav is at the original capture sample rate (16 kHz).
        VoiceCloner.synthesize() accepts a reference_sample_rate parameter and
        resamples internally.
        """
        if not self._segments:
            return None, None

        if self._cached_ref is not None:
            return self._cached_ref

        ranked = sorted(self._segments, key=lambda s: s.avg_logprob, reverse=True)

        selected: list[_Segment] = []
        total_s = 0.0
        for seg in ranked:
            if total_s >= self._max_ref_s:
                break
            selected.append(seg)
            total_s += seg.duration_s

        # Sort selected segments back into capture order for a natural audio clip
        segment_ids = {id(s): i for i, s in enumerate(self._segments)}
        selected_sorted = sorted(selected, key=lambda s: segment_ids[id(s)])

        wavs = [s.wav for s in selected_sorted]
        texts = [s.text for s in selected_sorted]
        sr = selected_sorted[0].sample_rate  # all segments share the mic sample rate

        combined_wav = np.concatenate(wavs).astype(np.float32)
        combined_text = " ".join(texts)

        self._cached_ref = (combined_wav, combined_text)
        return self._cached_ref

    @property
    def voiced_seconds(self) -> float:
        """Total seconds of accepted voiced audio accumulated so far."""
        return self._voiced_seconds

    @property
    def tier(self) -> int:
        """Progressive sharpening tier (0–3)."""
        for i, threshold in enumerate(reversed(self._tier_thresholds), start=1):
            if self._voiced_seconds >= threshold:
                return len(self._tier_thresholds) - i + 1
        return 0

    def migration_alpha(self, tier: Optional[int] = None) -> float:
        """Return the blend factor for the given tier (Feature 7).

        0.0 = full neutral seed, 1.0 = full participant voice.
        If tier is None, uses the current buffer tier.
        """
        t = self.tier if tier is None else tier
        t = max(0, min(t, len(self._migration_alphas) - 1))
        return self._migration_alphas[t]

    def reset(self) -> None:
        """Clear all accumulated audio — call at session end."""
        self._segments.clear()
        self._voiced_seconds = 0.0
        self._cached_ref = None
