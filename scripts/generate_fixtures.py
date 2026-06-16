"""Generate voice fixture clips for offline simulation and regression testing.

Creates tests/fixtures/voices/ with WAV clips for three simulated speakers:
  speaker_f1 — extracted from the F5-TTS bundled test clip (real human voice)
  speaker_m1 — procedural speech-like audio (lower fundamental, different rhythm)
  speaker_nn1 — procedural with slower tempo (simulates careful non-native speech)

For real voice diversity, replace these files with LibriSpeech test-clean clips:
  https://www.openslr.org/12/  (test-clean is ~346 MB)
  Choose one female, one male, one non-native speaker; trim to ≤30 s.

Usage:
    uv run python scripts/generate_fixtures.py
"""
from __future__ import annotations

import pkgutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "voices"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16_000


# ── helpers ───────────────────────────────────────────────────────────────────

def _sine_burst(freq_hz: float, duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Single voiced phoneme approximation: windowed sine burst."""
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, endpoint=False)
    wave = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    # Hann window for smooth onset/offset
    window = np.hanning(n).astype(np.float32)
    return wave * window * 0.25


def _silence(duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def _make_speech_like(
    f0: float,           # fundamental (voice pitch)
    syllables: list[tuple[float, float]],  # [(voiced_s, gap_s), ...]
    sr: int = SAMPLE_RATE,
    noise_level: float = 0.02,
) -> np.ndarray:
    """Build a rough speech-like waveform: alternating voiced bursts and gaps."""
    parts: list[np.ndarray] = []
    rng = np.random.default_rng(42)
    for voiced_s, gap_s in syllables:
        # Voiced: fundamental + harmonics at 1:1/2:1/3 amplitude ratio
        burst = (
            _sine_burst(f0, voiced_s, sr)
            + _sine_burst(f0 * 2, voiced_s, sr) * 0.5
            + _sine_burst(f0 * 3, voiced_s, sr) * 0.33
        ) / 1.83  # normalise
        # Add gentle broadband noise (fricative simulation)
        noise = rng.standard_normal(len(burst)).astype(np.float32) * noise_level
        parts.append(burst + noise)
        if gap_s > 0:
            parts.append(_silence(gap_s, sr))
    return np.concatenate(parts)


def _write_fixture(name: str, wav: np.ndarray, transcript: str) -> None:
    wav_path = FIXTURES_DIR / f"{name}.wav"
    txt_path = FIXTURES_DIR / f"{name}.txt"
    sf.write(str(wav_path), wav, SAMPLE_RATE)
    txt_path.write_text(transcript, encoding="utf-8")
    duration_s = len(wav) / SAMPLE_RATE
    print(f'  {wav_path.name}  ({duration_s:.1f}s)  "{transcript[:50]}"')


# ── speaker_f1: extracted from F5-TTS bundled test clip ──────────────────────

def _make_speaker_f1() -> None:
    print("speaker_f1 — extracting F5-TTS bundled test clip …")
    wav_path = FIXTURES_DIR / "speaker_f1.wav"
    txt_path = FIXTURES_DIR / "speaker_f1.txt"

    try:
        import tempfile
        wav_bytes = pkgutil.get_data("f5_tts_mlx", "tests/test_en_1_ref_short.wav")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        src_wav, src_sr = sf.read(tmp_path, dtype="float32")
        if src_wav.ndim > 1:
            src_wav = src_wav[:, 0]
        # Resample to 16 kHz if needed
        if src_sr != SAMPLE_RATE:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(src_sr), SAMPLE_RATE)
            src_wav = resample_poly(
                src_wav.astype(np.float64), SAMPLE_RATE // g, int(src_sr) // g
            ).astype(np.float32)
        sf.write(str(wav_path), src_wav, SAMPLE_RATE)
        txt_path.write_text(
            "Some call me nature, others call me mother nature.",
            encoding="utf-8",
        )
        print(f"  {wav_path.name}  ({len(src_wav)/SAMPLE_RATE:.1f}s)  (real voice — F5-TTS test clip)")
    except Exception as exc:
        print(f"  Warning: could not extract F5-TTS clip ({exc}); generating synthetic fallback.")
        _make_speaker_f1_synthetic()


def _make_speaker_f1_synthetic() -> None:
    """Fallback if F5-TTS clip is unavailable."""
    syllables = [
        (0.12, 0.04), (0.10, 0.03), (0.11, 0.05), (0.09, 0.04),
        (0.12, 0.03), (0.10, 0.04), (0.11, 0.05), (0.09, 0.03),
        (0.12, 0.04), (0.10, 0.03),
    ]
    wav = _make_speech_like(f0=220.0, syllables=syllables)
    _write_fixture(
        "speaker_f1",
        wav,
        "Some call me nature, others call me mother nature.",
    )


# ── speaker_m1: procedural male-range voice ───────────────────────────────────

def _make_speaker_m1() -> None:
    print("speaker_m1 — generating procedural (male-range pitch ~120 Hz) …")
    # ~10 s of natural-paced speech rhythm
    syllables = [
        (0.15, 0.05), (0.13, 0.04), (0.14, 0.06), (0.12, 0.04),
        (0.15, 0.05), (0.14, 0.04), (0.13, 0.05), (0.12, 0.06),
        (0.15, 0.04), (0.14, 0.05), (0.13, 0.04), (0.12, 0.05),
        (0.14, 0.06), (0.15, 0.04), (0.13, 0.05), (0.12, 0.04),
    ]
    wav = _make_speech_like(f0=120.0, syllables=syllables, noise_level=0.025)
    _write_fixture(
        "speaker_m1",
        wav,
        "The weather today is mild and clear with a gentle breeze.",
    )


# ── speaker_nn1: slower tempo, higher pitch variation ─────────────────────────

def _make_speaker_nn1() -> None:
    print("speaker_nn1 — generating procedural (non-native: slower, ~175 Hz) …")
    # Slower rhythm: longer gaps, more careful pacing
    syllables = [
        (0.18, 0.09), (0.16, 0.08), (0.17, 0.09), (0.15, 0.08),
        (0.18, 0.10), (0.16, 0.09), (0.17, 0.08), (0.15, 0.09),
        (0.18, 0.09), (0.16, 0.08), (0.17, 0.09), (0.15, 0.10),
    ]
    wav = _make_speech_like(f0=175.0, syllables=syllables, noise_level=0.015)
    _write_fixture(
        "speaker_nn1",
        wav,
        "I am speaking slowly and carefully so you can hear every word.",
    )


# ── README ────────────────────────────────────────────────────────────────────

def _write_readme() -> None:
    readme = FIXTURES_DIR / "README.md"
    readme.write_text(
        """\
# Voice Fixtures

Pre-recorded (or procedurally-generated) voice clips for offline pipeline
simulation and regression testing.

| File          | Type            | Duration | Notes                                   |
|---------------|-----------------|----------|-----------------------------------------|
| speaker_f1    | Real voice      | ~3 s     | Extracted from F5-TTS bundled test clip |
| speaker_m1    | Synthetic       | ~3 s     | Procedural male-range pitch (~120 Hz)   |
| speaker_nn1   | Synthetic       | ~4 s     | Procedural slower tempo (~175 Hz)       |

## Replacing with real LibriSpeech clips

For authentic voice-clone regression testing, replace the synthetic files with
real human voice clips:

1. Download LibriSpeech test-clean from https://www.openslr.org/12/
2. Choose one female speaker (e.g. 1089/), one male (e.g. 1284/), one
   non-native English speaker from a separate dataset (e.g. L2-ARCTIC).
3. Trim each to 5–30 s.  Rename to speaker_f1.wav / speaker_m1.wav / speaker_nn1.wav.
4. Write matching .txt transcript files.
5. Re-run: `uv run python scripts/generate_fixtures.py --replace-only`

Synthetic clips verify pipeline structure; real clips verify audible clone quality.
""",
        encoding="utf-8",
    )
    print(f"  {readme.name}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    replace_only = "--replace-only" not in sys.argv  # always regenerate by default

    print(f"Generating voice fixtures → {FIXTURES_DIR}\n")
    _make_speaker_f1()
    _make_speaker_m1()
    _make_speaker_nn1()
    _write_readme()
    print("\nDone.")
