"""Zero-shot voice cloning via F5-TTS-MLX.

Feature 4: Replace the Tacotron2+WaveRNN+encoder chain with F5-TTS-MLX, a
flow-matching TTS that clones a voice from a few seconds of reference audio
with no separate vocoder and no fine-tuning.

Public API:
    VoiceCloner.synthesize(text, reference_wav, reference_text) -> np.ndarray
"""
from __future__ import annotations

import pkgutil
import time
from typing import Optional

import numpy as np

# F5-TTS native sample rate
SAMPLE_RATE = 24_000

# RMS target F5-TTS was trained at; normalise reference audio to this level
_TARGET_RMS = 0.1
_HOP_LENGTH = 256
_FRAMES_PER_SEC = SAMPLE_RATE / _HOP_LENGTH

DEFAULT_MODEL = "lucasnewman/f5-tts-mlx"

# Bundled test clip used for JIT warmup (avoids a network call at startup)
_WARMUP_TEXT = "Some call me nature, others call me mother nature."


def _resample(wav: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample float32 audio from src_rate to dst_rate using polyphase filter."""
    if src_rate == dst_rate:
        return wav
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(src_rate, dst_rate)
    up, down = dst_rate // g, src_rate // g
    resampled = resample_poly(wav.astype(np.float64), up, down)
    return resampled.astype(np.float32)


class VoiceCloner:
    """Wraps F5-TTS-MLX for zero-shot voice cloning.

    The model is loaded and JIT-warmed on the first synthesize() call (or
    explicitly via warm()). After that it is cached for the session lifetime.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        steps: int = 8,
        cfg_strength: float = 2.0,
        speed: float = 1.0,
        seed: Optional[int] = None,
    ):
        self._model_id = model
        self._steps = steps
        self._cfg_strength = cfg_strength
        self._speed = speed
        self._seed = seed
        self._f5tts = None       # loaded on first call
        self._cvt = None         # convert_char_to_pinyin helper

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def warm(self) -> None:
        """Load and JIT-compile the model with a throwaway synthesis.

        Call this at startup so the participant's first question is not slow.
        """
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._f5tts is not None:
            return

        from f5_tts_mlx.cfm import F5TTS
        from f5_tts_mlx.utils import convert_char_to_pinyin
        import mlx.core as mx
        import soundfile as sf
        import tempfile, io

        self._mx = mx
        self._cvt = convert_char_to_pinyin

        print(f"[TTS] Loading F5-TTS model: {self._model_id}")
        t0 = time.perf_counter()
        self._f5tts = F5TTS.from_pretrained(self._model_id)
        print(f"[TTS] Model loaded in {time.perf_counter() - t0:.1f}s")

        # JIT warmup: synthesize one sentence using the bundled test clip
        wav_bytes = pkgutil.get_data("f5_tts_mlx", "tests/test_en_1_ref_short.wav")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        ref_audio, _ = sf.read(tmp_path)
        ref_audio = mx.array(ref_audio.astype(np.float32))
        text_in = convert_char_to_pinyin([_WARMUP_TEXT + " " + _WARMUP_TEXT])
        warmup_dur = int(ref_audio.shape[0] / _HOP_LENGTH * 2)
        wave, _ = self._f5tts.sample(
            mx.expand_dims(ref_audio, axis=0),
            text=text_in,
            duration=warmup_dur,
            steps=self._steps,
            method="rk4",
            cfg_strength=self._cfg_strength,
            sway_sampling_coef=-1.0,
            seed=0,
        )
        mx.eval(wave)
        print("[TTS] F5-TTS JIT warmup complete.")

    def synthesize(
        self,
        text: str,
        reference_wav: np.ndarray,
        reference_text: str,
        reference_sample_rate: int = 24_000,
    ) -> np.ndarray:
        """Synthesize `text` in the voice described by `reference_wav`.

        Args:
            text: The text to speak.
            reference_wav: Float32 reference audio at `reference_sample_rate`.
            reference_text: Transcript of the reference audio (F5-TTS conditions
                on both audio and its transcript).
            reference_sample_rate: Sample rate of `reference_wav`.  The audio is
                resampled to 24 kHz internally if needed.

        Returns:
            Float32 numpy array at 24 kHz (SAMPLE_RATE).
        """
        if not text.strip():
            return np.array([], dtype=np.float32)

        self._ensure_loaded()
        mx = self._mx

        t0 = time.perf_counter()

        # 1. Prepare reference audio at 24 kHz
        ref = reference_wav.astype(np.float32)
        if reference_sample_rate != SAMPLE_RATE:
            ref = _resample(ref, reference_sample_rate, SAMPLE_RATE)

        # 2. Normalise RMS to F5-TTS training level
        rms = float(np.sqrt(np.mean(ref ** 2)))
        if rms > 0:
            ref = ref * (_TARGET_RMS / rms)

        ref_mx = mx.array(ref)

        # 3. Build combined text prompt: reference text + generation text
        combined = self._cvt([reference_text + " " + text])

        # 4. Estimate duration via frame-count heuristic
        ref_frames = ref_mx.shape[0] // _HOP_LENGTH
        ref_char_len = len(reference_text.encode("utf-8"))
        gen_char_len = len(text.encode("utf-8"))
        if ref_char_len > 0:
            gen_frames = int(ref_frames / ref_char_len * gen_char_len / self._speed)
        else:
            gen_frames = int(gen_char_len * _FRAMES_PER_SEC / 10)
        duration = ref_frames + gen_frames

        # 5. Sample
        wave, _ = self._f5tts.sample(
            mx.expand_dims(ref_mx, axis=0),
            text=combined,
            duration=duration,
            steps=self._steps,
            method="rk4",
            speed=self._speed,
            cfg_strength=self._cfg_strength,
            sway_sampling_coef=-1.0,
            seed=self._seed,
        )

        # 6. Trim reference prefix and materialise
        wave = wave[ref_mx.shape[0]:]
        mx.eval(wave)

        result = np.array(wave, dtype=np.float32)
        latency_s = time.perf_counter() - t0
        print(f"[TTS] Synthesized {len(result)/SAMPLE_RATE:.2f}s audio in {latency_s:.2f}s")

        return result
