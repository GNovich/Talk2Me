"""VAD-gated utterance recorder.

Uses Silero VAD (ONNX) for speech onset/offset detection. Falls back to a
simple energy threshold if the ONNX runtime or model is unavailable.

Public API:
    record_utterance(mic, max_seconds, silence_ms) -> np.ndarray
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .io import Microphone, SAMPLE_RATE

# Silero VAD chunk size requirement
_SILERO_CHUNK_SAMPLES = 512   # 32 ms at 16 kHz
_SILERO_SAMPLE_RATE = 16_000

# Energy-fallback gate: speech if RMS above this multiple of ambient
_ENERGY_RATIO = 1.8


def _load_silero() -> Optional[object]:
    """Try to load Silero VAD ONNX session; return None on failure."""
    try:
        import onnxruntime as ort
        from pathlib import Path
        import urllib.request
        import tempfile, os

        model_url = (
            "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
        )
        cache_dir = Path.home() / ".cache" / "talk2me"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "silero_vad.onnx"

        if not model_path.exists():
            # Silero ONNX model is ~1.8 MB; download once
            urllib.request.urlretrieve(model_url, model_path)

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        session = ort.InferenceSession(str(model_path), sess_options=opts,
                                       providers=["CPUExecutionProvider"])
        return session
    except Exception:
        return None


class _SileroGate:
    """Stateful Silero VAD wrapper."""

    def __init__(self, session):
        self._session = session
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._sr = np.array([_SILERO_SAMPLE_RATE], dtype=np.int64)

    def is_speech(self, chunk: np.ndarray, threshold: float = 0.5) -> bool:
        if len(chunk) < _SILERO_CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, _SILERO_CHUNK_SAMPLES - len(chunk)))
        chunk = chunk[:_SILERO_CHUNK_SAMPLES].reshape(1, -1)
        outs = self._session.run(
            None,
            {"input": chunk, "h": self._h, "c": self._c, "sr": self._sr},
        )
        prob, self._h, self._c = outs[0], outs[1], outs[2]
        return bool(prob[0][0] >= threshold)

    def reset(self) -> None:
        self._h[:] = 0
        self._c[:] = 0


class _EnergyGate:
    """Simple RMS energy fallback gate."""

    def __init__(self, ambient_rms: float):
        self._threshold = ambient_rms * _ENERGY_RATIO

    def is_speech(self, chunk: np.ndarray, threshold: float = 0.5) -> bool:
        rms = float(np.sqrt(np.mean(chunk**2)))
        return rms > self._threshold

    def reset(self) -> None:
        pass


def _make_gate(mic: Microphone, calibration_seconds: float):
    session = _load_silero()
    if session is not None:
        return _SileroGate(session)
    # Fallback: measure ambient noise
    ambient = mic.calibrate_noise(calibration_seconds)
    return _EnergyGate(ambient)


def record_utterance(
    mic: Microphone,
    max_seconds: float = 30.0,
    silence_ms: float = 800.0,
    calibration_seconds: float = 0.0,
    _gate=None,
) -> np.ndarray:
    """Record one utterance, return float32 numpy array at SAMPLE_RATE Hz.

    Listens for speech onset via VAD, captures until `silence_ms` of trailing
    silence, or until `max_seconds` total capture time.

    Args:
        mic: Microphone instance to read from.
        max_seconds: Hard cap on utterance length.
        silence_ms: Milliseconds of continuous non-speech that ends the utterance.
        calibration_seconds: Ambient-noise calibration duration for energy fallback.
        _gate: Pre-built VAD gate (avoids re-loading for each call).

    Returns:
        float32 numpy array of captured audio, possibly empty if nothing detected.
    """
    if _gate is None:
        _gate = _make_gate(mic, calibration_seconds)

    chunk_samples = _SILERO_CHUNK_SAMPLES
    silence_chunks_needed = max(1, int((silence_ms / 1000.0) * SAMPLE_RATE / chunk_samples))
    max_chunks = int(max_seconds * SAMPLE_RATE / chunk_samples)

    frames: list[np.ndarray] = []
    speaking = False
    silence_chunks = 0

    with mic.stream(chunk_frames=chunk_samples) as stream:
        _gate.reset()
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_samples)
            chunk = chunk[:, 0].astype(np.float32)

            is_speech = _gate.is_speech(chunk)

            if is_speech:
                speaking = True
                silence_chunks = 0
                frames.append(chunk)
            elif speaking:
                frames.append(chunk)
                silence_chunks += 1
                if silence_chunks >= silence_chunks_needed:
                    break
            # If not yet speaking, keep waiting (pre-speech chunks are discarded)

    if not frames:
        return np.array([], dtype=np.float32)

    wav = np.concatenate(frames)
    # Trim trailing silence
    trim_samples = silence_chunks * chunk_samples
    if trim_samples and trim_samples < len(wav):
        wav = wav[:-trim_samples]
    return wav


def build_gate(mic: Microphone, calibration_seconds: float = 2.0):
    """Pre-build a VAD gate for reuse across multiple record_utterance calls."""
    return _make_gate(mic, calibration_seconds)
