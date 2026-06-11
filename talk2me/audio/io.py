"""Microphone and Speaker wrappers for low-latency audio I/O.

All audio is float32 at SAMPLE_RATE Hz, mono, in-memory numpy arrays.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import sounddevice as sd


SAMPLE_RATE = 16_000  # Hz — matches Whisper expectation
_TAIL_PAD_SAMPLES = SAMPLE_RATE  # ~1 s pad to dodge sounddevice tail-cut bug


def list_devices() -> list[dict]:
    """Return all available audio devices."""
    return sd.query_devices()


def _find_device(name_or_index: str | int | None, kind: str) -> Optional[int]:
    """Resolve a device name or index to a sounddevice device index.

    Returns None to let sounddevice use its own default.
    """
    if name_or_index is None:
        return None
    if isinstance(name_or_index, int):
        return name_or_index
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if name_or_index.lower() in dev["name"].lower():
            if kind == "input" and dev["max_input_channels"] > 0:
                return i
            if kind == "output" and dev["max_output_channels"] > 0:
                return i
    raise ValueError(
        f"Audio device {name_or_index!r} not found for {kind}. "
        f"Available: {[d['name'] for d in devices]}"
    )


class Microphone:
    """Continuous microphone capture at SAMPLE_RATE Hz, mono float32."""

    def __init__(self, device: str | int | None = None, gain: float = 1.0):
        self._device_id = _find_device(device, "input")
        self._gain = gain

    def record(self, duration_seconds: float) -> np.ndarray:
        """Capture `duration_seconds` of audio, return float32 array."""
        frames = int(duration_seconds * SAMPLE_RATE)
        audio = sd.rec(
            frames,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self._device_id,
        )
        sd.wait()
        wav = audio[:, 0] * self._gain
        return wav.astype(np.float32)

    def stream(self, chunk_frames: int = 512):
        """Return a sounddevice InputStream configured for this mic.

        Caller is responsible for opening/closing via a context manager.
        """
        return sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_frames,
            device=self._device_id,
        )

    def calibrate_noise(self, duration_seconds: float = 2.0) -> float:
        """Record ambient noise and return its RMS energy for threshold setting."""
        wav = self.record(duration_seconds)
        rms = float(np.sqrt(np.mean(wav**2)))
        return rms


class Speaker:
    """Blocking and non-blocking audio playback."""

    def __init__(self, device: str | int | None = None, sample_rate: int = SAMPLE_RATE):
        self._device_id = _find_device(device, "output")
        self._sample_rate = sample_rate
        self._lock = threading.Lock()

    def play(self, wav: np.ndarray, blocking: bool = True) -> None:
        """Play a float32 waveform.

        Pads ~1 s of silence at the end to avoid sounddevice's tail-cut bug.
        """
        padded = np.pad(wav, (0, _TAIL_PAD_SAMPLES), mode="constant")
        with self._lock:
            sd.stop()
            sd.play(padded, samplerate=self._sample_rate, device=self._device_id,
                    blocking=blocking)

    def stop(self) -> None:
        sd.stop()
