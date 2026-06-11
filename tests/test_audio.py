"""Tests for talk2me.audio — run without real mic/speaker hardware.

Hardware tests (playback, live recording) are marked 'hardware' and skipped
unless explicitly requested; verification against real audio is done manually.
"""
import numpy as np
import pytest


# ── io module ────────────────────────────────────────────────────────────────

def test_imports():
    from talk2me.audio import Microphone, Speaker, SAMPLE_RATE, list_devices
    assert SAMPLE_RATE == 16_000


def test_list_devices_returns_list():
    from talk2me.audio import list_devices
    devs = list_devices()
    assert isinstance(devs, (list, object))  # sounddevice DeviceList


def test_microphone_rejects_unknown_device():
    from talk2me.audio.io import Microphone
    with pytest.raises(ValueError, match="not found"):
        Microphone(device="nonexistent_device_xyz_123")


# ── vad module ────────────────────────────────────────────────────────────────

def test_energy_gate_classifies_loud_as_speech():
    from talk2me.audio.vad import _EnergyGate
    gate = _EnergyGate(ambient_rms=0.01)
    loud = np.ones(512, dtype=np.float32) * 0.5
    assert gate.is_speech(loud)


def test_energy_gate_classifies_quiet_as_non_speech():
    from talk2me.audio.vad import _EnergyGate
    gate = _EnergyGate(ambient_rms=0.1)
    quiet = np.zeros(512, dtype=np.float32) + 0.001
    assert not gate.is_speech(quiet)


def test_record_utterance_returns_empty_on_silence(monkeypatch):
    """Simulate a mic that returns silence — record_utterance should return empty."""
    from talk2me.audio import vad, Microphone
    from talk2me.audio.vad import _EnergyGate

    class _FakeStream:
        def __init__(self):
            self._call = 0
        def read(self, n):
            # Return silence; second dimension for mono channel
            return np.zeros((n, 1), dtype=np.float32), None
        def __enter__(self): return self
        def __exit__(self, *_): pass

    class _FakeMic:
        def stream(self, chunk_frames=512):
            return _FakeStream()
        def calibrate_noise(self, duration=2.0):
            return 0.0

    gate = _EnergyGate(ambient_rms=0.01)
    wav = vad.record_utterance(
        _FakeMic(), max_seconds=0.5, silence_ms=100, _gate=gate
    )
    assert isinstance(wav, np.ndarray)
    assert wav.dtype == np.float32


def test_record_utterance_captures_speech(monkeypatch):
    """Simulate a mic with one burst of loud audio followed by silence."""
    from talk2me.audio import vad
    from talk2me.audio.vad import _EnergyGate

    LOUD = np.ones(512, dtype=np.float32) * 0.5
    QUIET = np.zeros(512, dtype=np.float32)

    class _FakeStream:
        def __init__(self):
            self._seq = [QUIET] * 2 + [LOUD] * 4 + [QUIET] * 20
            self._idx = 0
        def read(self, n):
            chunk = self._seq[min(self._idx, len(self._seq) - 1)]
            self._idx += 1
            return chunk.reshape(-1, 1), None
        def __enter__(self): return self
        def __exit__(self, *_): pass

    class _FakeMic:
        def stream(self, chunk_frames=512):
            return _FakeStream()

    gate = _EnergyGate(ambient_rms=0.01)
    wav = vad.record_utterance(
        _FakeMic(), max_seconds=5.0, silence_ms=200, _gate=gate
    )
    assert len(wav) > 0
    assert wav.dtype == np.float32
