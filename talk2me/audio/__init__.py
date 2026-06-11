"""Audio I/O and VAD layer."""

from .io import Microphone, Speaker, SAMPLE_RATE, list_devices
from .vad import record_utterance, build_gate

__all__ = ["Microphone", "Speaker", "SAMPLE_RATE", "list_devices", "record_utterance", "build_gate"]
