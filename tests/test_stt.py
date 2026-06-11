"""Tests for talk2me.stt.

Model-loading tests are skipped if mlx_whisper is not available (CI without
weights). Integration transcription tests require the model weights to be
downloaded and are marked 'model'.
"""
import numpy as np
import pytest

from talk2me.stt import Transcriber, TranscriptResult


def test_transcript_result_fields():
    r = TranscriptResult(text="hello", no_speech_prob=0.1, avg_logprob=-0.5, latency_s=0.3)
    assert r.text == "hello"
    assert r.no_speech_prob == pytest.approx(0.1)


def test_transcriber_empty_input_returns_empty():
    t = Transcriber()
    # Empty wav should short-circuit before loading the model
    result = t.transcribe(np.array([], dtype=np.float32))
    assert result.text == ""
    assert result.no_speech_prob == pytest.approx(1.0)
    assert result.latency_s == pytest.approx(0.0)


def test_transcriber_none_input_returns_empty():
    t = Transcriber()
    result = t.transcribe(None)
    assert result.text == ""


def test_is_speech_with_high_no_speech_prob():
    t = Transcriber()
    r = TranscriptResult(text="", no_speech_prob=0.9, avg_logprob=-5.0, latency_s=0.1)
    assert not t.is_speech(r)


def test_is_speech_with_low_no_speech_prob_and_text():
    t = Transcriber()
    r = TranscriptResult(text="hello world", no_speech_prob=0.05, avg_logprob=-0.3, latency_s=0.2)
    assert t.is_speech(r)


def test_is_speech_empty_text_always_false():
    t = Transcriber()
    r = TranscriptResult(text="", no_speech_prob=0.1, avg_logprob=-0.3, latency_s=0.1)
    assert not t.is_speech(r)


@pytest.mark.model
def test_transcriber_real_audio():
    """Integration: transcribe 1 s of silence — should return empty/no-speech."""
    t = Transcriber()
    silence = np.zeros(16_000, dtype=np.float32)
    result = t.transcribe(silence)
    # Silence should trigger no-speech
    assert result.no_speech_prob > 0.5 or result.text == ""
    assert result.latency_s >= 0.0
