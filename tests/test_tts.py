"""Tests for talk2me.tts — VoiceCloner and ReferenceBuffer.

Model-loading/synthesis tests are marked 'model' and skipped without weights.
All structural and logic tests run without hardware or model downloads.
"""
import numpy as np
import pytest

from talk2me.tts.voice_cloner import VoiceCloner, SAMPLE_RATE, _resample
from talk2me.tts.reference_buffer import ReferenceBuffer
from talk2me.stt.whisper import TranscriptResult


# ── VoiceCloner structural ────────────────────────────────────────────────────

def test_voice_cloner_sample_rate():
    vc = VoiceCloner()
    assert vc.sample_rate == 24_000


def test_voice_cloner_empty_text_returns_empty():
    vc = VoiceCloner()
    # synthesize() must short-circuit before loading model when text is blank
    result = vc.synthesize("", np.zeros(24_000, dtype=np.float32), "some ref text")
    assert isinstance(result, np.ndarray)
    assert len(result) == 0


def test_voice_cloner_whitespace_only_returns_empty():
    vc = VoiceCloner()
    result = vc.synthesize("   ", np.zeros(24_000, dtype=np.float32), "ref text")
    assert len(result) == 0


# ── _resample utility ─────────────────────────────────────────────────────────

def test_resample_passthrough_same_rate():
    wav = np.ones(1000, dtype=np.float32)
    out = _resample(wav, 16_000, 16_000)
    np.testing.assert_array_equal(out, wav)


def test_resample_upsample_length():
    wav = np.ones(16_000, dtype=np.float32)
    out = _resample(wav, 16_000, 24_000)
    # 24000/16000 * 16000 = 24000 samples
    assert len(out) == pytest.approx(24_000, abs=10)
    assert out.dtype == np.float32


def test_resample_downsample_length():
    wav = np.ones(24_000, dtype=np.float32)
    out = _resample(wav, 24_000, 16_000)
    assert len(out) == pytest.approx(16_000, abs=10)


# ── ReferenceBuffer ───────────────────────────────────────────────────────────

def _make_result(text="hello world", avg_logprob=-0.5, no_speech_prob=0.05):
    return TranscriptResult(
        text=text,
        no_speech_prob=no_speech_prob,
        avg_logprob=avg_logprob,
        latency_s=0.1,
    )


def _loud_wav(seconds=1.0, sr=16_000, amplitude=0.3):
    return (np.random.default_rng(42).uniform(-1, 1, int(seconds * sr)) * amplitude).astype(np.float32)


def test_buffer_starts_empty():
    buf = ReferenceBuffer()
    assert buf.tier == 0
    assert buf.voiced_seconds == pytest.approx(0.0)
    wav, text = buf.best_reference()
    assert wav is None
    assert text is None


def test_buffer_rejects_silent_segment():
    buf = ReferenceBuffer()
    silent = np.zeros(16_000, dtype=np.float32)
    accepted = buf.push(silent, _make_result())
    assert not accepted
    assert buf.tier == 0


def test_buffer_rejects_clipped_segment():
    buf = ReferenceBuffer()
    clipped = np.ones(16_000, dtype=np.float32)  # peak == 1.0 > 0.95
    accepted = buf.push(clipped, _make_result())
    assert not accepted


def test_buffer_rejects_bad_transcript():
    buf = ReferenceBuffer()
    wav = _loud_wav(1.0)
    bad_result = _make_result(text="hi", avg_logprob=-3.0)  # below threshold
    accepted = buf.push(wav, bad_result)
    assert not accepted


def test_buffer_rejects_empty_text():
    buf = ReferenceBuffer()
    wav = _loud_wav(1.0)
    empty_result = _make_result(text="", avg_logprob=-0.5)
    accepted = buf.push(wav, empty_result)
    assert not accepted


def test_buffer_accepts_good_segment():
    buf = ReferenceBuffer()
    wav = _loud_wav(2.0)
    accepted = buf.push(wav, _make_result())
    assert accepted
    assert buf.voiced_seconds == pytest.approx(2.0, abs=0.05)


def test_buffer_tier_advances():
    buf = ReferenceBuffer(tier_thresholds=(3.0, 9.0, 20.0))
    # push enough loud audio to cross tier thresholds
    assert buf.tier == 0
    buf.push(_loud_wav(2.0), _make_result(text="one"))
    assert buf.tier == 0  # < 3 s
    buf.push(_loud_wav(2.0), _make_result(text="two"))
    assert buf.tier == 1  # ≥ 3 s
    buf.push(_loud_wav(5.0), _make_result(text="three"))
    assert buf.tier == 2  # ≥ 9 s
    buf.push(_loud_wav(12.0), _make_result(text="four"))
    assert buf.tier == 3  # ≥ 20 s


def test_buffer_best_reference_concatenates():
    buf = ReferenceBuffer()
    wav1 = _loud_wav(2.0)
    wav2 = _loud_wav(2.0)
    buf.push(wav1, _make_result(text="first utterance", avg_logprob=-0.3))
    buf.push(wav2, _make_result(text="second utterance", avg_logprob=-0.5))
    ref_wav, ref_text = buf.best_reference()
    assert ref_wav is not None
    assert len(ref_wav) == len(wav1) + len(wav2)
    assert "first utterance" in ref_text
    assert "second utterance" in ref_text


def test_buffer_quality_selection_prefers_high_logprob():
    """With max_reference_seconds=2.0 and two 2-second segments, the higher-
    quality one should be chosen."""
    buf = ReferenceBuffer(max_reference_seconds=2.0)
    low_q = _loud_wav(2.0)
    high_q = _loud_wav(2.0)
    buf.push(low_q, _make_result(text="low quality", avg_logprob=-1.2))
    buf.push(high_q, _make_result(text="high quality", avg_logprob=-0.1))
    ref_wav, ref_text = buf.best_reference()
    # Should include the high-quality segment
    assert "high quality" in ref_text


def test_buffer_respects_max_reference_cap():
    """Total reference duration must not exceed max_reference_seconds."""
    max_s = 5.0
    sr = 16_000
    buf = ReferenceBuffer(max_reference_seconds=max_s)
    for i in range(4):
        buf.push(_loud_wav(3.0), _make_result(text=f"utterance {i}"))
    ref_wav, _ = buf.best_reference()
    assert ref_wav is not None
    duration_s = len(ref_wav) / sr
    assert duration_s <= max_s + 3.0  # at most one segment can spill over the cap


def test_buffer_reset_clears_all():
    buf = ReferenceBuffer()
    buf.push(_loud_wav(5.0), _make_result(text="something"))
    assert buf.voiced_seconds > 0
    buf.reset()
    assert buf.voiced_seconds == pytest.approx(0.0)
    assert buf.tier == 0
    wav, text = buf.best_reference()
    assert wav is None
    assert text is None


def test_buffer_cache_is_invalidated_on_push():
    buf = ReferenceBuffer()
    buf.push(_loud_wav(2.0), _make_result(text="first"))
    ref1, _ = buf.best_reference()
    buf.push(_loud_wav(2.0), _make_result(text="second"))
    ref2, _ = buf.best_reference()
    # Second call should return a longer clip
    assert len(ref2) > len(ref1)


# ── model integration (requires weights) ─────────────────────────────────────

@pytest.mark.model
def test_voice_cloner_synthesizes_real_audio():
    """Integration: synthesize a short phrase from a bundled reference clip."""
    import pkgutil
    import tempfile
    import soundfile as sf

    wav_bytes = pkgutil.get_data("f5_tts_mlx", "tests/test_en_1_ref_short.wav")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name
    ref_audio, sr = sf.read(tmp_path)
    ref_audio = ref_audio.astype(np.float32)

    vc = VoiceCloner()
    out = vc.synthesize(
        "What brought you here today?",
        reference_wav=ref_audio,
        reference_text="Some call me nature, others call me mother nature.",
        reference_sample_rate=sr,
    )
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert len(out) > 0
    duration_s = len(out) / vc.sample_rate
    assert 0.5 < duration_s < 10.0
