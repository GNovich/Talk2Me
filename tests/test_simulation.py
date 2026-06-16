"""Simulation tests — Feature 15.

Structure tests (no model weights) verify that SimulatedParticipant loads,
cycles through clips, and handles edge cases correctly.

Full simulation tests are marked @pytest.mark.simulation and require model
weights (@pytest.mark.model).  Run them with:
    uv run pytest -m simulation tests/test_simulation.py
"""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from tests.simulation import SimulatedParticipant, FIXTURES_DIR


# ── fixture sanity checks ─────────────────────────────────────────────────────

def test_fixtures_directory_exists():
    assert FIXTURES_DIR.is_dir(), (
        f"{FIXTURES_DIR} missing — run: uv run python scripts/generate_fixtures.py"
    )


@pytest.mark.parametrize("speaker", ["speaker_f1", "speaker_m1", "speaker_nn1"])
def test_fixture_files_present(speaker):
    wav = FIXTURES_DIR / f"{speaker}.wav"
    txt = FIXTURES_DIR / f"{speaker}.txt"
    assert wav.exists(), f"Missing fixture: {wav}"
    assert txt.exists(), f"Missing transcript: {txt}"
    assert len(txt.read_text(encoding="utf-8").strip()) > 0, f"Empty transcript: {txt}"


@pytest.mark.parametrize("speaker", ["speaker_f1", "speaker_m1", "speaker_nn1"])
def test_fixture_audio_properties(speaker):
    import soundfile as sf
    wav_path = FIXTURES_DIR / f"{speaker}.wav"
    wav, sr = sf.read(str(wav_path), dtype="float32")
    assert sr == 16_000, f"{speaker}.wav should be 16 kHz (got {sr})"
    assert len(wav) > 0
    duration_s = len(wav) / sr
    assert duration_s >= 1.0, f"{speaker}.wav too short: {duration_s:.2f}s"
    assert duration_s <= 35.0, f"{speaker}.wav too long: {duration_s:.2f}s"
    # Should have meaningful signal (not silent)
    rms = float(np.sqrt(np.mean(wav ** 2)))
    assert rms > 0.001, f"{speaker}.wav appears silent (RMS={rms:.4f})"


# ── SimulatedParticipant structure tests ──────────────────────────────────────

def test_participant_loads_clips():
    p = SimulatedParticipant("speaker_f1")
    assert p.clip_count >= 1


def test_participant_next_clip_returns_array_and_text():
    p = SimulatedParticipant("speaker_f1")
    wav, text = p.next_clip()
    assert isinstance(wav, np.ndarray)
    assert wav.dtype == np.float32
    assert len(wav) > 0
    assert isinstance(text, str)


def test_participant_cycles_clips():
    p = SimulatedParticipant("speaker_f1")
    count = p.clip_count
    # Advance past all clips; should wrap around
    for _ in range(count + 1):
        p.next_clip()
    # Still works after cycling
    wav, text = p.next_clip()
    assert len(wav) > 0


def test_participant_reset_restarts_cycle():
    p = SimulatedParticipant("speaker_f1")
    wav1, text1 = p.next_clip()
    p.reset()
    wav2, text2 = p.next_clip()
    np.testing.assert_array_equal(wav1, wav2)
    assert text1 == text2


def test_participant_unknown_speaker_raises():
    with pytest.raises(FileNotFoundError):
        SimulatedParticipant("speaker_nonexistent_xyz")


def test_participant_custom_fixtures_dir(tmp_path):
    """SimulatedParticipant works with a custom fixtures directory."""
    import soundfile as sf
    # Write a tiny fixture
    wav = (np.sin(np.linspace(0, 1, 16_000)) * 0.2).astype(np.float32)
    (tmp_path / "speaker_test.wav").parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(tmp_path / "speaker_test.wav"), wav, 16_000)
    (tmp_path / "speaker_test.txt").write_text("test utterance", encoding="utf-8")
    p = SimulatedParticipant("speaker_test", fixtures_dir=tmp_path)
    loaded_wav, text = p.next_clip()
    assert text == "test utterance"
    assert len(loaded_wav) == len(wav)


# ── run_turn without model (mocked components) ────────────────────────────────

class _MockTranscriptResult:
    def __init__(self, text="hello world"):
        self.text = text
        self.no_speech_prob = 0.05
        self.avg_logprob = -0.4
        self.latency_s = 0.01


class _MockTranscriber:
    def transcribe(self, wav):
        return _MockTranscriptResult("hello world simulation test")


class _MockCloner:
    sample_rate = 24_000
    def synthesize(self, text, reference_wav, reference_text, reference_sample_rate, **kw):
        return np.zeros(24_000, dtype=np.float32)  # 1 s silence


def test_run_turn_returns_expected_keys():
    from talk2me.tts.reference_buffer import ReferenceBuffer
    p = SimulatedParticipant("speaker_f1")
    ref_buffer = ReferenceBuffer()
    result = p.run_turn(_MockTranscriber(), ref_buffer, _MockCloner(), "What brought you here?")
    expected_keys = {"stt_s", "tts_s", "total_s", "transcript", "synth_duration_s", "tier", "voiced_s"}
    assert expected_keys <= result.keys()


def test_run_turn_total_equals_stt_plus_tts():
    from talk2me.tts.reference_buffer import ReferenceBuffer
    p = SimulatedParticipant("speaker_f1")
    ref_buffer = ReferenceBuffer()
    result = p.run_turn(_MockTranscriber(), ref_buffer, _MockCloner(), "What do you see?")
    assert result["total_s"] == pytest.approx(result["stt_s"] + result["tts_s"])


def test_run_turn_saves_wav(tmp_path):
    from talk2me.tts.reference_buffer import ReferenceBuffer
    p = SimulatedParticipant("speaker_f1")
    ref_buffer = ReferenceBuffer()
    p.run_turn(_MockTranscriber(), ref_buffer, _MockCloner(), "Some question?",
               out_dir=tmp_path, turn_idx=0)
    saved = list(tmp_path.glob("sim_speaker_f1_*.wav"))
    assert len(saved) == 1


def test_run_session_returns_one_result_per_question():
    p = SimulatedParticipant("speaker_f1")
    questions = ["Question one?", "Question two?", "Question three?"]
    results = p.run_session(_MockTranscriber(), _MockCloner(), questions)
    assert len(results) == 3
    for i, r in enumerate(results):
        assert r["turn"] == i
        assert r["question"] == questions[i]


# ── full simulation (requires model weights) ──────────────────────────────────

@pytest.mark.simulation
@pytest.mark.model
def test_simulation_f1_full_pipeline(tmp_path):
    """Run speaker_f1 through 3 simulated turns and verify latency and output."""
    from talk2me.stt.whisper import Transcriber
    from talk2me.tts.voice_cloner import VoiceCloner

    transcriber = Transcriber()
    cloner = VoiceCloner()

    questions = [
        "What brought you here today?",
        "What does this space feel like to you?",
        "Describe what you notice first when you enter a room.",
    ]

    p = SimulatedParticipant("speaker_f1")
    out_dir = tmp_path / "saved_audio"
    results = p.run_session(transcriber, cloner, questions, out_dir=out_dir)

    assert len(results) == 3
    for r in results:
        # Pipeline should complete in reasonable time (generous limit for CI/dev)
        assert r["total_s"] < 60.0, f"Latency too high: {r['total_s']:.1f}s"
        # Synthesis should produce audible output
        assert r["synth_duration_s"] > 0.1

    # Verify saved audio files
    saved = sorted(out_dir.glob("sim_speaker_f1_*.wav"))
    assert len(saved) == 3

    # Report latency summary
    avg_total = sum(r["total_s"] for r in results) / len(results)
    avg_stt = sum(r["stt_s"] for r in results) / len(results)
    avg_tts = sum(r["tts_s"] for r in results) / len(results)
    print(
        f"\n[simulation] speaker_f1 latency averages: "
        f"STT={avg_stt:.2f}s  TTS={avg_tts:.2f}s  total={avg_total:.2f}s"
    )


@pytest.mark.simulation
@pytest.mark.model
@pytest.mark.parametrize("speaker", ["speaker_m1", "speaker_nn1"])
def test_simulation_synthetic_speakers(speaker, tmp_path):
    """Verify the pipeline accepts synthetic fixture clips without crashing."""
    from talk2me.stt.whisper import Transcriber
    from talk2me.tts.voice_cloner import VoiceCloner

    transcriber = Transcriber()
    cloner = VoiceCloner()

    p = SimulatedParticipant(speaker)
    out_dir = tmp_path / "saved_audio"
    results = p.run_session(
        transcriber, cloner,
        ["What do you feel?", "Tell me more."],
        out_dir=out_dir,
    )
    assert len(results) == 2
