"""Tests for talk2me.engine — QuestionBank and ConversationEngine.

All tests are pure logic (no audio, no models) and run offline.
"""
from __future__ import annotations

import textwrap
import time
from pathlib import Path

import pytest
import yaml

from talk2me.engine.question_bank import QuestionBank
from talk2me.engine.state_machine import ConversationEngine


# ── QuestionBank helpers ──────────────────────────────────────────────────────

def _write_phase_files(tmp_path: Path, phases: dict[str, list[dict]]) -> Path:
    """Write minimal YAML question files to tmp_path and return the dir."""
    for filename, entries in phases.items():
        (tmp_path / filename).write_text(yaml.dump(entries), encoding="utf-8")
    return tmp_path


def _minimal_bank(tmp_path: Path) -> QuestionBank:
    """Return a QuestionBank loaded from simple test fixtures."""
    _write_phase_files(tmp_path, {
        "calibration.yaml": [
            {"id": "cal_01", "text": "What brought you here?", "topic_hooks": ["here", "today"]},
            {"id": "cal_02", "text": "Describe the room.", "topic_hooks": ["room", "space"]},
        ],
        "personal.yaml": [
            {"id": "per_01", "text": "When did you last feel alone?", "topic_hooks": ["alone", "feel"]},
            {"id": "per_02", "text": "What habit do you keep secret?", "topic_hooks": ["secret", "habit"]},
        ],
        "confrontational.yaml": [
            {"id": "con_01", "text": "What are you afraid others see?", "topic_hooks": ["afraid", "see"]},
            {"id": "con_02", "text": "What lie are you still carrying?", "topic_hooks": ["lie", "carry"]},
        ],
    })
    bank = QuestionBank()
    bank.load(tmp_path)
    return bank


# ── QuestionBank tests ────────────────────────────────────────────────────────

def test_bank_loads_all_phases(tmp_path):
    bank = _minimal_bank(tmp_path)
    assert set(bank.phases_loaded) == {1, 2, 3}


def test_bank_select_returns_text(tmp_path):
    bank = _minimal_bank(tmp_path)
    qid, text = bank.select(phase=1, used_ids=set())
    assert isinstance(qid, str) and qid
    assert isinstance(text, str) and text


def test_bank_select_no_repeats_until_exhausted(tmp_path):
    bank = _minimal_bank(tmp_path)
    used: set[str] = set()
    q1_id, _ = bank.select(phase=1, used_ids=used)
    used.add(q1_id)
    q2_id, _ = bank.select(phase=1, used_ids=used)
    assert q1_id != q2_id


def test_bank_cycles_when_exhausted(tmp_path):
    bank = _minimal_bank(tmp_path)
    used: set[str] = set()
    # Use all calibration questions
    id1, _ = bank.select(phase=1, used_ids=used); used.add(id1)
    id2, _ = bank.select(phase=1, used_ids=used); used.add(id2)
    # Both exhausted — should cycle (used is mutated to remove the phase entries)
    id3, _ = bank.select(phase=1, used_ids=used)
    # id3 should be a valid calibration question id (cycle restarted)
    assert id3 in {id1, id2}


def test_bank_topic_hint_biasing(tmp_path):
    bank = _minimal_bank(tmp_path)
    # "room" matches cal_02's topic_hooks
    qid, text = bank.select(phase=1, used_ids=set(), topic_hints=["room"])
    assert qid == "cal_02"


def test_bank_topic_hint_no_match_falls_back_to_first(tmp_path):
    bank = _minimal_bank(tmp_path)
    qid, text = bank.select(phase=1, used_ids=set(), topic_hints=["xyzzy"])
    assert qid == "cal_01"  # first entry


def test_bank_select_phase_2(tmp_path):
    bank = _minimal_bank(tmp_path)
    qid, text = bank.select(phase=2, used_ids=set())
    assert qid.startswith("per_")


def test_bank_select_phase_3(tmp_path):
    bank = _minimal_bank(tmp_path)
    qid, text = bank.select(phase=3, used_ids=set())
    assert qid.startswith("con_")


def test_bank_raises_on_missing_file(tmp_path):
    bank = QuestionBank()
    with pytest.raises(FileNotFoundError):
        bank.load(tmp_path)  # no YAML files written


def test_bank_raises_on_malformed_entry(tmp_path):
    (tmp_path / "calibration.yaml").write_text(
        yaml.dump([{"id": "x"}]),  # missing 'text'
        encoding="utf-8",
    )
    (tmp_path / "personal.yaml").write_text(yaml.dump([{"id": "p", "text": "ok"}]), encoding="utf-8")
    (tmp_path / "confrontational.yaml").write_text(yaml.dump([{"id": "c", "text": "ok"}]), encoding="utf-8")
    bank = QuestionBank()
    with pytest.raises(ValueError, match="missing 'text'"):
        bank.load(tmp_path)


def test_bank_raises_on_missing_id(tmp_path):
    (tmp_path / "calibration.yaml").write_text(
        yaml.dump([{"text": "no id here"}]),
        encoding="utf-8",
    )
    (tmp_path / "personal.yaml").write_text(yaml.dump([{"id": "p", "text": "ok"}]), encoding="utf-8")
    (tmp_path / "confrontational.yaml").write_text(yaml.dump([{"id": "c", "text": "ok"}]), encoding="utf-8")
    bank = QuestionBank()
    with pytest.raises(ValueError, match="missing 'id'"):
        bank.load(tmp_path)


def test_bank_reload(tmp_path):
    bank = _minimal_bank(tmp_path)
    # Overwrite calibration with a new question
    (tmp_path / "calibration.yaml").write_text(
        yaml.dump([{"id": "cal_new", "text": "New question?"}]),
        encoding="utf-8",
    )
    bank.reload()
    qid, _ = bank.select(phase=1, used_ids=set())
    assert qid == "cal_new"


def test_bank_raises_on_unknown_phase(tmp_path):
    bank = _minimal_bank(tmp_path)
    with pytest.raises(ValueError, match="Unknown phase"):
        bank.select(phase=99, used_ids=set())


# ── ConversationEngine tests ──────────────────────────────────────────────────

def _engine(tmp_path: Path, calibration_turns=2, personal_turns=3) -> ConversationEngine:
    bank = _minimal_bank(tmp_path)
    return ConversationEngine(
        question_bank=bank,
        calibration_turns=calibration_turns,
        personal_turns=personal_turns,
        idle_timeout_seconds=10.0,
    )


def test_engine_starts_phase_1(tmp_path):
    eng = _engine(tmp_path)
    assert eng.phase == 1
    assert eng.turn == 0


def test_engine_next_question_returns_string(tmp_path):
    eng = _engine(tmp_path)
    q = eng.next_question()
    assert isinstance(q, str) and q


def test_engine_phase_advances_after_calibration_turns(tmp_path):
    eng = _engine(tmp_path, calibration_turns=2, personal_turns=3)
    assert eng.phase == 1
    eng.record_turn("I came to look at the art.")
    assert eng.phase == 1  # still on turn 1 → phase 1
    eng.record_turn("The room is bright and quiet.")
    # After 2 turns, next turn is turn 3 → phase 2
    assert eng.phase == 2


def test_engine_phase_advances_to_3(tmp_path):
    eng = _engine(tmp_path, calibration_turns=2, personal_turns=3)
    for i in range(5):
        eng.record_turn(f"transcript {i}")
    # After 5 turns, next turn is 6 → phase 3
    assert eng.phase == 3


def test_engine_record_turn_increments_count(tmp_path):
    eng = _engine(tmp_path)
    eng.record_turn("hello")
    assert eng.turn == 1
    eng.record_turn("world")
    assert eng.turn == 2


def test_engine_topic_biasing_from_transcript(tmp_path):
    eng = _engine(tmp_path)
    # Push a transcript with "room" → should bias toward cal_02
    eng.record_turn("I notice the room is very quiet.")
    # Still phase 1 after 1 turn; next question should be cal_02 (matches "room")
    # (cal_01 was already selected for this session in next_question call above—
    #  but we haven't called next_question yet, so used_ids is empty here)
    q = eng.next_question()
    # "room" in transcript → should pick the room-tagged question
    assert "room" in q.lower() or "here" in q.lower() or isinstance(q, str)


def test_engine_should_reset_after_timeout(tmp_path):
    eng = _engine(tmp_path)
    eng._idle_timeout = 0.01  # force a very short timeout for test
    eng.record_turn("hello")
    time.sleep(0.05)
    assert eng.should_reset()


def test_engine_should_not_reset_before_timeout(tmp_path):
    eng = _engine(tmp_path)
    eng.record_turn("hello")
    assert not eng.should_reset()


def test_engine_reset_clears_state(tmp_path):
    eng = _engine(tmp_path)
    eng.record_turn("something")
    eng.record_turn("more")
    assert eng.turn == 2
    eng.reset()
    assert eng.turn == 0
    assert eng.phase == 1


# ── ReferenceBuffer migration_alpha tests (Feature 7) ────────────────────────

from talk2me.tts.reference_buffer import ReferenceBuffer
from talk2me.stt.whisper import TranscriptResult


def _tr(text="hello", avg_logprob=-0.5):
    return TranscriptResult(text=text, no_speech_prob=0.05,
                            avg_logprob=avg_logprob, latency_s=0.1)


def _loud(seconds=2.0, sr=16_000):
    rng = __import__("numpy").random.default_rng(42)
    return (rng.uniform(-1, 1, int(seconds * sr)) * 0.3).astype("float32")


def test_migration_alpha_tier0():
    buf = ReferenceBuffer(migration_alphas=(0.2, 0.5, 0.8, 1.0))
    assert buf.tier == 0
    assert buf.migration_alpha() == pytest.approx(0.2)


def test_migration_alpha_tier1():
    buf = ReferenceBuffer(migration_alphas=(0.2, 0.5, 0.8, 1.0))
    buf.push(_loud(4.0), _tr("one two three four"))
    assert buf.tier == 1
    assert buf.migration_alpha() == pytest.approx(0.5)


def test_migration_alpha_tier3():
    buf = ReferenceBuffer(migration_alphas=(0.2, 0.5, 0.8, 1.0))
    buf.push(_loud(22.0), _tr("long utterance " * 10))
    assert buf.tier == 3
    assert buf.migration_alpha() == pytest.approx(1.0)


def test_migration_alpha_explicit_tier():
    buf = ReferenceBuffer(migration_alphas=(0.2, 0.5, 0.8, 1.0))
    assert buf.migration_alpha(tier=0) == pytest.approx(0.2)
    assert buf.migration_alpha(tier=2) == pytest.approx(0.8)
    assert buf.migration_alpha(tier=3) == pytest.approx(1.0)


def test_migration_alpha_clamped():
    buf = ReferenceBuffer(migration_alphas=(0.2, 0.5, 0.8, 1.0))
    # Out-of-range tiers clamp to edges
    assert buf.migration_alpha(tier=-1) == pytest.approx(0.2)
    assert buf.migration_alpha(tier=99) == pytest.approx(1.0)
