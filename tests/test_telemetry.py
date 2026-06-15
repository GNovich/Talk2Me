"""Tests for Feature 13 — Telemetry, logging & latency dashboard.

Verifies:
- log_turn() writes a valid JSON line with all expected fields
- session_summary() computes correct averages from the JSONL log
- session_summary() returns {} for a missing log file
- session_summary() returns {} for an empty log file
- --report doesn't crash on an empty log file
- health banner shows [OK] / [SLOW] / [!!] at the correct thresholds
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from talk2me.telemetry import TelemetryLogger


def _make_logger(tmp_path: Path) -> TelemetryLogger:
    return TelemetryLogger(tmp_path / "logs")


def _sample_turn(logger: TelemetryLogger, *, turn: int = 1, total_s: float = 2.0):
    return logger.log_turn(
        turn=turn,
        phase=1,
        tier=0,
        alpha=0.2,
        stt_s=0.5,
        tts_s=1.2,
        play_s=total_s - 0.5 - 1.2,
        total_s=total_s,
    )


# ── log_turn writes a valid JSON line ────────────────────────────────────────

def test_log_turn_creates_jsonl_file(tmp_path):
    logger = _make_logger(tmp_path)
    _sample_turn(logger)
    log_files = list((tmp_path / "logs").glob("telemetry_*.jsonl"))
    assert len(log_files) == 1


def test_log_turn_valid_json_line(tmp_path):
    logger = _make_logger(tmp_path)
    _sample_turn(logger, turn=3)
    log_file = next((tmp_path / "logs").glob("telemetry_*.jsonl"))
    lines = [l for l in log_file.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["turn"] == 3
    assert record["phase"] == 1
    assert record["tier"] == 0
    assert "stt_s" in record and "tts_s" in record and "play_s" in record
    assert "total_s" in record and "ts" in record


def test_log_turn_appends_multiple_lines(tmp_path):
    logger = _make_logger(tmp_path)
    for i in range(1, 4):
        _sample_turn(logger, turn=i)
    log_file = next((tmp_path / "logs").glob("telemetry_*.jsonl"))
    lines = [l for l in log_file.read_text().splitlines() if l.strip()]
    assert len(lines) == 3


# ── session_summary ──────────────────────────────────────────────────────────

def test_session_summary_correct_averages(tmp_path):
    logger = _make_logger(tmp_path)
    logger.log_turn(turn=1, phase=1, tier=0, alpha=0.2,
                    stt_s=0.4, tts_s=1.0, play_s=0.6, total_s=2.0)
    logger.log_turn(turn=2, phase=1, tier=1, alpha=0.5,
                    stt_s=0.6, tts_s=1.4, play_s=0.4, total_s=2.4)

    summary = logger.session_summary()
    assert summary["turns"] == 2
    assert abs(summary["avg_stt_s"] - 0.5) < 0.001
    assert abs(summary["avg_tts_s"] - 1.2) < 0.001
    assert abs(summary["avg_total_s"] - 2.2) < 0.001


def test_session_summary_missing_file_returns_empty(tmp_path):
    logger = _make_logger(tmp_path)
    result = logger.session_summary("2000-01-01")
    assert result == {}


def test_session_summary_empty_file_returns_empty(tmp_path):
    logger = _make_logger(tmp_path)
    log_path = (tmp_path / "logs") / "telemetry_2000-01-01.jsonl"
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    log_path.write_text("")
    result = logger.session_summary("2000-01-01")
    assert result == {}


def test_session_summary_over_budget_count(tmp_path):
    logger = _make_logger(tmp_path)
    # Two turns under budget, one over
    for total in (2.0, 2.5, 4.0):
        logger.log_turn(turn=1, phase=1, tier=0, alpha=1.0,
                        stt_s=0.5, tts_s=1.0, play_s=total - 1.5, total_s=total)
    summary = logger.session_summary()
    assert summary["over_budget"] == 1


# ── health banner ─────────────────────────────────────────────────────────────

def test_banner_ok(tmp_path):
    logger = _make_logger(tmp_path)
    banner = _sample_turn(logger, total_s=2.5)
    assert "OK" in banner


def test_banner_slow(tmp_path):
    logger = _make_logger(tmp_path)
    banner = _sample_turn(logger, total_s=4.0)
    assert "SLOW" in banner


def test_banner_critical(tmp_path):
    logger = _make_logger(tmp_path)
    banner = _sample_turn(logger, total_s=6.0)
    assert "!!" in banner


# ── print_report on empty / missing log ──────────────────────────────────────

def test_print_report_no_crash_on_missing_file(tmp_path, capsys):
    logger = _make_logger(tmp_path)
    logger.print_report("2000-01-01")  # must not raise
    out = capsys.readouterr().out
    assert "No telemetry log" in out


def test_print_report_no_crash_on_empty_file(tmp_path, capsys):
    logger = _make_logger(tmp_path)
    log_path = (tmp_path / "logs") / "telemetry_2000-01-01.jsonl"
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    log_path.write_text("")
    logger.print_report("2000-01-01")  # must not raise
    out = capsys.readouterr().out
    assert "no records" in out


def test_print_report_shows_turns(tmp_path, capsys):
    logger = _make_logger(tmp_path)
    import time as _time
    date_str = _time.strftime("%Y-%m-%d")
    logger.log_turn(turn=1, phase=1, tier=0, alpha=0.2,
                    stt_s=0.5, tts_s=1.2, play_s=0.3, total_s=2.0)
    logger.print_report(date_str)
    out = capsys.readouterr().out
    assert "1" in out
    assert "2.00" in out
