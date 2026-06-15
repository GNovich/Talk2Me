"""Feature 13 — Telemetry, logging & latency dashboard.

TelemetryLogger writes one JSON line per completed turn to
``logs/telemetry_YYYY-MM-DD.jsonl``.  The log contains only timing and
phase metadata — no participant audio, text, or biometric data.

Usage in run_loop::

    tlog = TelemetryLogger(project_root / "logs")
    # after each turn:
    tlog.log_turn(turn=logical_turn, phase=engine.phase, tier=ref_buffer.tier,
                  alpha=alpha, stt_s=stt_s, tts_s=tts_s, play_s=play_s,
                  total_s=total_s)

CLI report (invoked via ``talk2me --report YYYY-MM-DD``)::

    TelemetryLogger(log_dir).print_report(date_str)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


class TelemetryLogger:
    """Append-only JSONL telemetry sink with a built-in report generator."""

    _OVER_BUDGET_S = 3.0
    _SLOW_S = 5.0

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _log_path(self, date_str: Optional[str] = None) -> Path:
        if date_str is None:
            date_str = time.strftime("%Y-%m-%d")
        return self._log_dir / f"telemetry_{date_str}.jsonl"

    def log_turn(
        self,
        *,
        turn: int,
        phase: int,
        tier: int,
        alpha: float,
        stt_s: float,
        tts_s: float,
        play_s: float,
        total_s: float,
    ) -> str:
        """Append one telemetry record and return the one-line health banner."""
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "turn": turn,
            "phase": phase,
            "tier": tier,
            "alpha": round(alpha, 3),
            "stt_s": round(stt_s, 3),
            "tts_s": round(tts_s, 3),
            "play_s": round(play_s, 3),
            "total_s": round(total_s, 3),
        }
        try:
            with open(self._log_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass  # non-fatal — never crash the loop for telemetry

        return self._health_banner(turn=turn, total_s=total_s)

    @staticmethod
    def _health_banner(*, turn: int, total_s: float) -> str:
        if total_s > TelemetryLogger._SLOW_S:
            status = "[ !! ]"
        elif total_s > TelemetryLogger._OVER_BUDGET_S:
            status = "[SLOW]"
        else:
            status = "[ OK ]"
        return f"{status} turn={turn} total={total_s:.2f}s"

    def session_summary(self, date_str: Optional[str] = None) -> dict:
        """Read the JSONL log and return per-stage averages for the day.

        Returns an empty dict if the file does not exist or has no records.
        """
        log_path = self._log_path(date_str)
        if not log_path.exists():
            return {}

        records: list[dict] = []
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not records:
            return {}

        def _avg(key: str) -> float:
            return sum(r[key] for r in records) / len(records)

        over_budget = sum(1 for r in records if r["total_s"] > self._OVER_BUDGET_S)
        return {
            "turns": len(records),
            "avg_stt_s": round(_avg("stt_s"), 3),
            "avg_tts_s": round(_avg("tts_s"), 3),
            "avg_play_s": round(_avg("play_s"), 3),
            "avg_total_s": round(_avg("total_s"), 3),
            "over_budget": over_budget,
            "budget_threshold_s": self._OVER_BUDGET_S,
        }

    def print_report(self, date_str: Optional[str] = None) -> None:
        """Print a human-readable table of per-turn latencies + session average."""
        if date_str is None:
            date_str = time.strftime("%Y-%m-%d")

        log_path = self._log_path(date_str)
        if not log_path.exists():
            print(f"[report] No telemetry log for {date_str}  (looked for {log_path})")
            return

        records: list[dict] = []
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not records:
            print(f"[report] Telemetry log exists but contains no records: {log_path}")
            return

        header = f"{'Turn':>4}  {'Phase':>5}  {'Tier':>4}  {'Alpha':>5}  {'STT':>6}  {'TTS':>6}  {'Play':>6}  {'Total':>6}  Status"
        print(f"\nTalk2Me telemetry report — {date_str}")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for r in records:
            total = r["total_s"]
            if total > self._SLOW_S:
                flag = "!!"
            elif total > self._OVER_BUDGET_S:
                flag = "SLOW"
            else:
                flag = ""
            print(
                f"{r['turn']:>4}  {r['phase']:>5}  {r['tier']:>4}  {r['alpha']:>5.2f}  "
                f"{r['stt_s']:>5.2f}s  {r['tts_s']:>5.2f}s  {r['play_s']:>5.2f}s  "
                f"{r['total_s']:>5.2f}s  {flag}"
            )

        summary = self.session_summary(date_str)
        print("-" * len(header))
        print(
            f"{'AVG':>4}  {'':>5}  {'':>4}  {'':>5}  "
            f"{summary['avg_stt_s']:>5.2f}s  {summary['avg_tts_s']:>5.2f}s  "
            f"{summary['avg_play_s']:>5.2f}s  {summary['avg_total_s']:>5.2f}s"
        )
        print(
            f"\n  {summary['turns']} turns  |  "
            f"{summary['over_budget']} over {self._OVER_BUDGET_S:.0f}s budget\n"
        )
