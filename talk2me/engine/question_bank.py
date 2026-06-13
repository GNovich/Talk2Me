"""Curated question bank for the Talk2Me conversation engine.

Feature 8: load and select from three-phase question banks stored as YAML.
The bank is human-editable and hot-reloadable without restarting the installation.

Public API:
    QuestionBank.load(questions_dir) -> None
    QuestionBank.select(phase, used_ids, topic_hints=[]) -> tuple[str, str]
    QuestionBank.reload() -> None
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import yaml


_PHASE_FILES = {
    1: "calibration.yaml",
    2: "personal.yaml",
    3: "confrontational.yaml",
}


def _validate_entry(entry: dict, source: str) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"{source}: entry is not a dict: {entry!r}")
    if "text" not in entry or not str(entry["text"]).strip():
        raise ValueError(f"{source}: entry missing 'text': {entry!r}")
    if "id" not in entry or not str(entry["id"]).strip():
        raise ValueError(f"{source}: entry missing 'id': {entry!r}")


class QuestionBank:
    """Loads and serves questions from three-phase YAML files.

    Questions are organised into phases:
        Phase 1 — calibration (harvest reference audio + ease in)
        Phase 2 — personal (lightly intrusive)
        Phase 3 — confrontational (intense, in the participant's own voice)

    Selection is deterministic non-repeating within a session; when all entries
    in a phase are exhausted the bank cycles. Topic hints bias selection toward
    entries whose topic_hooks overlap with recent transcript words.
    """

    def __init__(self) -> None:
        self._questions_dir: Optional[Path] = None
        self._bank: dict[int, list[dict]] = {1: [], 2: [], 3: []}

    def load(self, questions_dir: str | Path) -> None:
        """Load all phase YAML files from the given directory."""
        self._questions_dir = Path(questions_dir)
        self._load_all()

    def reload(self) -> None:
        """Re-read YAML files without restarting (for between-session tuning)."""
        if self._questions_dir is None:
            raise RuntimeError("Call load() before reload()")
        self._load_all()

    def _load_all(self) -> None:
        for phase, filename in _PHASE_FILES.items():
            path = self._questions_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Question file not found: {path}")
            with open(path, encoding="utf-8") as f:
                entries = yaml.safe_load(f) or []
            if not isinstance(entries, list):
                raise ValueError(f"{path}: expected a YAML list, got {type(entries).__name__}")
            for entry in entries:
                _validate_entry(entry, str(path))
            self._bank[phase] = entries

    def select(
        self,
        phase: int,
        used_ids: set[str],
        topic_hints: list[str] | None = None,
    ) -> tuple[str, str]:
        """Select the next question for the given phase.

        Args:
            phase: 1, 2, or 3.
            used_ids: Set of question IDs already asked this session.
                      The bank cycles (used_ids is cleared on exhaustion).
            topic_hints: Words from recent transcripts used to bias selection
                         toward topic-matched entries.

        Returns:
            (question_id, question_text)
        """
        if phase not in self._bank:
            raise ValueError(f"Unknown phase {phase!r}; must be 1, 2, or 3")

        pool = self._bank[phase]
        if not pool:
            raise RuntimeError(f"Phase {phase} question bank is empty")

        unused = [e for e in pool if e["id"] not in used_ids]
        if not unused:
            # Exhausted — cycle: clear all used IDs for this phase and start over
            used_ids -= {e["id"] for e in pool}
            unused = list(pool)

        hints = [h.lower() for h in (topic_hints or [])]

        if hints:
            # Score entries by number of matching topic_hooks
            def score(entry: dict) -> int:
                hooks = [h.lower() for h in entry.get("topic_hooks", [])]
                return sum(1 for w in hints if any(w in hook or hook in w for hook in hooks))

            max_score = max(score(e) for e in unused)
            if max_score > 0:
                best = [e for e in unused if score(e) == max_score]
                chosen = random.choice(best)
                return chosen["id"], str(chosen["text"])

        # No topic match — pick the first unused entry (deterministic order)
        chosen = unused[0]
        return chosen["id"], str(chosen["text"])

    @property
    def phases_loaded(self) -> list[int]:
        return [p for p, entries in self._bank.items() if entries]
