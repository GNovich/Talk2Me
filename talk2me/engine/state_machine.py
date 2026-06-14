"""Conversation state machine for the Talk2Me art installation.

Feature 9: tracks turn count, phase, elapsed time, and transcript history,
then selects the next question from the QuestionBank (Feature 8) using
topic-hint biasing. Replaces the hard-coded placeholder question in app.py.

Feature 10 wired: optional LLMAdapter passed at construction time; when
present, next_question() personalizes the bank selection before returning it.
Both the original and adapted texts are logged for curatorial review.

Public API:
    ConversationEngine.next_question() -> str
    ConversationEngine.record_turn(transcript, question_asked) -> None
    ConversationEngine.should_reset() -> bool
    ConversationEngine.reset() -> None
    ConversationEngine.turn -> int
    ConversationEngine.phase -> int
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from talk2me.engine.question_bank import QuestionBank

if TYPE_CHECKING:
    from talk2me.engine.llm_adapter import LLMAdapter

# How many recent transcript words to pass as topic hints
_TOPIC_WINDOW_TURNS = 3
_MIN_WORD_LEN = 4  # skip short stopwords


class ConversationEngine:
    """Directs the arc of the conversation.

    Phase transitions are driven by turn count:
        Phase 1 (calibration): turns 1 – calibration_turns
        Phase 2 (personal):    turns calibration_turns+1 – calibration_turns+personal_turns
        Phase 3 (confrontational): remaining turns

    The engine is pure logic — no audio dependencies — so it is fully
    unit-testable by feeding scripted transcript strings.
    """

    def __init__(
        self,
        question_bank: QuestionBank,
        calibration_turns: int = 2,
        personal_turns: int = 3,
        idle_timeout_seconds: float = 60.0,
        llm_adapter: Optional["LLMAdapter"] = None,
    ) -> None:
        self._bank = question_bank
        self._calibration_turns = calibration_turns
        self._personal_turns = personal_turns
        self._idle_timeout = idle_timeout_seconds
        self._llm_adapter = llm_adapter

        self._turn: int = 0
        self._phase: int = 1
        self._transcript_history: list[str] = []
        self._used_ids: set[str] = set()
        self._last_activity: float = time.monotonic()

    # ── public state ──────────────────────────────────────────────────────────

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def phase(self) -> int:
        return self._phase

    # ── main interface ────────────────────────────────────────────────────────

    def next_question(self) -> str:
        """Return the next question text for the current phase.

        The question is drawn from the bank with topic-hint biasing from recent
        transcripts. When an LLMAdapter is configured it lightly personalizes
        the bank text; both originals and adaptations are logged.
        The question ID is tracked in used_ids; it is not recorded as a
        completed turn until record_turn() is called.
        """
        hints = self._topic_hints()
        qid, original_text = self._bank.select(
            phase=self._phase,
            used_ids=self._used_ids,
            topic_hints=hints,
        )
        self._used_ids.add(qid)

        if self._llm_adapter is not None:
            adapted_text = self._llm_adapter.personalize(
                original_text, self._transcript_history, self._phase
            )
            if adapted_text != original_text:
                print(
                    f"[engine] LLM adapted [{qid}]: {original_text!r} "
                    f"→ {adapted_text!r}"
                )
            return adapted_text

        return original_text

    def record_turn(self, transcript: str, question_asked: str = "") -> None:
        """Update state after a completed turn.

        Args:
            transcript: What the participant said (Whisper transcript).
            question_asked: The question that was asked (for the session log).
        """
        self._turn += 1
        self._last_activity = time.monotonic()
        if transcript.strip():
            self._transcript_history.append(transcript.strip())

        # Advance phase based on turn count
        self._phase = self._phase_for_turn(self._turn + 1)  # next turn's phase

    def should_reset(self) -> bool:
        """Return True if idle timeout has elapsed since the last turn."""
        return (time.monotonic() - self._last_activity) >= self._idle_timeout

    def reset(self) -> None:
        """Clear all state for the next participant."""
        self._turn = 0
        self._phase = 1
        self._transcript_history = []
        self._used_ids = set()
        self._last_activity = time.monotonic()

    # ── internals ────────────────────────────────────────────────────────────

    def _phase_for_turn(self, turn: int) -> int:
        """Return the phase that applies at the start of `turn`."""
        if turn <= self._calibration_turns:
            return 1
        if turn <= self._calibration_turns + self._personal_turns:
            return 2
        return 3

    def _topic_hints(self) -> list[str]:
        """Extract candidate keyword hints from the most recent turns."""
        recent = self._transcript_history[-_TOPIC_WINDOW_TURNS:]
        words: list[str] = []
        for text in recent:
            for w in text.lower().split():
                clean = w.strip(".,!?;:'\"()[]")
                if len(clean) >= _MIN_WORD_LEN:
                    words.append(clean)
        return words
