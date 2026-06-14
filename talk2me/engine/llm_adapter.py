"""Optional LLM-adaptive question personalizer for Talk2Me.

Feature 10: wraps mlx_lm (Llama-3.2-3B-Instruct-4bit by default) to lightly
rewrite a bank question so it references something the participant actually said,
without inventing new topics or drifting from the artist's intended tone.

Gated by ``engine.llm: true`` in exhibit.yaml. When the flag is false this
module is never imported and the LLM is never loaded.

Public API:
    LLMAdapter(model_id, max_new_tokens) -> LLMAdapter
    LLMAdapter.warm() -> None
    LLMAdapter.personalize(candidate_question, transcript_history, phase) -> str
"""
from __future__ import annotations

import time
from typing import Optional

_DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
_DEFAULT_MAX_TOKENS = 60

# Tight system prompt: preserves tone, forbids topic invention.
_SYSTEM_PROMPT = (
    "You are editing a single interview question for an art installation. "
    "RULES — follow every one:\n"
    "1. Preserve the exact meaning, emotional intensity, and phase-appropriate tone.\n"
    "2. Do NOT introduce any new topics or lines of questioning not already present.\n"
    "3. You may reference ONE specific detail the participant just mentioned to make "
    "the question feel personal, but only if it fits completely naturally.\n"
    "4. Return ONLY the revised question text — no preamble, no explanation, "
    "no surrounding quotes."
)


class LLMAdapter:
    """Personalizes a bank question using a local mlx_lm instruct model.

    Load cost: ~1–2 s on first construction (4-bit Llama-3.2-3B).
    Inference cost: ~0.2–0.5 s per call on M2/M3 — within the ≤1 s budget.
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        max_new_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self._load()

    def _load(self) -> None:
        try:
            from mlx_lm import load as mlx_lm_load
        except ImportError as exc:
            raise RuntimeError(
                "mlx-lm is required for LLM-adaptive question selection. "
                "Install with: pip install 'talk2me[llm]'  or  pip install mlx-lm"
            ) from exc
        self._model, self._tokenizer = mlx_lm_load(self._model_id)

    def warm(self) -> None:
        """JIT-warm the model so the participant's first turn is fast."""
        self.personalize("Are you comfortable here?", [], phase=1)

    def personalize(
        self,
        candidate_question: str,
        transcript_history: list[str],
        phase: int,
    ) -> str:
        """Return a lightly adapted version of *candidate_question*.

        The question is always returned as a string. If the model output is
        empty, contains no question mark, or is more than 3× longer than the
        input, the original *candidate_question* is returned unchanged (drift
        guard).

        Args:
            candidate_question: The raw question selected from the bank.
            transcript_history: All participant transcripts so far (earliest first).
            phase: Current conversation phase (1=calibration, 2=personal, 3=confrontational).

        Returns:
            Adapted question text, or *candidate_question* on failure/drift.
        """
        if self._model is None or self._tokenizer is None:
            return candidate_question

        recent = transcript_history[-3:] if transcript_history else []
        context = " / ".join(recent) if recent else "(no transcript yet)"

        user_msg = (
            f"Phase {phase} question to personalise:\n"
            f'"{candidate_question}"\n\n'
            f"Participant context (last ≤3 turns, earliest first):\n{context}\n\n"
            "Return only the revised question:"
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        try:
            from mlx_lm import generate as mlx_lm_generate

            prompt = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            t0 = time.perf_counter()
            raw = mlx_lm_generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self._max_new_tokens,
                verbose=False,
            )
            elapsed = time.perf_counter() - t0

            adapted = raw.strip().strip('"').strip("'").strip()

            # Drift guard: empty, too long, or no question mark → fall back
            max_len = len(candidate_question) * 3
            if not adapted or len(adapted) > max_len or "?" not in adapted:
                print(
                    f"[llm] {elapsed:.2f}s  drift-guard rejected output — "
                    f"keeping original: {candidate_question!r}"
                )
                return candidate_question

            print(
                f"[llm] {elapsed:.2f}s  orig={candidate_question!r}  "
                f"adapted={adapted!r}"
            )
            return adapted

        except Exception as exc:
            print(f"[llm] Adapter error ({exc!r}) — falling back to original")
            return candidate_question
