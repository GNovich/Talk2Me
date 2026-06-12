# PROMPT QUEUE — updated 2026-06-12

> Tasks are pulled in binding order from `ROADMAP.md`. Do not start a feature
> before all earlier ones are `COMPLETED`. Max 3 active tasks at a time.

---

## TASK 1 — Repository teardown & MLX scaffold  (ROADMAP Feature 1)
**Status:** COMPLETED 2026-06-11

Legacy Tacotron2/WaveRNN/encoder stack removed; MLX package scaffold created.
See session_2026-06-11.md for details.

---

## TASK 2 — Audio I/O + VAD  (ROADMAP Feature 2)
**Status:** COMPLETED 2026-06-11

`Microphone`, `Speaker`, Silero VAD gate, `record_utterance()` API implemented.
Verified via unit tests; on-hardware audio verification deferred to exhibit Mac.

---

## TASK 3 — Local STT with Whisper-MLX  (ROADMAP Feature 3)
**Status:** COMPLETED 2026-06-11

`Transcriber.transcribe() -> TranscriptResult` wrapping mlx-whisper, model
warmed at startup, no_speech_prob + avg_logprob exposed for downstream use.

---

## TASK 4 — Zero-shot voice cloning with F5-TTS-MLX  (ROADMAP Feature 4)
**Status:** COMPLETED 2026-06-12

`VoiceCloner.synthesize(text, reference_wav, reference_text, reference_sample_rate)`
in `talk2me/tts/voice_cloner.py`. F5-TTS model lazy-loaded + JIT-warmed at startup.
16 kHz mic audio resampled to 24 kHz (F5-TTS native) via `scipy.signal.resample_poly`.
Audible synthesis verification deferred to exhibit Mac.

---

## TASK 5 — End-to-end single-turn loop  (ROADMAP Feature 5)
**Status:** COMPLETED 2026-06-12

`talk2me/app.py` implements the full `record → transcribe → synthesize → play`
loop with per-stage `time.perf_counter()` latency breakdown. Pre-warms both
models at startup. Edge cases handled. On-hardware latency verification deferred.

---

## TASK 6 — Rolling reference accumulation  (ROADMAP Feature 6)
**Status:** COMPLETED 2026-06-12

`ReferenceBuffer` in `talk2me/tts/reference_buffer.py`. Quality-ranked segment
accumulation with 3-tier sharpening (3 s → 9 s → 20 s). 14 unit tests pass.
Wired into `app.py`. Audible sharpening verification deferred to exhibit Mac.

---

## TASK 7 — Neutral→self voice migration  (ROADMAP Feature 7)
**Status:** PENDING — PLAN ONLY

**Plan:**
- Add a `neutral_seed_wav` (float32, 24 kHz) and `neutral_seed_text` (str) to
  `VoiceCloner` — loaded from `assets/neutral_seed.wav` if present; generated
  once with Kokoro (from PdfReader) or bundled from the F5-TTS test clip.
- Add `migration_alpha` parameter to `VoiceCloner.synthesize()`: 0.0 = full
  neutral seed, 1.0 = full participant reference.  At fractional values, blend
  the two reference clips by weighted concatenation (simple approach that F5-TTS
  already handles via the ref-audio conditioning; no need to interpolate latent
  space).
- `ReferenceBuffer` exposes `migration_alpha(tier)` helper: maps tier 0→0.2,
  1→0.5, 2→0.8, 3→1.0 (all configurable in `exhibit.yaml` under
  `engine.migration_alphas`).
- `app.py` checks `cfg.engine.migration` flag:
  - `false`: pass `alpha=1.0` always (clone from turn 1, current behaviour)
  - `true`: call `ref_buffer.migration_alpha(ref_buffer.tier)` each turn
- `assets/neutral_seed.wav` + `neutral_seed_text.txt` to be generated/placed by
  the operator; if missing, migration silently falls back to alpha=1.0 with a
  warning.
- Expose toggle in `config/exhibit.yaml`: `engine.migration: true|false`.
- **Verify:** scripted test that at tier 0 alpha=0.2 and at tier 3 alpha=1.0;
  audible ramp verification on exhibit Mac.

---

## TASK 8 — Curated question bank  (ROADMAP Feature 8)
**Status:** PENDING — PLAN ONLY

**Plan:**
- `questions/calibration.yaml`, `questions/personal.yaml`,
  `questions/confrontational.yaml` — each a YAML list of dicts with keys
  `text` (str) and optional `topic_hooks` (list of keywords for Feature 9
  matching).
- `talk2me/engine/question_bank.py`: `QuestionBank` class.
  - `load(questions_dir)` reads all three YAML files, validates schema
    (every entry must have `text`; phase inferred from filename).
  - `select(phase, used_ids, topic_hints=[])` returns the next unused entry
    in the given phase; deterministic non-repeating order; when exhausted,
    cycles (or signals exhaustion if Feature 9 wants to advance phase).
  - Hot-reloadable: `reload()` re-reads YAML files without restarting.
  - Generate first-draft question content for all three phases as part of
    implementation (artist will tune between sessions).
- Schema validator: raises `ValueError` on malformed entries.
- **Verify:** unit tests with synthetic YAML; assert correct selection order,
  no-repeat until exhaustion, phase boundary handling.

---

## TASK 9 — Conversation state machine  (ROADMAP Feature 9)
**Status:** PENDING — PLAN ONLY

**Plan:**
- `talk2me/engine/state_machine.py`: `ConversationEngine` class.
  - Tracks: `turn`, `phase` (1–3), `elapsed_s`, `transcript_history` (list of
    strings), `used_question_ids` (per phase).
  - Phase transitions: phase 1 → 2 after `calibration_turns` turns (from config);
    phase 2 → 3 after `personal_turns` turns.
  - `next_question(transcript_history) -> str`: calls `QuestionBank.select()`
    biased toward entries whose `topic_hooks` match recent transcripts
    (naive keyword scan — `str.lower()` contains check, no NLP library required).
  - `record_turn(wav, transcript, question_asked)` updates internal state.
  - `should_reset() -> bool`: True if `elapsed_s > idle_timeout_seconds` config key.
  - `reset()`: clears all state for a new participant.
  - No audio dependencies — pure logic, fully unit-testable with scripted
    transcript strings.
- Wire into `app.py`: replace hard-coded placeholder question with
  `engine.next_question(transcript_history)`.
- **Verify:** unit tests feeding scripted transcripts; assert phase advancement
  on turn thresholds, topic-hook biasing, reset on timeout.
