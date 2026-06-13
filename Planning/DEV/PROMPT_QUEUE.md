# PROMPT QUEUE — updated 2026-06-13

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
**Status:** COMPLETED 2026-06-13

`migration_alpha` blend parameter in `VoiceCloner.synthesize()`. Proportional
neutral+participant audio concatenation. `ReferenceBuffer.migration_alpha(tier)`
maps tier→alpha. Toggle in `exhibit.yaml`. Neutral seed files not bundled —
operator generates with Kokoro and places in `assets/`. Audible ramp deferred.

---

## TASK 8 — Curated question bank  (ROADMAP Feature 8)
**Status:** COMPLETED 2026-06-13

`questions/calibration.yaml` (6Q), `questions/personal.yaml` (10Q),
`questions/confrontational.yaml` (12Q). `QuestionBank` with schema validation,
non-repeating selection, topic-hint biasing, hot-reload. 15 tests pass.

---

## TASK 9 — Conversation state machine  (ROADMAP Feature 9)
**Status:** COMPLETED 2026-06-13

`ConversationEngine` in `talk2me/engine/state_machine.py`. Phase tracking,
topic-biased question selection, idle-timeout reset. Wired into `app.py`;
placeholder question replaced. 6 unit tests pass.

---

## TASK 10 — Optional LLM-adaptive question selection  (ROADMAP Feature 10)
**Status:** PENDING — PLAN ONLY

**Plan:**
- Gate the entire feature behind `engine.llm: false` in `exhibit.yaml`;
  when false (default), skip all LLM code and use Feature 9's pure selector.
- New file `talk2me/engine/llm_adapter.py`: `LLMAdapter` class.
  - `__init__(model_id, max_new_tokens, device)`: loads `mlx_lm` model +
    tokenizer once; keep as default `mlx-community/Llama-3.2-3B-Instruct-4bit`.
  - `personalize(candidate_question, transcript_history, phase) -> str`:
    - Builds a tightly constrained system prompt: "You are editing a single
      question. Preserve its exact meaning, tone, and phase intensity. Do NOT
      invent new topics. Return only the question text, nothing else."
    - Interpolates `candidate_question` and last 3 transcript turns as context.
    - Caps generation at ~50 tokens; strips any preamble from the output.
    - Falls back to `candidate_question` unchanged if the model returns empty
      or the output is >3× longer than the input (drift guard).
  - `warm()`: run one throwaway call at startup to JIT-compile.
- `ConversationEngine.next_question()` checks `use_llm` flag (passed at init);
  if true, wraps the bank selection output with `LLMAdapter.personalize()`.
- Log both the original bank text and the adapted text per turn to
  `saved_audio/` or a structured log file for curatorial review.
- Budget: LLM call must complete in ≤1 s on M2/M3 (3B 4-bit is ~0.3 s);
  run concurrently with TTS warm-up if possible.
- **Verify:** unit test that with `engine.llm: false` the adapter is never
  instantiated; scripted test that the fall-back triggers correctly on long
  output; on-hardware latency test that LLM call stays within budget.

---

## TASK 11 — Kiosk runtime & session lifecycle  (ROADMAP Feature 11)
**Status:** PENDING — PLAN ONLY

**Plan:**
- Headless mode: no GUI; suppress Python tracebacks to stdout in kiosk mode;
  redirect logs to a file (`logs/session_YYYY-MM-DD.log`).
- Supervisor loop: the `run_loop()` already has try/except per turn; add an
  outer restart loop so an uncaught fatal error restarts the pipeline within
  5 s (max 3 restarts before standby).
- Idle/reset behaviour: Feature 9's `should_reset()` + `reset()` already handle
  per-session clearing; Feature 11 adds a longer *standby* state (>5 min total
  silence) where the installation plays a neutral ambient tone to signal
  readiness.
- Launch-on-boot: write a `launchd` plist at
  `scripts/com.talk2me.exhibit.plist` that sets `RunAtLoad=true` and
  `KeepAlive=true`; include an `install_launchd.sh` helper.
- Attendant reset key: listen for a configurable key chord (default `Ctrl+R`)
  at the outer loop; immediately wipe session and return to idle.
- **Verify:** confirm outer restart loop recovers from an injected
  `RuntimeError`; confirm launchd plist is valid XML (`plutil -lint`).

---

## TASK 12 — Consent, safety & privacy layer  (ROADMAP Feature 12)
**Status:** PENDING — PLAN ONLY

**Plan:**
- `talk2me/app.py`: add a `consent_gate()` function called before each session
  start. In headless/kiosk mode: display text on stdout ("Press ENTER to begin
  and confirm you consent to voice recording for this session. Press Q to quit.")
  and wait for input. In attended mode: the gallery attendant confirms.
- Panic key (`Ctrl+C` already handled; add `Ctrl+P` or configurable key):
  immediately stops playback (`Speaker.stop()`), wipes `ReferenceBuffer`,
  resets `ConversationEngine`, and prints a clear attendant message.
- Session-end purge: add `_purge_session(ref_buffer, engine)` that calls
  `ref_buffer.reset()` and `engine.reset()`. Called on: normal end, panic,
  idle timeout reset, and outer-loop restart.
- Opt-in logging: add `privacy.save_transcripts: false` to `exhibit.yaml`;
  when false (default), transcripts are never written to disk. When true,
  anonymized (no timestamps, no names) transcripts are appended to
  `logs/transcripts_YYYY-MM-DD.log`.
- Assert network egress: add a startup check that raises `RuntimeError` if any
  of the model load paths resolve to a non-local URL (i.e., require download
  at runtime rather than from cache).
- Document data handling in `PRIVACY.md` at project root for the gallery.
- **Verify:** unit test that `_purge_session` calls reset on both objects;
  confirm that with `privacy.save_transcripts: false`, no transcript files are
  created after a simulated session.
