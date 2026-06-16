# PROMPT QUEUE — updated 2026-06-16

> Tasks are pulled in binding order from `ROADMAP.md`. Do not start a feature
> before all earlier ones are `COMPLETED`. Max 3 active tasks at a time.

---

## TASK 15 — Voice recording library for simulation and testing  (ROADMAP Feature 15)
**Status:** COMPLETED 2026-06-16

`scripts/generate_fixtures.py` creates three fixture clips in
`tests/fixtures/voices/`: `speaker_f1.wav` (5.3 s real voice — F5-TTS test clip),
`speaker_m1.wav` (2.9 s procedural male), `speaker_nn1.wav` (3.0 s procedural slower).
`SimulatedParticipant` in `tests/simulation.py`: `next_clip()` / `reset()` /
`run_turn()` / `run_session()` API; feeds fixture WAVs through
`Transcriber → ReferenceBuffer → VoiceCloner`; saves `saved_audio/sim_*`.
20 tests: 17 structure (offline), 3 `@pytest.mark.simulation @pytest.mark.model`.
Audible clone-quality regression deferred to exhibit Mac.

---

## TASK 16 — Operator/attendant UI  (ROADMAP Feature 16)
**Status:** COMPLETED 2026-06-16

`talk2me/ui/server.py` (`DashboardServer`, stdlib `http.server`).
`GET /status` → JSON state snapshot.
`POST /control {"action": "reset"|"panic"|"shutdown"}` → SIGUSR1/2/SIGTERM.
`talk2me/ui/dashboard.html` — dark-mode single-page dashboard; polls `/status`
every 2 s; Status, Latency, Conversation history, Controls panels.
Gate: `ui: false` in `exhibit.yaml` (default headless); `ui_port: 8765`.
Wired into `supervisor_loop()`, `main()`, and `run_loop()`.
8 tests pass.

---

## TASK 17 — Latency reduction and output fidelity  (ROADMAP Feature 17)
**Status:** COMPLETED 2026-06-16

(b) Thread overlap: `engine.next_question()` submitted to `ThreadPoolExecutor(1)`
concurrently with `ref_buffer` ops after transcription.
(c) Dual Whisper: `stt.model_fast` config key; calibration turns use fast model.
(d) `tts.nfe_steps: 8` in `exhibit.yaml`; wired to `VoiceCloner(steps=nfe_steps)`.
(a) Streaming TTS deferred — requires forking F5-TTS-MLX upstream.
6 new tests pass. On-hardware latency benchmarks deferred to exhibit Mac.

---

## TASK 18 — AI seed voice + progressive personalisation  (ROADMAP Feature 18)
**Status:** PLAN

### Plan

**Goal.** Ensure the opening voice is always a complete, natural-sounding
neutral AI voice — not an under-referenced rough clone — and smooth the
migration from neutral → participant into a continuous curve rather than
step-wise tier jumps.

**Sub-tasks:**

1. **Generate and commit `assets/neutral_seed.wav`**
   - Run `python scripts/generate_neutral_seed.py` using
     `mlx-community/Kokoro-82M-bf16` to synthesise a short (~5 s) neutral seed
   - Write matching `assets/neutral_seed_text.txt`
   - Commit both files (small WAV, safe to commit)
   - Blocked if `mlx_audio` or Kokoro weights are unavailable

2. **Continuous migration curve**
   - Add `tts.migration_curve: linear` to `exhibit.yaml`
     (choices: `linear`, `ease-in`, `step`)
   - Replace `ReferenceBuffer.migration_alpha(tier)` tier-step logic with a
     voiced-seconds-driven continuous function in `ReferenceBuffer`
   - `step` → current 4-tier lookup (backwards-compatible)
   - `linear` → `voiced_s / target_voiced_s` clamped 0–1
   - `ease-in` → quadratic ramp: `(voiced_s / target_voiced_s)^2`
   - `target_voiced_s` defaults to the tier-3 threshold (20 s)

3. **Verification using simulation fixtures**
   - Use `SimulatedParticipant` (Feature 15) to run 10-turn session
   - Check alpha ramp at each turn for each curve type
   - Audibly compare `linear` vs `step` on exhibit Mac using `speaker_f1` fixture

**Files to touch:**
- `scripts/generate_neutral_seed.py` (new)
- `assets/neutral_seed.wav` + `assets/neutral_seed_text.txt` (new, generated)
- `talk2me/tts/reference_buffer.py` — replace `migration_alpha(tier)` with voiced-seconds curve
- `config/exhibit.yaml` — add `tts.migration_curve`
- `talk2me/app.py` — pass `migration_curve` to `ReferenceBuffer` constructor
- `tests/test_tts.py` — add curve tests

**Blocked on:**
- Kokoro model weights (`mlx-community/Kokoro-82M-bf16`) in local cache
- If not available, generate neutral seed with `soundfile` + numpy (sine + formants)
  as a temporary placeholder; document that it must be replaced with a real Kokoro synthesis
