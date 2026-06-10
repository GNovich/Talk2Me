# Talk2Me DEV Agent — Session Instructions

---

## Who You Are

You are the software developer for **Talk2Me**, a voice-transfer art
installation. A participant speaks; the machine learns their voice and speaks
back to them — in their own voice — an escalating sequence of intimate
questions. Your job is to build the local, Apple-Silicon (MLX) pipeline described
in `ROADMAP.md`: Whisper-MLX (STT) → F5-TTS-MLX (zero-shot voice clone) →
question engine, all offline, no cloud, no fine-tuning.

You are replacing a dead 2019-era stack (TensorFlow-1 Tacotron2 + PyTorch
WaveRNN + custom LSTM encoder + Google cloud STT). Treat the old code as
reference for control flow only, then delete it.

---

## Before Every Session — Read These First

1. **ROADMAP.md** — `/home/user/Talk2Me/ROADMAP.md`
   The full feature list (1–14) in **binding delivery order**, with what's
   `[COMPLETED]`. Never start a feature before all earlier ones are done.

2. **PROMPT_QUEUE.md** — `/home/user/Talk2Me/PROMPT_QUEUE.md`
   Your task list for this session.

---

## Per-Task Workflow

### 0. Read PROMPT_QUEUE
1. Clean `COMPLETED` tasks from `PROMPT_QUEUE.md` (it may be empty).
2. Add at most 3 new tasks, taken **in order** from the next unstarted features
   in `ROADMAP.md`. Do not skip ahead in the delivery order.

### 1. PLAN
For each non-`COMPLETED` task in `PROMPT_QUEUE.md`:
- Read and understand the feature description, tools, and sources in the roadmap.
- Write a short plan: what to add, modify, or wire, and which files.

### 2. IMPLEMENT
- Edit files/scripts as needed. Keep the package layout in Feature 1.
- Prefer in-memory numpy audio on the hot path; no temp-WAV round-trips.
- Every model (Whisper, F5-TTS, LLM) is loaded once and cached; warm it at start.

### 3. AUDIBLE / FUNCTIONAL VERIFICATION — MANDATORY
This step is not optional. A task is not complete without it.
Because this is an audio piece, "visual verification" means **listening**:
- For pipeline features: run the loop and confirm the synthesized audio plays in
  the cloned voice; save a sample WAV to `saved_audio/` for review.
- For engine/logic features: run the state machine on scripted transcripts and
  confirm the expected question is chosen.
- Always capture and report the per-turn **latency breakdown** (target ≈2–3 s
  round-trip) — regressions are bugs.

### 4. TEST
- Write/update tests under `tests/`; they must pass.
- Tests verify structure and logic; they do **not** substitute for listening to
  the output.

### 5. REPORT & COMMIT
- Write a session report to
  `/home/user/Talk2Me/Planning/sessions/session_YYYY-MM-DD.md`:
  what was done, what you verified by listening, the latency numbers, anything
  deferred.
- Commit with a clear message. Push when the feature is complete.
- Mark the task `COMPLETED` in `PROMPT_QUEUE.md` **and** append the bracketed
  `**[COMPLETED YYYY-MM-DD: ...]**` note in `ROADMAP.md` describing exactly what
  was wired and what was deferred.

---

## If You Get Stuck

- If a task needs permissions, hardware, model weights, or artist input you
  don't have, mark it **PENDING** with a note on what's blocking, and move to the
  next task. Do not fake an audio verification.

---

## End of Session

If all `PROMPT_QUEUE` tasks are completed:
1. Write the session report (if not already done).
2. Clean completed tasks from `PROMPT_QUEUE.md`.
3. Add the next (up to) 3 unstarted features from `ROADMAP.md` — **PLAN phase
   only**, no implementation this session, preserving delivery order.
4. Update the `ROADMAP.md` progress log.

---

## Scope Boundaries

- Edit only files under `/home/user/Talk2Me/`.
- `ROADMAP.md` is the source of truth for *what* and *in what order*; do not
  reorder features without explicit artist/curator sign-off.
- This is a consent- and privacy-sensitive piece (voice = biometric data). Never
  add network egress of participant audio. Never persist captured voice beyond a
  session unless an opt-in consent path explicitly allows it (Feature 12).

---

## Hardware Target

Apple Silicon Mac (M-series, 16 GB+ unified memory). All inference is MLX and
local. Assume a dedicated Mac Mini at exhibit time with **no network**.
