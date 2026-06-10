# PROMPT QUEUE — 2026-06-10

> Tasks are pulled in binding order from `ROADMAP.md`. Do not start a feature
> before all earlier ones are `COMPLETED`. Max 3 active tasks at a time.

---

## TASK 1 — Repository teardown & MLX scaffold  (ROADMAP Feature 1)
**Status:** PENDING

Strip the legacy Tacotron2/WaveRNN/encoder stack and stand up the clean MLX
package.

**Plan:**
- Tag/branch the current legacy state for archival, then `git rm` `encoder/`,
  `synthesizer/`, `vocoder/`, `toolbox/`, all `*_train.py` / `*_preprocess.py`,
  `demo_cli.py`, `demo_toolbox.py`, `demo_toolbox_collab.ipynb`. Preserve
  `utils/logmmse.py` and `saved_audio/`.
- Replace `requirements.txt` with `pyproject.toml` (Python 3.11+, MLX deps:
  `mlx-whisper`, `f5-tts-mlx`, `sounddevice`, `numpy`; add `mlx-lm` later).
- Create package layout: `talk2me/{audio,stt,tts,engine}/`, `talk2me/app.py`,
  `config/`, `questions/`, `tests/`.
- Add a `SessionStart` hook (`uv sync` + run tests) for web/CI sessions.
- Update README with concept + macOS/M-series hardware target.
- **Verify:** `uv sync` succeeds; empty test suite runs; package imports.

---

## TASK 2 — Audio I/O + VAD  (ROADMAP Feature 2)
**Status:** PENDING

Real-time capture/playback layer with VAD-gated utterance recording.

**Plan:**
- `talk2me/audio/io.py`: `Microphone` (sounddevice, 16 kHz mono float32) and
  `Speaker` (blocking + non-blocking; ~1 s silence pad to dodge tail-cut bug).
- Device selection by config (enumerate, fall back to default — do NOT hard-code
  "Built-in Microphone").
- `talk2me/audio/vad.py`: Silero-VAD gate (fallback webrtcvad). Implement
  `record_utterance(max_seconds, silence_ms) -> np.ndarray`: onset detect,
  capture until trailing-silence threshold.
- Ambient-noise calibration pass; expose gain + silence threshold in `config/`.
- **Verify:** record a spoken phrase, play it back, confirm clean audio; assert
  utterance boundaries on a silence-padded sample.

---

## TASK 3 — Local STT with Whisper-MLX  (ROADMAP Feature 3)
**Status:** PENDING

Replace cloud STT with offline `mlx-whisper`.

**Plan:**
- `talk2me/stt/whisper.py`: `Transcriber.transcribe(wav) -> str`, default
  `mlx-community/whisper-large-v3-turbo`, `--stt-model` override.
- Return no-speech / avg-logprob scores (used later for VAD confirmation and
  reference-segment quality ranking in Feature 6).
- Handle empty/inaudible input → empty string.
- Warm the model at startup.
- **Verify:** transcribe a recorded utterance from TASK 2; report transcription +
  latency (target ≈0.3–0.5 s for 10 s audio on M-series).

---

> Features 4–14 remain in `ROADMAP.md`. After TASKS 1–3 complete, pull the next
> three (F5-TTS-MLX cloning, end-to-end single-turn loop, rolling reference
> accumulation) into this queue in order.
