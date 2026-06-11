# PROMPT QUEUE — updated 2026-06-11

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
**Status:** PENDING

**Plan:**
- `talk2me/tts/voice_cloner.py`: implement `VoiceCloner` class wrapping
  `f5-tts-mlx`. API: `VoiceCloner.synthesize(text, reference_wav,
  reference_text) -> np.ndarray` (float32 at native F5 sample rate, resample
  to 24 kHz / Speaker-native as needed).
- Load/JIT-warm F5-TTS model once at `__init__`; run one throwaway synthesis
  (dummy text + bundled neutral reference clip) so the participant's first turn
  is not slow.
- Cache loaded model instance across turns (do not reload per call).
- Wire `Speaker.play()` from Feature 2 for output.
- Decouple from live capture: verify first against a static reference WAV in
  `saved_audio/` (record a voice sample manually, transcribe it, synthesize a
  question in that voice, save the output WAV for listening review).
- Expose `sample_rate` property; Speaker must use it.
- **Verify:** synthesize "What brought you here today?" in a static reference
  voice, play it back, save to `saved_audio/f4_verify_YYYY-MM-DD.wav`. Log
  synthesis latency (target ≈1.5–2 s for a 20–30 word question on M2/M3).

---

## TASK 5 — End-to-end single-turn loop  (ROADMAP Feature 5)
**Status:** PENDING

**Plan:**
- `talk2me/app.py`: replace stub with a working orchestration loop:
  `record_utterance → Transcriber.transcribe → (hard-coded placeholder
  question) → VoiceCloner.synthesize → Speaker.play`.
- Instrument every stage with `time.perf_counter()` timing; print per-turn
  latency breakdown: `[STT: X.Xs | TTS: X.Xs | play: X.Xs | total: X.Xs]`.
- Handle edge cases: empty transcript → re-prompt; synthesis error → log and
  re-prompt; KeyboardInterrupt → graceful shutdown.
- Load config from `config/exhibit.yaml` (device selection, thresholds, model).
- Pre-warm both Whisper and F5-TTS at startup before first participant turn.
- **Verify:** run the loop end-to-end at least 3 turns; observe synthesized
  voice matches reference; record total round-trip latency (target ≈2–3 s).
  Save a sample output WAV to `saved_audio/` per turn.

---

## TASK 6 — Rolling reference accumulation  (ROADMAP Feature 6)
**Status:** PENDING

**Plan:**
- `talk2me/tts/reference_buffer.py`: `ReferenceBuffer` class. Maintains a
  deque of (wav, transcript, avg_logprob) tuples from participant utterances.
- Quality selection: prefer segments with avg_logprob > threshold (well-
  transcribed), discard clipped (peak > 0.95) or near-silent (RMS < floor).
- Tier thresholds (from `config/exhibit.yaml`): when accumulated voiced duration
  crosses 3 s → 9 s → 20 s, rebuild `reference_wav` + `reference_text` from
  the best segments up to 30 s cap.
- `ReferenceBuffer.best_reference() -> (np.ndarray, str)` returns current best
  reference clip for F5-TTS conditioning.
- Wire into `app.py`: after each turn, push transcript result + wav to buffer;
  call `best_reference()` to get the reference for the *next* turn's synthesis.
- **Verify (logic):** unit tests with synthetic (loud/quiet) segment sequences,
  assert correct tier transitions and segment selection. No audio hardware needed.
- **Verify (audible):** run the full loop for 6+ turns, listen for the voice
  clone sharpening over time; save output WAVs per turn to `saved_audio/`.

> Features 7–14 remain in `ROADMAP.md`.
