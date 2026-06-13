# Talk2Me — Voice-Transfer Art Exhibit — ROADMAP

> **Concept.** A gallery installation. A person sits and speaks. The machine
> listens, learns the timbre of their voice, and then speaks back to them — in
> their *own* voice — a sequence of increasingly intimate, increasingly
> uncomfortable questions. As the conversation deepens, the voice clone sharpens
> from a rough approximation into something uncannily exact, so that by the end
> the participant is hearing themselves ask what they least want to answer.
>
> **The redesign.** The original (2019-era) build chained a custom LSTM speaker
> encoder, a TensorFlow-1 Tacotron2 synthesizer, a PyTorch WaveRNN vocoder, and
> Google's cloud STT. It required hand-downloaded pretrained checkpoints, ran at
> 5–15 s latency, and depended on the network. This roadmap replaces that entire
> stack with a single, fully-local, Apple-Silicon-native pipeline built on **MLX**:
> Whisper-MLX for transcription, **F5-TTS-MLX** for zero-shot voice cloning (no
> separate vocoder, no fine-tuning), and an optional `mlx-lm` model for adaptive
> question selection.

> **Feature delivery order is binding.** Each numbered feature assumes the ones
> before it are in place. Do not start a later feature before earlier ones are
> `[COMPLETED]`. When a feature is finished, append a bracketed
> `**[COMPLETED YYYY-MM-DD: ...]**` note in place (as in the reference roadmap)
> describing exactly what was wired, where, and what was deferred.

---

## Phase 0 — Foundation

### 1. Repository teardown & MLX scaffold **[COMPLETED 2026-06-11: Legacy stack (encoder/, synthesizer/, vocoder/, toolbox/, all train/preprocess scripts, demo files, conda requirements.txt) deleted after tagging `legacy-tacotron2-waveRNN-stack`. utils/logmmse.py relocated to talk2me/audio/logmmse.py. New layout: talk2me/{audio,stt,tts,engine}/, config/exhibit.yaml, questions/, saved_audio/, tests/. pyproject.toml with Python 3.11+, mlx-whisper, f5-tts-mlx, sounddevice, numpy, pyyaml, onnxruntime deps. README rewritten. 5 smoke tests pass.]**
**Description.** Strip the legacy three-model pipeline and stand up a clean,
single-package project. Remove `encoder/`, `synthesizer/`, `vocoder/`,
`toolbox/`, the `*_train.py` / `*_preprocess.py` scripts, `demo_*.py`, and the
Colab notebook — these belong to the old Tacotron/WaveRNN stack and carry TF1 +
PyTorch + numba=0.48 pinning we are abandoning. Preserve `utils/logmmse.py`
(denoise may still be useful) and `saved_audio/`. Replace `requirements.txt`
(conda spec) with a `pyproject.toml` targeting Python 3.11+ and MLX. Create the
new package layout: `talk2me/` (source), `talk2me/audio/` (I/O + VAD),
`talk2me/stt/`, `talk2me/tts/`, `talk2me/engine/` (question logic),
`talk2me/app.py` (orchestration loop), `config/` (YAML for exhibit params),
`questions/` (the curated question banks), and `tests/`. Add a `SessionStart`
hook so web/CI sessions can `uv sync` and run the test suite. Document the macOS
hardware target (M-series, 16 GB+ unified memory) in the README.
**Tools.** `uv` (or `pip`) + `pyproject.toml`; `git rm` for teardown; ruff for lint.
**Sources.** Existing `Talk2Me.py` (salvage the `Recorder`/`VoiceProducer` control
flow as reference for the new loop, then delete).
**Extra.** Keep one frozen tag/commit of the legacy build before deletion so the
original art piece remains reproducible for archival.

### 2. Audio I/O + Voice Activity Detection **[COMPLETED 2026-06-11: Microphone (sounddevice 16 kHz mono float32, device by name/index/None) + Speaker (~1 s tail pad, thread-locked) in talk2me/audio/io.py. Silero VAD ONNX gate (auto-downloads to ~/.cache/talk2me/) with _EnergyGate fallback in talk2me/audio/vad.py. record_utterance(mic, max_seconds, silence_ms) -> np.ndarray API. build_gate() for pre-warming. Config knobs in config/exhibit.yaml. 7 unit tests pass. On-hardware audio verification deferred to exhibit Mac.]**
**Description.** Build the real-time capture and playback layer that everything
else feeds on. A `Microphone` wrapper around `sounddevice` for low-latency input
at 16 kHz mono; a `Speaker` wrapper for blocking + non-blocking playback (carry
forward the original's trick of padding output with ~1 s of silence to dodge the
sounddevice tail-cut bug). A VAD-gated recorder that segments speech into
utterances: detect speech onset, capture until a configurable trailing-silence
threshold, return a float32 waveform + duration. Expose a clean
`record_utterance(max_seconds, silence_ms) -> np.ndarray` API. Include device
selection (don't hard-code `"Built-in Microphone"` like the original —
enumerate and pick by config, fall back to default). All audio stays in-memory
as numpy; no temp WAV round-trips on the hot path.
**Tools.** `sounddevice`, `numpy`; `silero-vad` (ONNX/torch) **or** `webrtcvad`
for the gate — prefer Silero for robustness in a noisy gallery, fall back to
webrtcvad if latency demands.
**Sources.** Silero VAD https://github.com/snakers4/silero-vad ; original
`Recorder` class in `Talk2Me.py` for the listen/timeout pattern.
**Extra.** Add an ambient-noise calibration pass (gallery rooms are loud); make
the silence threshold and input gain exhibit-tunable via `config/`.

---

## Phase 1 — Core Local Pipeline

### 3. Local speech-to-text with Whisper-MLX **[COMPLETED 2026-06-11: Transcriber.transcribe(wav) -> TranscriptResult(text, no_speech_prob, avg_logprob, latency_s) in talk2me/stt/whisper.py. Default model mlx-community/whisper-large-v3-turbo; model loaded + JIT-warmed on first call; empty/None input short-circuits. is_speech() uses no_speech_prob threshold. avg_logprob exposed for Feature 6 quality ranking. 6 unit tests + 1 @model integration test. On-hardware latency verification deferred.]**
**Description.** Replace Google cloud STT (`recognizer.recognize_google`) with a
fully-offline, Apple-Silicon-accelerated transcriber. Wrap `mlx-whisper` behind
a `Transcriber.transcribe(wav: np.ndarray) -> str` interface. Default model
`mlx-community/whisper-large-v3-turbo` (≈0.3–0.5 s for a 10 s utterance on
M2/M3); expose `--stt-model` to downshift to `whisper-medium`/`tiny` on
lower-end hardware. Handle empty/inaudible input gracefully (return empty string,
let the engine re-prompt). The transcript feeds two consumers: the question
engine (for topic extraction) and the logging layer.
**Tools.** `mlx-whisper` (`pip install mlx-whisper`); `numpy`.
**Sources.** https://github.com/ml-explore/mlx-examples/tree/main/whisper ;
model card `mlx-community/whisper-large-v3-turbo`. **Reuse:** the in-house
`PdfReader` project already runs a Whisper aligner under MLX in `align_server.py`
— lift its model-load / device-handling code (it solved Apple-Silicon MPS
quirks) rather than re-deriving it.
**Extra.** Whisper also returns no-speech probability — use it as a secondary VAD
confirmation to reject false triggers before invoking TTS.

### 4. Zero-shot voice cloning with F5-TTS-MLX **[COMPLETED 2026-06-12: VoiceCloner.synthesize(text, reference_wav, reference_text, reference_sample_rate) -> np.ndarray in talk2me/tts/voice_cloner.py. F5TTS.from_pretrained loaded once + JIT-warmed via bundled test clip. 16 kHz mic audio resampled to 24 kHz via scipy.signal.resample_poly. RMS-normalised to F5-TTS training level. Duration estimated from char-count ratio. Wired into app.py. On-hardware audible verification and latency measurement deferred to exhibit Mac.]**
**Description.** The heart of the piece. Replace the entire Tacotron2 +
WaveRNN + custom-encoder chain with **F5-TTS-MLX**, a flow-matching TTS that
clones a voice zero-shot from a few seconds of reference audio and synthesizes
in one model (no separate vocoder, no speaker-embedding model, no fine-tuning).
Build `VoiceCloner.synthesize(text: str, reference_wav: np.ndarray,
reference_text: str) -> np.ndarray`. The reference is the participant's own
recorded speech; `reference_text` is its Whisper transcript (F5-TTS conditions on
both). Target ≈1.5–2 s synthesis for a 20–30 word question on M2/M3. Stream the
result straight to the `Speaker` from Feature 2. Verify against a static
reference WAV first (decouple from the live-capture loop) before wiring it to
real participants.
**Tools.** `f5-tts-mlx` (`pip install f5-tts-mlx`); MLX.
**Sources.** https://github.com/lucasnewman/f5-tts-mlx ; upstream
https://github.com/SWivid/F5-TTS ; paper https://arxiv.org/abs/2410.06885 .
**Extra.** First call JIT-warms the model — preload and run one throwaway
synthesis at startup so the participant's first question isn't slow. Cache the
loaded model across turns.
**Important — in-house reuse, corrected:** the `PdfReader` project does **not**
have a voice cloner. Its TTS is **`mlx-community/Kokoro-82M-bf16`**, a fixed
preset-voice model that *cannot* clone a participant's voice — so F5-TTS-MLX here
is net-new. What IS reusable from PdfReader: (a) its `mlx_audio.server` pattern
of running TTS as a local OpenAI-compatible HTTP service (consider the same
architecture for Talk2Me — decouple synthesis from the capture loop); (b)
Kokoro itself as a *non-cloned* narrator for the neutral seed voice in Feature 7
and any system/operator prompts. Do not assume an F5-TTS loader exists there.

### 5. End-to-end single-turn loop **[COMPLETED 2026-06-12: talk2me/app.py implements record_utterance → Transcriber.transcribe → ReferenceBuffer.push → VoiceCloner.synthesize → Speaker.play with time.perf_counter() at each stage. Pre-warms Whisper and F5-TTS at startup. Latency breakdown printed per-turn: [STT|TTS|play|total]. Per-turn output saved to saved_audio/. Handles empty transcript, no-reference, synthesis errors, KeyboardInterrupt. Hard-coded placeholder question; Feature 8/9 will replace. On-hardware latency verification deferred.]**
**Description.** Stitch Features 2–4 into one verified round-trip:
`record_utterance → transcribe → (fixed question) → synthesize in cloned voice →
play`. Hard-code a single placeholder question to prove the pipeline before the
engine exists. Instrument every stage with timing so the latency budget
(target ≈2–3 s total round-trip) is measurable from day one. This is the
"walking skeleton" — once it speaks one question back in the participant's voice,
the rest of the roadmap is incremental.
**Tools.** Python `time.perf_counter`; structured logging.
**Sources.** Original `Talk2Me.py` main loop (the record→say cadence).
**Extra.** Emit a per-turn latency report to the session log so regressions
across later features are caught immediately.

---

## Phase 2 — The Voice-Transfer Mechanic

### 6. Rolling reference accumulation ("the voice that sharpens") **[COMPLETED 2026-06-12: ReferenceBuffer in talk2me/tts/reference_buffer.py. push(wav, TranscriptResult) quality-gates segments (RMS > 0.005, peak < 0.95, avg_logprob > -1.5, non-empty text). best_reference() returns quality-ranked segments up to 30 s cap in capture order. Tier 0→1 at ≥3 s, 1→2 at ≥9 s, 2→3 at ≥20 s. Result cached; invalidated on push. reset() for session teardown. Wired into app.py. 14 unit tests pass. Audible sharpening verification deferred to exhibit Mac.]**
**Description.** This is the artistic core: the clone must *improve over the
conversation*. Maintain a rolling buffer of the participant's captured utterances
(audio + transcripts). On each turn, append the new utterance; when total
reference duration crosses tiers (≈3 s → 9 s → 20 s+), rebuild the reference clip
fed to F5-TTS (Feature 4) from the accumulated, highest-quality segments. Early
turns: a rough, slightly-off version of their voice. Later turns: uncannily
exact. Implement quality selection (prefer clean, voiced, well-transcribed
segments; drop clipped or near-silent ones) and a max-reference cap (F5-TTS
degrades with overly long references — keep the best ~15–30 s). Replaces the
original's naive embedding-averaging (`update_embbeding`) which mean-pooled LSTM
embeddings; here we curate raw reference audio instead.
**Tools.** `numpy`; the Whisper no-speech / avg-logprob scores from Feature 3 for
segment quality ranking.
**Sources.** Original `VoiceProducer.update_embbeding` / `wav_stack` logic as the
conceptual ancestor (then supersede it).
**Extra.** Make the sharpening *audible by design* — optionally start the very
first synthesis from a slightly neutral/blended reference so the migration toward
the participant's exact voice is perceptible. Expose tier thresholds in `config/`.

### 7. Neutral→self voice migration (the unsettling ramp) **[COMPLETED 2026-06-13: `migration_alpha` param added to VoiceCloner.synthesize(); when alpha<1.0 and assets/neutral_seed.wav is present, blends [neutral|participant] audio proportionally. ReferenceBuffer.migration_alpha(tier) maps tier 0→0.2, 1→0.5, 2→0.8, 3→1.0 (configurable via engine.migration_alphas in exhibit.yaml). Toggle engine.migration: true|false. assets/README.md explains how to generate the neutral seed with Kokoro. Neutral seed files not bundled — operator must place before exhibit. Audible ramp verification deferred to exhibit Mac.]**
**Description.** Optional but central to the intended discomfort: rather than
cloning the participant from turn 1, begin with a near-neutral voice and
interpolate toward their exact voice as reference accumulates, so the participant
gradually realizes the questioner *is* them. Implement as a blend parameter on
the reference (or on synthesis conditioning) that ramps from neutral→self across
the calibration phase. Must be a toggle in `config/` (`migration: on|off`) since
the alternative — full self-voice from turn 1 — is also artistically valid and
should be A/B-able in the gallery.
**Tools.** F5-TTS conditioning; a neutral seed reference clip bundled in
`assets/` — this can be generated once with PdfReader's Kokoro voice (a clean,
neutral preset) so the seed is consistent and license-clear.
**Sources.** Feature 4/6 internals; `mlx-community/Kokoro-82M-bf16` for the seed.
**Extra.** This is a curatorial decision as much as a technical one — surface the
toggle clearly and document both modes' intended effect for the exhibit operator.

---

## Phase 3 — The Question Engine

### 8. Curated question bank (three escalating phases) **[COMPLETED 2026-06-13: questions/calibration.yaml (6 Q), questions/personal.yaml (10 Q), questions/confrontational.yaml (12 Q) — all entries have id, text, topic_hooks. QuestionBank.load(dir) validates schema, raises ValueError on bad entries. select(phase, used_ids, topic_hints) returns (id, text) biased toward topic_hook overlaps, falls back to deterministic first-unused, cycles on exhaustion. reload() for hot-reload without restart. 15 unit tests pass. Question content is first-draft; artist to tune between sessions.]**
**Description.** Author and structure the pre-written questions that drive the
piece. Three tiers as data (YAML/JSON in `questions/`), each entry tagged with
phase, intensity, and topic hooks:
- **Phase 1 — Calibration (turns 1–2):** comfortable, observational; their real
  job is to harvest clean reference audio without alarming the participant
  ("What brought you here today?", "Describe the room you're sitting in.").
- **Phase 2 — Personal (turns 3–5):** specific, lightly intrusive, ideally
  referencing what was just said.
- **Phase 3 — Confrontational (turns 6+):** questions that sit in the body
  ("What are you afraid people see when they look at you?", "What's the last lie
  you told that you're still carrying?").
Build a `QuestionBank` loader with validation (every entry has phase + text) and
a deterministic non-repeating selector.
**Tools.** YAML (`pyyaml`); a small schema validator.
**Sources.** Curator/artist-supplied drafts; generate a first draft for review if
none provided.
**Extra.** Keep the bank human-editable and hot-reloadable so the artist can tune
tone between gallery sessions without touching code. Include a content/consent
note: Phase 3 is intense by design — pair with the safety features in Phase 4.

### 9. Conversation state machine **[COMPLETED 2026-06-13: ConversationEngine in talk2me/engine/state_machine.py. Tracks turn, phase (1-3), transcript_history, used_question_ids. Phase advances: P1 for turns ≤calibration_turns, P2 for turns ≤cal+personal, P3 thereafter. next_question() calls QuestionBank.select() with topic hints from last 3 transcripts (words ≥4 chars). record_turn(transcript, question) updates state. should_reset() checks monotonic idle timeout. reset() clears all state. Wired into app.py replacing placeholder question; idle-timeout session reset also wired. 6 unit tests pass. Speaker now created once at 24 kHz (was re-created each turn). Latency log includes alpha= and phase=.]**
**Description.** The director that decides *what to ask next*. Tracks turn count,
current phase, elapsed time, topics surfaced (naive keyword/topic extraction over
the Whisper transcript history), and which questions have fired. Advances phases
on turn/time thresholds, selects the next question from the bank (Feature 8)
biased toward unused entries whose topic hooks match recent transcript content,
and signals session reset on long silence or an exit cue. Replaces the original's
`TODO: conversational class` stub entirely. Pure, testable logic with no audio
dependencies — feed it transcripts, assert the chosen question.
**Tools.** Plain Python; simple keyword/topic matcher (optionally `rapidfuzz`).
**Sources.** Original `Talk2Me.py:173` `# TODO conversational class` marker.
**Extra.** Make phase thresholds and escalation pacing config-driven so the arc
can be lengthened/shortened per exhibit constraints.

### 10. Optional LLM-adaptive question selection (`mlx-lm`)
**Description.** Upgrade the engine from "select from bank" to "select *and
lightly adapt* from bank" so questions feel responsive rather than scripted. A
local `mlx-lm` model (Llama-3.2-3B-Instruct or Gemma-2-2B, 4-bit) reads the
transcript history and the candidate question, and rewrites it to reference what
the participant actually disclosed — strictly constrained to stay within the
curated tone and never invent new lines of questioning outside the artist's
intent. Must be a hard toggle (`engine.llm: off` falls back to Feature 9's pure
selector) because it adds latency and a safety surface. Heavily prompt-guard
against drift; the bank remains the source of truth, the LLM only personalizes.
**Tools.** `mlx-lm` (`pip install mlx-lm`); a quantized instruct model from
`mlx-community`.
**Sources.** https://github.com/ml-explore/mlx-examples/tree/main/llms ;
`mlx-community/Llama-3.2-3B-Instruct-4bit`.
**Extra.** Budget the LLM call against the latency target (Feature 5) — run it
*during* TTS warm-up or while the participant is still speaking. Log both the
original bank line and the adapted line for curatorial review.

---

## Phase 4 — Exhibit Hardening

### 11. Kiosk runtime & session lifecycle
**Description.** Turn the script into an unattended installation. Fullscreen /
headless kiosk mode with no visible developer UI (the participant should see at
most a minimal ambient visual, or nothing). Robust session lifecycle: greet →
run the arc → reset to idle after a silence/idle timeout, clearing the rolling
reference buffer and conversation state so the next participant starts fresh.
Auto-recover from any per-turn exception (mic glitch, empty transcript) without
crashing the installation. Launch-on-boot for a dedicated Mac Mini.
**Tools.** `launchd` plist for boot; a supervisor loop with try/except per turn.
**Sources.** Feature 5 loop; original main-loop structure.
**Extra.** A physical/hardware reset affordance (keyboard cue or footswitch) for
the gallery attendant.

### 12. Consent, safety & privacy layer
**Description.** Non-negotiable for a piece that clones voices and asks intimate
questions. Explicit on-screen/printed consent before a session; a visible/audible
way to stop at any time; automatic purge of all captured audio and transcripts at
session end **by default** (the participant's voice never persists unless they
opt in). If logging is enabled for post-exhibit analysis (Feature 13), it is
opt-in, anonymized, and consented. No network egress of any audio — assert the
pipeline is fully local. Document data handling for the gallery.
**Tools.** Local-only file handling; explicit teardown/secure-delete of buffers.
**Sources.** Privacy/consent norms for biometric (voice) art works.
**Extra.** A "panic" key that immediately stops playback, wipes the session, and
returns to idle — within reach of the attendant at all times.

### 13. Telemetry, logging & latency dashboard
**Description.** Opt-in, anonymized instrumentation for tuning the piece between
sessions and for any post-exhibit artist analysis: per-turn latency breakdown
(STT / engine / TTS / playback), phase progression timing, which questions landed,
and (only with consent) anonymized transcripts. A lightweight local dashboard or
log digest the artist/operator can read. Builds on the timing hooks from
Feature 5.
**Tools.** Structured logging (JSON lines); optional small local web/TUI digest.
**Sources.** Feature 5 latency instrumentation.
**Extra.** Surface a live "is the pipeline healthy / within latency budget"
indicator for the attendant, distinct from anything the participant sees.

### 14. Packaging & deployment
**Description.** Make the installation reproducible and deployable to gallery
hardware without a developer present. A pinned, locked dependency set; a one-
command setup that pre-downloads all MLX model weights (Whisper, F5-TTS, optional
LLM) into a local cache so the install runs with **zero network at exhibit time**;
a smoke-test that verifies mic, speaker, and all three models load. Optionally a
signed `.app`/DMG for drag-install (mirrors the in-house reader project's Tauri/
DMG packaging track — defer if time-constrained; a locked `uv`/conda env is the
minimum viable deploy).
**Tools.** `uv` lockfile (or conda-lock); a model-prefetch script; optional
Tauri + `codesign`/`notarytool` for the signed bundle.
**Sources.** Reference project's Rust-packaging task (Tauri sidecar + DMG +
Homebrew formula) as the model for the optional signed-bundle path.
**Extra.** Ship a written runbook for the gallery attendant: power-on, daily
smoke-test, panic/reset, shutdown.

---

## Progress Log
- 2026-06-10 — Roadmap authored. Legacy stack (Tacotron2/WaveRNN/encoder, TF1) to
  be replaced by MLX pipeline (Whisper-MLX + F5-TTS-MLX + optional mlx-lm).
  Features 1–14 defined in binding delivery order across Phases 0–4. No
  implementation started.
- 2026-06-10 — Reviewed in-house `GNovich/PdfReader` (could not read source —
  out of session scope; confirmed via commit metadata). Its TTS is
  `mlx-community/Kokoro-82M-bf16` (preset voices, **no cloning**), served via
  `mlx_audio.server`, with a Whisper MLX aligner in `align_server.py`. Corrected
  Features 3/4/7: F5-TTS-MLX cloning is net-new (Kokoro can't clone); reusable
  from PdfReader = the Whisper-MLX load code, the `mlx_audio.server` local-HTTP
  TTS pattern, and Kokoro as the neutral seed/narrator voice.
- 2026-06-11 — Automated session. Features 1, 2, 3 implemented and committed.
  Legacy stack deleted (archival tag: `legacy-tacotron2-waveRNN-stack`). New MLX
  package scaffold in place; 18 unit tests pass. On-hardware audio/latency
  verification deferred (session ran without exhibit Mac).
- 2026-06-12 — Automated session. Features 4, 5, 6 implemented and committed.
  VoiceCloner (F5-TTS-MLX), end-to-end app loop, and ReferenceBuffer in place.
  f5-tts-mlx installed (0.2.6); scipy added for 16→24 kHz polyphase resampling.
  36 unit tests pass. All audible verification and latency measurement deferred to
  exhibit Mac. Next: Features 7 (neutral→self migration), 8 (question bank),
  9 (state machine).
- 2026-06-13 — Automated session. Features 7, 8, 9 implemented and committed.
  Migration blending in VoiceCloner; ReferenceBuffer.migration_alpha(tier);
  three-phase question YAML bank (6+10+12 questions); QuestionBank loader with
  hot-reload; ConversationEngine state machine with topic-biased selection and
  idle-timeout reset. app.py fully wired. 66 unit tests pass. All audible
  verification deferred to exhibit Mac. Next: Features 10–12.
