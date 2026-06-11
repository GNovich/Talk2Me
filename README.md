# Talk2Me

> A gallery installation. A person sits and speaks. The machine listens, learns the
> timbre of their voice, and then speaks back to them — in their *own* voice — a
> sequence of increasingly intimate, increasingly uncomfortable questions. As the
> conversation deepens, the voice clone sharpens from a rough approximation into
> something uncannily exact, so that by the end the participant is hearing themselves
> ask what they least want to answer.

## Hardware target

**Apple Silicon Mac (M-series, 16 GB+ unified memory).** All inference is local via
MLX — no network required at exhibit time. Developed and tested on macOS 14+.

## Stack

| Layer | Library |
|---|---|
| Speech-to-text | `mlx-whisper` (`mlx-community/whisper-large-v3-turbo`) |
| Voice cloning / TTS | `f5-tts-mlx` (zero-shot, no fine-tuning, no separate vocoder) |
| Audio I/O | `sounddevice` + Silero VAD |
| Question engine | Pure Python state machine + optional `mlx-lm` adaptive selection |

## Setup

Requires Python 3.11+ and [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync
```

To include the optional LLM adaptive engine:

```bash
uv sync --extra llm
```

## Run tests

```bash
uv run pytest
```

## Run the installation

```bash
uv run talk2me
```

## Configuration

Edit `config/exhibit.yaml` between gallery sessions without touching source code.
Key knobs: input device, silence threshold, STT model, LLM toggle, voice migration
toggle (neutral→self ramp on/off).

## Architecture

```
talk2me/
  audio/      — Microphone, Speaker, VAD-gated recorder
  stt/        — Whisper-MLX transcriber
  tts/        — F5-TTS-MLX zero-shot voice cloner
  engine/     — Question bank, conversation state machine, optional LLM adapter
  app.py      — Orchestration loop
config/       — Exhibit-tunable YAML parameters
questions/    — Curated question banks (YAML, human-editable)
saved_audio/  — Sample WAVs saved during verification
tests/        — Pytest suite
```

## Privacy

This piece captures participant voice (biometric data). No audio is transmitted over
any network. Captured audio and transcripts are purged at session end by default.
Opt-in logging requires explicit consent (see Feature 12 in ROADMAP).

## Legacy build

The original 2019 build (Tacotron2 / WaveRNN / custom LSTM encoder / Google cloud
STT) is preserved at git tag `legacy-tacotron2-waveRNN-stack` for archival.

## See also

- `Planning/DEV/ROADMAP.md` — full feature roadmap in binding delivery order
- `Planning/DEV/PROMPT_QUEUE.md` — active dev task queue
