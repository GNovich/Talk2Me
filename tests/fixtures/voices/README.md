# Voice Fixtures

Pre-recorded (or procedurally-generated) voice clips for offline pipeline
simulation and regression testing.

| File          | Type            | Duration | Notes                                   |
|---------------|-----------------|----------|-----------------------------------------|
| speaker_f1    | Real voice      | ~3 s     | Extracted from F5-TTS bundled test clip |
| speaker_m1    | Synthetic       | ~3 s     | Procedural male-range pitch (~120 Hz)   |
| speaker_nn1   | Synthetic       | ~4 s     | Procedural slower tempo (~175 Hz)       |

## Replacing with real LibriSpeech clips

For authentic voice-clone regression testing, replace the synthetic files with
real human voice clips:

1. Download LibriSpeech test-clean from https://www.openslr.org/12/
2. Choose one female speaker (e.g. 1089/), one male (e.g. 1284/), one
   non-native English speaker from a separate dataset (e.g. L2-ARCTIC).
3. Trim each to 5–30 s.  Rename to speaker_f1.wav / speaker_m1.wav / speaker_nn1.wav.
4. Write matching .txt transcript files.
5. Re-run: `uv run python scripts/generate_fixtures.py --replace-only`

Synthetic clips verify pipeline structure; real clips verify audible clone quality.
