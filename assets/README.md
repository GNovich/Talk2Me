# assets/

Static files bundled with the installation.

## Neutral seed voice (Feature 7 — migration mode)

When `engine.migration: true` in `exhibit.yaml`, the installation ramps from a
neutral voice toward the participant's own voice across the session.

Place two files here before the exhibit:

- `neutral_seed.wav` — a short clip (5–15 s) of clean speech in the neutral
  seed voice, at any sample rate (resampled to 24 kHz internally).
- `neutral_seed_text.txt` — the exact transcript of `neutral_seed.wav`.

**Recommended source:** generate the seed once with the Kokoro TTS model
(`mlx-community/Kokoro-82M-bf16`) using a stable preset voice.  That keeps the
seed license-clear and consistent across exhibits.

If either file is missing, migration silently falls back to `alpha=1.0`
(participant voice from turn 1) with a startup warning.
