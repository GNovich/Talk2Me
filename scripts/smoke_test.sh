#!/usr/bin/env bash
# Talk2Me smoke test — run before each exhibit day.
# Verifies: mic device enumeration, speaker tone, Whisper model load, F5-TTS model load.
# Exit code 0 = all checks passed.

set -euo pipefail

PASS=0
FAIL=0

pass() { echo "[PASS] $1"; ((PASS++)) || true; }
fail() { echo "[FAIL] $1"; ((FAIL++)) || true; }

echo "================================================================"
echo " Talk2Me smoke test — $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo

# ── 1. Python environment ─────────────────────────────────────────────────────
echo "── 1. Python environment"
if python3 -c "import talk2me" 2>/dev/null; then
    pass "talk2me package importable"
else
    fail "talk2me package not importable (run: uv sync)"
fi

# ── 2. Audio devices ──────────────────────────────────────────────────────────
echo
echo "── 2. Audio devices"
if python3 -c "import sounddevice as sd; devs = sd.query_devices(); assert len(devs) > 0" 2>/dev/null; then
    DEV_COUNT=$(python3 -c "import sounddevice as sd; print(len(sd.query_devices()))")
    pass "Audio devices enumerated ($DEV_COUNT found)"
else
    fail "sounddevice could not enumerate audio devices"
fi

# ── 3. Speaker output (440 Hz tone, 0.5 s) ───────────────────────────────────
echo
echo "── 3. Speaker output"
if python3 - <<'PYEOF' 2>/dev/null
import numpy as np, sounddevice as sd
sr = 24000
t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False, dtype=np.float32)
tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
sd.play(tone, samplerate=sr)
sd.wait()
PYEOF
then
    pass "Speaker: 440 Hz tone played (0.5 s)"
else
    fail "Speaker: tone playback failed"
fi

# ── 4. Whisper model load ─────────────────────────────────────────────────────
echo
echo "── 4. Whisper STT model"
if python3 - <<'PYEOF' 2>/dev/null
from talk2me.stt.whisper import Transcriber
import numpy as np
t = Transcriber()
r = t.transcribe(np.zeros(16000, dtype=np.float32))
assert r is not None
PYEOF
then
    pass "Whisper: model loaded and transcribed silence"
else
    fail "Whisper: model load or transcribe failed (weights cached?)"
fi

# ── 5. F5-TTS model load ──────────────────────────────────────────────────────
echo
echo "── 5. F5-TTS voice cloner"
if python3 - <<'PYEOF' 2>/dev/null
from talk2me.tts.voice_cloner import VoiceCloner
c = VoiceCloner()
c.warm()
PYEOF
then
    pass "F5-TTS: model loaded and warm() completed"
else
    fail "F5-TTS: model load or warm() failed (weights cached?)"
fi

# ── 6. launchd plist validity ─────────────────────────────────────────────────
echo
echo "── 6. launchd plist"
PLIST="$(dirname "$0")/com.talk2me.exhibit.plist"
if [ -f "$PLIST" ]; then
    if plutil -lint "$PLIST" > /dev/null 2>&1; then
        pass "launchd plist valid XML"
    else
        fail "launchd plist is invalid XML"
    fi
else
    fail "launchd plist not found: $PLIST"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo
echo "================================================================"
echo " Results: ${PASS} passed, ${FAIL} failed"
echo "================================================================"
echo

if [ "$FAIL" -gt 0 ]; then
    echo "ACTION REQUIRED: fix the failing checks before opening the exhibit."
    exit 1
fi

echo "All checks passed — installation is ready."
exit 0
