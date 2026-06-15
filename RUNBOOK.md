# Talk2Me — Attendant Runbook

> This document is for the gallery attendant.  No developer knowledge is needed.
> All commands run in the Terminal app.

---

## Daily startup

1. **Power on** the exhibit Mac Mini.  Wait ~30 seconds for the desktop to appear.
2. **Open Terminal** (`⌘ Space` → type `Terminal` → `Return`).
3. **Run the smoke test** to confirm all systems are ready:

   ```bash
   cd ~/Talk2Me
   bash scripts/smoke_test.sh
   ```

   All checks should say `[PASS]`.  If any say `[FAIL]`, see
   [Troubleshooting](#troubleshooting) below.

4. The installation starts automatically on boot (via launchd).  If it is already
   running you will see the standby tone playing from the speaker.  If not, start it:

   ```bash
   launchctl start com.talk2me.exhibit
   ```

---

## During the exhibit

### Normal operation

- A participant sits.  The installation plays a soft ambient tone in standby.
- Press **ENTER** in the consent terminal (or the physical button if wired) to
  begin a session.
- The participant speaks; the machine speaks back their questions in their own voice.
- At the end of the arc (or after 60 s of silence) the installation purges all
  captured audio and returns to standby automatically.

### Attendant reset (between participants)

Send a soft reset to start a fresh session immediately:

```bash
kill -USR1 $(launchctl list | awk '/talk2me/ {print $1}')
```

Or press the physical reset button if the installation is wired with one.

### Panic / emergency stop

If you need to immediately stop playback and wipe all session data:

```bash
kill -USR2 $(launchctl list | awk '/talk2me/ {print $1}')
```

The installation will silence, purge all participant audio, and return to standby.

---

## Reading the latency report

After the exhibit closes, print the day's timing data:

```bash
cd ~/Talk2Me
talk2me --report $(date +%Y-%m-%d)
```

The table shows per-turn latency.  Turns marked `SLOW` (>3 s) or `!!` (>5 s) are
worth noting for the artist's review.

---

## Shutdown

At the end of the day, stop the installation cleanly:

```bash
launchctl stop com.talk2me.exhibit
```

Then power off the Mac normally.

---

## First-time setup (developer / curator only)

Run once while the Mac has internet access:

```bash
cd ~/Talk2Me
uv sync                         # install Python dependencies
python scripts/prefetch_models.py   # download all model weights
```

To also download the optional LLM (adds ~2 GB, enables adaptive questions):

```bash
python scripts/prefetch_models.py --llm
```

After that, the Mac can operate with **no network**.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Smoke test: `[FAIL] talk2me package not importable` | Dependencies not installed | Run `uv sync` |
| Smoke test: `[FAIL] Whisper model load failed` | Model weights not cached | Run `python scripts/prefetch_models.py` |
| Smoke test: `[FAIL] F5-TTS model load failed` | Same as above | Same fix |
| Smoke test: `[FAIL] Speaker tone failed` | Wrong audio device | Check System Settings → Sound → Output |
| Installation won't start | launchd not loaded | Run `bash scripts/install_launchd.sh` |
| Latency > 5 s per turn | Thermal throttling | Check Activity Monitor; reboot if CPU is sustained at 100 % |
| No audio output | Mute or wrong device | Check speaker volume and output device in System Settings |

---

## Contact

For issues beyond this runbook, contact the technical lead before the next
session opens.
