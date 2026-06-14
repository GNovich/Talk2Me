# Talk2Me — Data Handling & Privacy Policy

This document is for gallery operators, venue staff, and participants.

---

## What this installation does with your voice

**Talk2Me** is an interactive art installation. When you participate:

1. Your voice is captured through a microphone.
2. The audio is transcribed locally on the installation computer.
3. The transcription and a short audio sample are used to synthesize a voice
   that sounds like yours — which is then used to ask you questions.

All of this happens **locally on a dedicated computer in the gallery**.
Your voice audio and transcripts are **never sent over the internet or to any
external service**.

---

## What is stored and for how long

| Data | Default | With explicit opt-in |
|------|---------|----------------------|
| Captured audio (your voice) | Held in memory only; permanently deleted at session end | Not retained beyond session even with opt-in |
| Speech transcripts | Held in memory only; permanently deleted at session end | Anonymized text appended to a local log file for curatorial review (no timestamps, no names) |
| Synthesized audio | Held in memory only; permanently deleted at session end | One WAV file per question saved locally for technical review |

**By default, no participant data is written to any file.**

---

## How to stop at any time

- Say "stop" or stand up and leave — the installation will detect silence and
  reset itself within 60 seconds.
- The gallery attendant has a physical reset at their station that immediately
  wipes all session data and returns the installation to idle.
- If you feel uncomfortable at any point, **you can leave immediately** — the
  session clears automatically.

---

## Biometric data notice

Your voice is biometric data. Under applicable privacy laws (GDPR, CCPA, BIPA,
and equivalents), we are required to inform you:

- Voice data is processed solely for the artistic experience described above.
- No voice data is sold, shared with third parties, or used for any purpose
  outside this installation.
- No voice profile or voiceprint is retained after your session ends.
- Opt-in transcript logging (if enabled) records text only — not audio — and
  contains no personally identifying information.

---

## For gallery operators: configuration

Privacy behaviour is controlled by `config/exhibit.yaml`:

```yaml
privacy:
  save_transcripts: false   # Change to true ONLY with participant opt-in consent
```

**`save_transcripts` must remain `false` unless:**
1. A clear opt-in consent mechanism is presented to the participant before
   the session starts.
2. The participant has explicitly agreed.
3. The gallery legal team has confirmed compliance with local privacy law.

---

## For gallery operators: incident response

If a participant requests deletion of their data mid-session, or if a privacy
incident occurs:

1. Press the attendant reset key (SIGUSR2 / panic) to immediately purge the
   current session's memory.
2. If transcript logging was enabled, locate `logs/transcripts_YYYY-MM-DD.log`
   and delete or redact the relevant lines.
3. Log the incident in the gallery incident register.

---

## Technical contact

For privacy questions or data requests related to this installation, contact
the artist / curator listed in the accompanying gallery materials.
