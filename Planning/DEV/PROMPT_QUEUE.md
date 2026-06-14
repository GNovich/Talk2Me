# PROMPT QUEUE — updated 2026-06-14

> Tasks are pulled in binding order from `ROADMAP.md`. Do not start a feature
> before all earlier ones are `COMPLETED`. Max 3 active tasks at a time.

---

## TASK 10 — Optional LLM-adaptive question selection  (ROADMAP Feature 10)
**Status:** COMPLETED 2026-06-14

`LLMAdapter` in `talk2me/engine/llm_adapter.py`. Wraps `mlx_lm` (Llama-3.2-3B-
Instruct-4bit). Drift guard falls back to original on empty/too-long/no-`?`
output. Wired into `ConversationEngine` via `llm_adapter=` param. Gated by
`engine.llm: false` (default). 5 unit tests pass. On-hardware latency
verification deferred to exhibit Mac.

---

## TASK 11 — Kiosk runtime & session lifecycle  (ROADMAP Feature 11)
**Status:** COMPLETED 2026-06-14

`supervisor_loop()` in `app.py` with max-3-restarts guard and standby mode.
SIGUSR1/SIGUSR2/SIGTERM signal handlers. `_setup_logging()` routes to file in
kiosk mode. `_standby_loop()` plays ambient 440 Hz pulses. `launchd` plist at
`scripts/com.talk2me.exhibit.plist` (valid XML). `scripts/install_launchd.sh`
helper. `--kiosk` CLI flag. 2 unit tests pass (supervisor crash recovery +
plist XML). On-hardware launchd test deferred.

---

## TASK 12 — Consent, safety & privacy layer  (ROADMAP Feature 12)
**Status:** COMPLETED 2026-06-14

`consent_gate()` before each session. `_purge_session(ref_buffer, engine)` in
`finally` block on every exit path. SIGUSR2 panic triggers immediate purge.
`_log_transcript()` opt-in only (`privacy.save_transcripts: false` default).
`_assert_no_network_egress()` blocks startup if models need download. `PRIVACY.md`
for gallery. 7 unit tests pass.

---

## TASK 13 — Telemetry, logging & latency dashboard  (ROADMAP Feature 13)
**Status:** PENDING — PLAN ONLY

**Plan:**
- Structured per-turn JSON logging: append one line per turn to
  `logs/telemetry_YYYY-MM-DD.jsonl` (always on, no participant data —
  latency numbers and phase only). Fields:
  `{ts, turn, phase, tier, alpha, stt_s, tts_s, play_s, total_s}`.
- New `talk2me/telemetry.py`: `TelemetryLogger` with `log_turn(...)` and
  `session_summary() -> dict` (averages + per-stage breakdown).
- CLI digest command `talk2me --report YYYY-MM-DD`: reads the JSONL file,
  prints a human-readable table of per-turn latencies and a session average.
  Flags any turn where `total_s > 3.0` as over-budget.
- Live attendant indicator: print a one-line health banner after each turn
  (`[OK]` if total < 3 s, `[SLOW]` if 3–5 s, `[!!]` if > 5 s).
- Wire `TelemetryLogger.log_turn()` into `app.py:run_loop()` after each
  completed turn.
- **Verify:** unit test that `log_turn()` writes a valid JSON line; unit test
  that `session_summary()` computes correct averages; check that
  `--report` doesn't crash on an empty log file.

---

## TASK 14 — Packaging & deployment  (ROADMAP Feature 14)
**Status:** PENDING — PLAN ONLY

**Plan:**
- `uv lock` to produce a pinned `uv.lock` file; commit it.
- `scripts/prefetch_models.py`: downloads and caches Whisper
  (`mlx-community/whisper-large-v3-turbo`), F5-TTS (`lucasnewman/f5-tts-mlx`),
  and optionally the LLM (`mlx-community/Llama-3.2-3B-Instruct-4bit`) using
  each library's own cache mechanism. Prints progress. Designed to run once
  while online before the exhibit Mac goes dark.
- `scripts/smoke_test.sh`: verifies mic input (list devices), speaker output
  (plays a tone), Whisper model loads (transcribes silence), F5-TTS model
  loads (synthesizes one word). Exit code 0 = healthy.
- Written attendant runbook: `RUNBOOK.md` — power-on, daily smoke test,
  attendant reset / panic / shutdown procedures, what to do if the pipeline
  hangs, contact information.
- Defer signed `.app`/DMG unless artist/curator requests it; `uv` env is the
  minimum viable deploy.
- **Verify:** `scripts/prefetch_models.py --dry-run` lists what would be
  downloaded; `plutil -lint scripts/com.talk2me.exhibit.plist` still passes;
  `scripts/smoke_test.sh` runs to completion on the development machine.
