# PROMPT QUEUE — updated 2026-06-15

> Tasks are pulled in binding order from `ROADMAP.md`. Do not start a feature
> before all earlier ones are `COMPLETED`. Max 3 active tasks at a time.

---

## TASK 13 — Telemetry, logging & latency dashboard  (ROADMAP Feature 13)
**Status:** COMPLETED 2026-06-15

`TelemetryLogger` in `talk2me/telemetry.py`. `log_turn()` appends one JSON line
per completed turn to `logs/telemetry_YYYY-MM-DD.jsonl` (latency + phase only,
zero participant data). Health banner `[OK]`/`[SLOW]`/`[!!]` printed after each
turn. `session_summary()` computes per-stage averages. `print_report()` renders
a human-readable per-turn table; handles missing/empty log gracefully.
Wired into `app.py` `run_loop()` (dev + kiosk) and `supervisor_loop()`.
`talk2me --report YYYY-MM-DD` CLI digest. 13 unit tests pass.
On-hardware latency budget verification deferred to exhibit Mac.

---

## TASK 14 — Packaging & deployment  (ROADMAP Feature 14)
**Status:** COMPLETED 2026-06-15

`uv.lock` generated (85 packages, Python 3.13.13).
`scripts/prefetch_models.py`: downloads Whisper + F5-TTS-MLX + Silero VAD ONNX;
`--llm` flag for optional LLM; `--dry-run` lists without downloading; reads
model IDs from `exhibit.yaml`. `scripts/smoke_test.sh`: 6 checks (import,
audio devices, speaker tone, Whisper load, F5-TTS load, launchd plist lint);
**6/6 pass on dev machine**. `RUNBOOK.md` written for non-developer gallery
attendant. Signed `.app`/DMG deferred; `uv` env is minimum viable deploy.

---

> All 14 roadmap features are COMPLETED.
> Remaining open items are operational, not code tasks:
> - On-hardware audible verification (exhibit Mac with participants)
> - Latency budget confirmation (≈2–3 s round-trip target)
> - launchd boot test on dedicated Mac Mini
> - assets/neutral_seed.wav placement (operator task)
> - Signed .app/DMG packaging (if curator requests)
