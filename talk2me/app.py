"""Talk2Me orchestration loop.

Feature 5: end-to-end single-turn pipeline —
    record_utterance → Transcriber.transcribe → VoiceCloner.synthesize → Speaker.play

Feature 6 wired: participant utterances accumulate in ReferenceBuffer; each
successive turn uses a progressively sharper voice clone.

Feature 7 wired: when engine.migration=true, VoiceCloner.synthesize() receives
a per-tier migration_alpha from ReferenceBuffer, blending neutral→self voice.

Features 8+9 wired: ConversationEngine (backed by QuestionBank) replaces the
hard-coded placeholder question with a phase-advancing, topic-biased selector.

Feature 10 wired: when engine.llm=true in exhibit.yaml, LLMAdapter is loaded
and passed to ConversationEngine to lightly personalize bank questions using a
local Llama-3.2-3B-Instruct-4bit model.

Feature 11 wired: outer supervisor loop with restart-on-crash logic, standby
mode, SIGUSR1 attendant reset, and kiosk/headless file logging.

Feature 12 wired: consent_gate() before each session, SIGUSR2 panic / purge,
_purge_session() ensures buffers are wiped on every exit path, opt-in
transcript logging, and startup network-egress assertion.

Feature 13 wired: TelemetryLogger appends one JSON line per turn to
logs/telemetry_YYYY-MM-DD.jsonl (latency + phase only, no participant data).
A one-line health banner ([OK]/[SLOW]/[!!]) is printed after each turn.
``talk2me --report YYYY-MM-DD`` prints the full per-turn table.

Feature 16 wired: when ui=true in exhibit.yaml, a DashboardServer starts in a
background thread, serving a local web dashboard for the attendant at
http://127.0.0.1:<ui_port>/.  Status is updated each turn.

Feature 17 wired: (b) ConversationEngine.next_question() runs concurrently with
ref_buffer operations in a ThreadPoolExecutor after transcription completes;
(c) a fast Whisper model (stt.model_fast) is used for calibration turns;
(d) F5-TTS nfe_steps is configurable via tts.nfe_steps in exhibit.yaml.

Latency is instrumented at every stage; per-turn breakdown is printed and
accumulated for the session log.
"""
from __future__ import annotations

import concurrent.futures
import datetime
import logging
import os
import signal
import sys
import time
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# ── module-level flags set by signal handlers ─────────────────────────────────
_reset_requested = threading.Event()   # SIGUSR1 → attendant session reset
_panic_requested = threading.Event()   # SIGUSR2 → panic / purge + standby
_shutdown_requested = threading.Event()  # SIGTERM / SIGINT → clean shutdown

_REPROMPT_QUESTION = "Take your time — I'm listening."
_STANDBY_TIMEOUT_S = 300.0   # >5 min of total silence → standby mode
_MAX_RESTARTS = 3             # supervisor loop max restarts before standby
_RESTART_DELAY_S = 5.0


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_config(path: str = "config/exhibit.yaml") -> dict:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).parent.parent / path
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _setup_logging(*, kiosk: bool, project_root: Path) -> None:
    """Configure Python logging.  In kiosk mode write to logs/; in dev mode use stderr."""
    level = logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    if kiosk:
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d")
        log_path = log_dir / f"session_{ts}.log"
        logging.basicConfig(filename=str(log_path), level=level, format=fmt)
        # Suppress raw tracebacks to stdout in kiosk mode
        sys.excepthook = lambda exc_type, exc_val, exc_tb: logging.error(
            "Unhandled exception", exc_info=(exc_type, exc_val, exc_tb)
        )
    else:
        logging.basicConfig(stream=sys.stderr, level=level, format=fmt)


def _install_signal_handlers() -> None:
    """Wire SIGUSR1 (reset) and SIGUSR2 (panic) for attendant use."""
    def _on_usr1(signum, frame):
        _reset_requested.set()

    def _on_usr2(signum, frame):
        _panic_requested.set()

    def _on_term(signum, frame):
        _shutdown_requested.set()

    signal.signal(signal.SIGUSR1, _on_usr1)
    signal.signal(signal.SIGUSR2, _on_usr2)
    signal.signal(signal.SIGTERM, _on_term)


def _assert_no_network_egress(cfg: dict) -> None:
    """Raise RuntimeError if any model path looks like a live download URL.

    At exhibit time the Mac has no network; all weights must be cached locally.
    This check catches misconfigured model IDs that would silently hang.
    """
    cache_root = Path.home() / ".cache"
    stt_model = cfg.get("stt", {}).get("model", "")
    # mlx-whisper caches under ~/.cache/huggingface or ~/.cache/mlx_whisper
    # f5-tts-mlx caches under ~/.cache/huggingface
    # If a model ID is provided (not a local path), verify the cache exists
    def _id_is_cached(model_id: str) -> bool:
        if not model_id or Path(model_id).exists():
            return True  # local path — OK
        hf_cache = cache_root / "huggingface" / "hub"
        # HuggingFace snapshot dirs are named models--<org>--<repo>
        safe_id = model_id.replace("/", "--")
        candidate = hf_cache / f"models--{safe_id}"
        return candidate.exists()

    if stt_model and not _id_is_cached(stt_model):
        raise RuntimeError(
            f"[app] STT model '{stt_model}' not found in local cache. "
            "Run 'uv run python scripts/prefetch_models.py' while online."
        )


def _save_wav(wav: np.ndarray, sample_rate: int, label: str, out_dir: Path) -> None:
    """Save a float32 waveform to saved_audio/ for post-session review."""
    try:
        import soundfile as sf
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{label}_{ts}.wav"
        sf.write(str(out_path), wav, sample_rate)
        print(f"[app] Saved: {out_path.name}")
    except Exception as e:
        print(f"[app] Warning: could not save WAV — {e}")


def _log_transcript(text: str, log_path: Path) -> None:
    """Append an anonymized transcript line (no timestamps, no names)."""
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text.strip() + "\n")
    except Exception as e:
        print(f"[app] Warning: could not log transcript — {e}")


def _purge_session(ref_buffer, engine) -> None:
    """Wipe all per-session state so no participant data lingers."""
    ref_buffer.reset()
    engine.reset()
    logging.info("Session purged (ReferenceBuffer + ConversationEngine reset).")
    print("[app] Session purged — participant data cleared.")


# ── consent gate (Feature 12) ─────────────────────────────────────────────────

def consent_gate(privacy_cfg: dict, *, kiosk: bool = False) -> bool:
    """Block until the participant (or attendant) confirms consent.

    Returns True if consent granted, False if the operator chose to quit.
    In kiosk mode, consent is given by pressing ENTER on the attendant terminal.
    """
    print("\n" + "=" * 60)
    print("  TALK2ME — voice installation")
    print("  This session will temporarily capture and process your voice.")
    print("  No audio is stored beyond this session without your opt-in.")
    print("  Press ENTER to begin, or Q then ENTER to skip / quit.")
    print("=" * 60 + "\n")
    try:
        choice = input("> ").strip().lower()
        return choice not in {"q", "quit", "no", "exit"}
    except (EOFError, KeyboardInterrupt):
        return False


# ── standby mode (Feature 11) ─────────────────────────────────────────────────

def _standby(speaker_device: Optional[str] = None) -> None:
    """Idle state after prolonged silence — play a soft ambient pulse and wait."""
    print("[app] Standby mode — installation idle.  Waiting for participant …")
    print("[app]   (Send SIGUSR1 to reset, SIGTERM to shut down.)")
    # Soft ambient tone: 440 Hz sine at very low amplitude, 2 s
    sr = 24_000
    t = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False, dtype=np.float32)
    ambient = (0.02 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    try:
        from talk2me.audio.io import Speaker
        spk = Speaker(device=speaker_device, sample_rate=sr)
        spk.play(ambient, blocking=True)
    except Exception:
        time.sleep(2.0)


# ── single-session run loop ───────────────────────────────────────────────────

def run_loop(
    *,
    config_path: str = "config/exhibit.yaml",
    max_turns: Optional[int] = None,
    save_audio: bool = True,
    skip_consent: bool = False,
    telemetry_logger=None,
    ui_server=None,
) -> None:
    """Run one full Talk2Me session (consent → turns → purge).

    Raises SystemExit only on clean shutdown request.  All other exceptions
    propagate so the supervisor loop can decide to restart.

    Args:
        config_path: Path to exhibit.yaml.
        max_turns: Hard cap on turns (for scripted verification).
        save_audio: Write per-turn synthesis WAVs to saved_audio/.
        skip_consent: Skip the consent gate (used in tests and supervisor restarts
                      when the attendant has already confirmed).
        ui_server: Optional DashboardServer instance (Feature 16).
    """
    cfg = _load_config(config_path)
    audio_cfg = cfg.get("audio", {})
    stt_cfg = cfg.get("stt", {})
    tts_cfg = cfg.get("tts", {})
    engine_cfg = cfg.get("engine", {})
    privacy_cfg = cfg.get("privacy", {})
    kiosk = cfg.get("kiosk", False)

    project_root = Path(__file__).parent.parent
    out_dir = project_root / "saved_audio"
    out_dir.mkdir(exist_ok=True)
    questions_dir = project_root / "questions"

    # Transcript logging (opt-in, anonymized)
    transcript_log_path: Optional[Path] = None
    if privacy_cfg.get("save_transcripts", False):
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d")
        transcript_log_path = log_dir / f"transcripts_{ts}.log"
        print(f"[app] Transcript logging enabled → {transcript_log_path.name}")

    # ── build components ──────────────────────────────────────────────────────
    from talk2me.audio.io import Microphone, Speaker, SAMPLE_RATE as MIC_RATE
    from talk2me.audio.vad import record_utterance, build_gate
    from talk2me.stt.whisper import Transcriber
    from talk2me.tts.voice_cloner import VoiceCloner
    from talk2me.tts.reference_buffer import ReferenceBuffer
    from talk2me.engine.question_bank import QuestionBank
    from talk2me.engine.state_machine import ConversationEngine

    mic = Microphone(
        device=audio_cfg.get("input_device"),
        gain=audio_cfg.get("input_gain", 1.0),
    )
    speaker_for_tts: Optional[Speaker] = None  # created once at 24 kHz

    # Feature 17c: dual Whisper models — fast for calibration, full for later
    full_model = stt_cfg.get("model", "mlx-community/whisper-large-v3-turbo")
    fast_model = stt_cfg.get("model_fast") or full_model
    transcriber = Transcriber(model=full_model)
    transcriber_fast: Optional[Transcriber] = None
    if fast_model != full_model:
        transcriber_fast = Transcriber(model=fast_model)
        print(f"[app] Dual Whisper: fast={fast_model!r} / full={full_model!r}")

    # Feature 17d: configurable F5-TTS nfe_steps
    nfe_steps = tts_cfg.get("nfe_steps", 8)
    cloner = VoiceCloner(steps=nfe_steps)

    migration_enabled = engine_cfg.get("migration", True)
    migration_alphas = tuple(engine_cfg.get("migration_alphas", [0.2, 0.5, 0.8, 1.0]))

    ref_buffer = ReferenceBuffer(migration_alphas=migration_alphas)

    bank = QuestionBank()
    bank.load(questions_dir)

    # Optional LLM adapter (Feature 10)
    llm_adapter = None
    if engine_cfg.get("llm", False):
        from talk2me.engine.llm_adapter import LLMAdapter
        llm_model = engine_cfg.get("llm_model", "mlx-community/Llama-3.2-3B-Instruct-4bit")
        print(f"[app] Loading LLM adapter: {llm_model} …")
        llm_adapter = LLMAdapter(model_id=llm_model)

    engine = ConversationEngine(
        question_bank=bank,
        calibration_turns=engine_cfg.get("calibration_turns", 2),
        personal_turns=engine_cfg.get("personal_turns", 3),
        idle_timeout_seconds=engine_cfg.get("idle_timeout_seconds", 60.0),
        llm_adapter=llm_adapter,
    )

    no_speech_threshold = stt_cfg.get("no_speech_threshold", 0.6)
    max_utterance_s = audio_cfg.get("max_utterance_seconds", 30)
    silence_ms = audio_cfg.get("silence_threshold_ms", 800)
    noise_cal_s = audio_cfg.get("noise_calibration_seconds", 2)

    # ── pre-warm models ───────────────────────────────────────────────────────
    print("[app] Pre-warming Whisper … (transcribe silent input)")
    _ = transcriber.transcribe(np.zeros(16_000, dtype=np.float32))

    if transcriber_fast is not None:
        print("[app] Pre-warming fast Whisper …")
        _ = transcriber_fast.transcribe(np.zeros(16_000, dtype=np.float32))

    print(f"[app] Pre-warming F5-TTS (nfe_steps={nfe_steps}) … (throwaway synthesis)")
    cloner.warm()

    if llm_adapter is not None:
        print("[app] Pre-warming LLM adapter …")
        llm_adapter.warm()

    if migration_enabled and not cloner.has_neutral_seed:
        print("[app] Warning: migration=true but no neutral seed found — "
              "place assets/neutral_seed.wav to enable voice ramp.")

    print("[app] Building VAD gate …")
    gate = build_gate(mic, calibration_seconds=noise_cal_s)

    # ── consent gate (Feature 12) ─────────────────────────────────────────────
    if not skip_consent:
        consented = consent_gate(privacy_cfg, kiosk=kiosk)
        if not consented:
            print("[app] No consent — returning to standby.")
            return

    print("[app] Ready.  Listening for participant …\n")

    # ── per-session state ─────────────────────────────────────────────────────
    logical_turn = 0
    session_latencies: list[dict] = []
    _reset_requested.clear()
    _panic_requested.clear()

    # ── main loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            if _shutdown_requested.is_set():
                print("\n[app] Shutdown requested — exiting.")
                _purge_session(ref_buffer, engine)
                raise SystemExit(0)

            if _panic_requested.is_set():
                print("\n[app] PANIC — purging session and returning to standby.")
                _purge_session(ref_buffer, engine)
                _panic_requested.clear()
                return

            if _reset_requested.is_set():
                print("[app] Attendant reset — purging session.")
                _purge_session(ref_buffer, engine)
                logical_turn = 0
                session_latencies.clear()
                speaker_for_tts = None
                _reset_requested.clear()
                if ui_server is not None:
                    ui_server.reset_session()
                print("[app] Session reset. Listening for next participant …\n")
                if not skip_consent:
                    consented = consent_gate(privacy_cfg, kiosk=kiosk)
                    if not consented:
                        return
                continue

            if max_turns is not None and logical_turn >= max_turns:
                break

            # Check idle timeout → reset session
            if logical_turn > 0 and engine.should_reset():
                print("[app] Idle timeout — purging session.")
                _purge_session(ref_buffer, engine)
                logical_turn = 0
                session_latencies.clear()
                speaker_for_tts = None
                if ui_server is not None:
                    ui_server.reset_session()
                print("[app] Session reset. Listening for next participant …\n")
                if not skip_consent:
                    consented = consent_gate(privacy_cfg, kiosk=kiosk)
                    if not consented:
                        return
                continue

            try:
                # 1. Record
                print("[app] Listening …")
                wav = record_utterance(
                    mic,
                    max_seconds=max_utterance_s,
                    silence_ms=silence_ms,
                    _gate=gate,
                )

                if len(wav) == 0:
                    print("[app] No speech detected, re-prompting.")
                    continue

                # 2. Transcribe — Feature 17c: use fast model during calibration turns
                t_stt = time.perf_counter()
                in_calibration = (logical_turn + 1) <= engine_cfg.get("calibration_turns", 2)
                active_transcriber = (
                    transcriber_fast if (transcriber_fast is not None and in_calibration)
                    else transcriber
                )
                result = active_transcriber.transcribe(wav)
                stt_s = time.perf_counter() - t_stt

                if not active_transcriber.is_speech(result, threshold=no_speech_threshold):
                    print(
                        f"[app] Likely non-speech "
                        f"(no_speech_prob={result.no_speech_prob:.2f}), re-prompting."
                    )
                    continue

                print(f"[app] Transcript: {result.text!r}")
                if transcript_log_path is not None:
                    _log_transcript(result.text, transcript_log_path)

                # 3+4. Push to ref buffer and choose question concurrently (Feature 17b).
                # question selection runs in a thread while ref_buffer ops proceed on main;
                # saves ~1-2 s when LLM adapter is active.
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
                    question_future = _pool.submit(engine.next_question)

                    # Ref buffer ops on main thread (fast)
                    ref_buffer.push(wav, result, sample_rate=MIC_RATE)
                    ref_wav, ref_text = ref_buffer.best_reference()

                    # Wait for question (usually already done)
                    question = question_future.result()

                logical_turn_next = logical_turn + 1
                print(f"[app] Phase {engine.phase}, turn {logical_turn_next}: {question!r}")

                if ref_wav is None or not ref_text:
                    print("[app] No usable reference audio yet, re-prompting.")
                    continue

                # 5. Determine migration alpha (Feature 7)
                if migration_enabled and cloner.has_neutral_seed:
                    alpha = ref_buffer.migration_alpha()
                else:
                    alpha = 1.0

                # 6. Synthesize
                t_tts = time.perf_counter()
                synth_wav = cloner.synthesize(
                    question,
                    reference_wav=ref_wav,
                    reference_text=ref_text,
                    reference_sample_rate=MIC_RATE,
                    migration_alpha=alpha,
                )
                tts_s = time.perf_counter() - t_tts

                # 7. Play
                t_play = time.perf_counter()
                if speaker_for_tts is None:
                    speaker_for_tts = Speaker(
                        device=audio_cfg.get("output_device"),
                        sample_rate=cloner.sample_rate,
                    )
                speaker_for_tts.play(synth_wav, blocking=True)
                play_s = time.perf_counter() - t_play

                # 8. Record completed turn
                engine.record_turn(transcript=result.text, question_asked=question)
                logical_turn += 1

                total_s = stt_s + tts_s + play_s
                latencies = dict(stt=stt_s, tts=tts_s, play=play_s, total=total_s)
                session_latencies.append(latencies)
                print(
                    f"[app] Latency: STT={stt_s:.2f}s | TTS={tts_s:.2f}s | "
                    f"play={play_s:.2f}s | total={total_s:.2f}s  "
                    f"(tier={ref_buffer.tier}, voiced={ref_buffer.voiced_seconds:.1f}s, "
                    f"alpha={alpha:.2f}, phase={engine.phase})"
                )

                banner = "—"
                if telemetry_logger is not None:
                    banner = telemetry_logger.log_turn(
                        turn=logical_turn,
                        phase=engine.phase,
                        tier=ref_buffer.tier,
                        alpha=alpha,
                        stt_s=stt_s,
                        tts_s=tts_s,
                        play_s=play_s,
                        total_s=total_s,
                    )
                    print(f"[telemetry] {banner}")

                # Feature 16: update attendant dashboard
                if ui_server is not None:
                    ui_server.update_status(
                        phase=engine.phase,
                        turn=logical_turn,
                        tier=ref_buffer.tier,
                        alpha=alpha,
                        health=banner,
                        session_active=True,
                        last_transcript=result.text,
                        last_question=question,
                        voiced_s=ref_buffer.voiced_seconds,
                        stt_s=stt_s,
                        tts_s=tts_s,
                        total_s=total_s,
                    )

                if save_audio:
                    _save_wav(
                        synth_wav, cloner.sample_rate,
                        f"t{logical_turn:02d}_p{engine.phase}_output",
                        out_dir,
                    )

            except KeyboardInterrupt:
                print("\n[app] KeyboardInterrupt — shutting down.")
                _purge_session(ref_buffer, engine)
                raise SystemExit(0)

            except Exception as exc:
                print(f"[app] Error on turn {logical_turn + 1}: {exc!r} — continuing.")
                logging.error("Turn error", exc_info=True)
                continue

    finally:
        # Always purge on any exit path (normal, exception, return)
        _purge_session(ref_buffer, engine)

    # ── session summary ───────────────────────────────────────────────────────
    if session_latencies:
        avg_total = sum(lat["total"] for lat in session_latencies) / len(session_latencies)
        print(
            f"\n[app] Session complete — {len(session_latencies)} turns, "
            f"avg latency {avg_total:.2f}s"
        )
    print("[app] Goodbye.")


# ── supervisor loop (Feature 11) ──────────────────────────────────────────────

def supervisor_loop(
    *,
    config_path: str = "config/exhibit.yaml",
    save_audio: bool = True,
) -> None:
    """Outer kiosk supervisor: restarts run_loop on crash, enters standby on failure.

    - On clean exit (SystemExit or KeyboardInterrupt): shut down.
    - On any other exception: log, wait _RESTART_DELAY_S, restart (max _MAX_RESTARTS).
    - After _MAX_RESTARTS consecutive crashes: enter standby.
    - After standby timeout with no signal: try again.
    """
    cfg = _load_config(config_path)
    audio_cfg = cfg.get("audio", {})
    project_root = Path(__file__).parent.parent

    _install_signal_handlers()
    kiosk = cfg.get("kiosk", False)
    _setup_logging(kiosk=kiosk, project_root=project_root)

    # Network egress assertion (Feature 12)
    try:
        _assert_no_network_egress(cfg)
    except RuntimeError as e:
        print(f"[supervisor] STARTUP BLOCKED: {e}")
        logging.critical(str(e))
        sys.exit(1)

    consecutive_crashes = 0
    first_run = True

    from talk2me.telemetry import TelemetryLogger
    tlog = TelemetryLogger(project_root / "logs")

    # Feature 16: optional attendant UI dashboard
    ui_server = None
    if cfg.get("ui", False):
        from talk2me.ui.server import DashboardServer
        ui_server = DashboardServer(port=cfg.get("ui_port", 8765))
        ui_server.start()

    print(f"[supervisor] Talk2Me starting (PID {os.getpid()}).")
    print(f"[supervisor]   SIGUSR1 = attendant reset  |  SIGUSR2 = panic/purge  |  SIGTERM = shutdown")

    while not _shutdown_requested.is_set():
        try:
            run_loop(
                config_path=config_path,
                save_audio=save_audio,
                skip_consent=not first_run,  # attendant already consented at startup
                telemetry_logger=tlog,
                ui_server=ui_server,
            )
            consecutive_crashes = 0
            first_run = False

        except SystemExit:
            print("[supervisor] Clean shutdown.")
            return

        except KeyboardInterrupt:
            print("[supervisor] Keyboard interrupt — shutting down.")
            return

        except Exception as exc:
            consecutive_crashes += 1
            logging.error(
                "run_loop crashed (crash #%d): %r", consecutive_crashes, exc,
                exc_info=True,
            )
            print(
                f"[supervisor] Pipeline crash #{consecutive_crashes}: {exc!r}"
            )

            if consecutive_crashes >= _MAX_RESTARTS:
                print(
                    f"[supervisor] {_MAX_RESTARTS} consecutive crashes — "
                    "entering standby.  Fix the issue and send SIGUSR1 to retry."
                )
                _standby_loop(audio_cfg.get("output_device"))
                consecutive_crashes = 0
            else:
                print(f"[supervisor] Restarting in {_RESTART_DELAY_S:.0f}s …")
                time.sleep(_RESTART_DELAY_S)

        first_run = False

    print("[supervisor] Shutdown complete.")


def _standby_loop(speaker_device: Optional[str] = None) -> None:
    """Play ambient pulses every 30 s until a signal clears standby."""
    _reset_requested.clear()
    while not _shutdown_requested.is_set() and not _reset_requested.is_set():
        _standby(speaker_device)
        # Sleep in short chunks so we respond quickly to signals
        for _ in range(150):  # ~30 s total
            if _shutdown_requested.is_set() or _reset_requested.is_set():
                break
            time.sleep(0.2)
    _reset_requested.clear()


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the `talk2me` CLI command."""
    import argparse

    parser = argparse.ArgumentParser(description="Talk2Me art installation")
    parser.add_argument("--config", default="config/exhibit.yaml")
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--no-save-audio", action="store_true")
    parser.add_argument(
        "--kiosk",
        action="store_true",
        help="Run in supervised kiosk mode with crash-restart and file logging.",
    )
    parser.add_argument(
        "--report",
        metavar="YYYY-MM-DD",
        help="Print the latency report for the given date and exit.",
    )
    args = parser.parse_args()

    if args.report:
        from talk2me.telemetry import TelemetryLogger
        project_root = Path(__file__).parent.parent
        TelemetryLogger(project_root / "logs").print_report(args.report)
        return

    if args.kiosk:
        supervisor_loop(
            config_path=args.config,
            save_audio=not args.no_save_audio,
        )
    else:
        _install_signal_handlers()
        project_root = Path(__file__).parent.parent
        from talk2me.telemetry import TelemetryLogger
        tlog = TelemetryLogger(project_root / "logs")
        # Feature 16: optional UI in dev mode
        ui_server = None
        cfg_for_ui = _load_config(args.config)
        if cfg_for_ui.get("ui", False):
            from talk2me.ui.server import DashboardServer
            ui_server = DashboardServer(port=cfg_for_ui.get("ui_port", 8765))
            ui_server.start()
        run_loop(
            config_path=args.config,
            max_turns=args.max_turns,
            save_audio=not args.no_save_audio,
            telemetry_logger=tlog,
            ui_server=ui_server,
        )


if __name__ == "__main__":
    main()
