"""Talk2Me orchestration loop.

Feature 5: end-to-end single-turn pipeline —
    record_utterance → Transcriber.transcribe → VoiceCloner.synthesize → Speaker.play

Feature 6 wired: participant utterances accumulate in ReferenceBuffer; each
successive turn uses a progressively sharper voice clone.

Feature 7 wired: when engine.migration=true, VoiceCloner.synthesize() receives
a per-tier migration_alpha from ReferenceBuffer, blending neutral→self voice.

Features 8+9 wired: ConversationEngine (backed by QuestionBank) replaces the
hard-coded placeholder question with a phase-advancing, topic-biased selector.

Latency is instrumented at every stage; per-turn breakdown is printed and
accumulated for the session log.
"""
from __future__ import annotations

import time
import datetime
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


_REPROMPT_QUESTION = "Take your time — I'm listening."


def _load_config(path: str = "config/exhibit.yaml") -> dict:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).parent.parent / path
    with open(cfg_path) as f:
        return yaml.safe_load(f)


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


def run_loop(
    *,
    config_path: str = "config/exhibit.yaml",
    max_turns: Optional[int] = None,
    save_audio: bool = True,
) -> None:
    """Run the Talk2Me exhibit loop.

    Args:
        config_path: Path to exhibit.yaml (absolute or relative to project root).
        max_turns: Optional hard cap on turns (useful for scripted verification).
        save_audio: If True, save per-turn synthesis output to saved_audio/.
    """
    cfg = _load_config(config_path)
    audio_cfg = cfg.get("audio", {})
    stt_cfg = cfg.get("stt", {})
    engine_cfg = cfg.get("engine", {})

    out_dir = Path(__file__).parent.parent / "saved_audio"
    out_dir.mkdir(exist_ok=True)

    questions_dir = Path(__file__).parent.parent / "questions"

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
    speaker = Speaker(device=audio_cfg.get("output_device"))
    transcriber = Transcriber(model=stt_cfg.get("model", "mlx-community/whisper-large-v3-turbo"))
    cloner = VoiceCloner()

    migration_enabled = engine_cfg.get("migration", True)
    migration_alphas = tuple(engine_cfg.get("migration_alphas", [0.2, 0.5, 0.8, 1.0]))

    ref_buffer = ReferenceBuffer(migration_alphas=migration_alphas)

    bank = QuestionBank()
    bank.load(questions_dir)

    engine = ConversationEngine(
        question_bank=bank,
        calibration_turns=engine_cfg.get("calibration_turns", 2),
        personal_turns=engine_cfg.get("personal_turns", 3),
        idle_timeout_seconds=engine_cfg.get("idle_timeout_seconds", 60.0),
    )

    no_speech_threshold = stt_cfg.get("no_speech_threshold", 0.6)
    max_utterance_s = audio_cfg.get("max_utterance_seconds", 30)
    silence_ms = audio_cfg.get("silence_threshold_ms", 800)
    noise_cal_s = audio_cfg.get("noise_calibration_seconds", 2)

    # ── pre-warm models ───────────────────────────────────────────────────────
    print("[app] Pre-warming Whisper … (transcribe silent input)")
    _ = transcriber.transcribe(np.zeros(16_000, dtype=np.float32))

    print("[app] Pre-warming F5-TTS … (throwaway synthesis)")
    cloner.warm()

    if migration_enabled and not cloner.has_neutral_seed:
        print("[app] Warning: migration=true but no neutral seed found — "
              "place assets/neutral_seed.wav to enable voice ramp.")

    print("[app] Building VAD gate …")
    gate = build_gate(mic, calibration_seconds=noise_cal_s)

    print("[app] Ready.  Listening for participant …\n")

    # ── per-session state ─────────────────────────────────────────────────────
    logical_turn = 0          # counts only completed turns
    session_latencies: list[dict] = []
    speaker_for_tts: Optional[Speaker] = None  # created once at 24 kHz

    # ── main loop ─────────────────────────────────────────────────────────────
    while True:
        if max_turns is not None and logical_turn >= max_turns:
            break

        # Check idle timeout → reset session
        if logical_turn > 0 and engine.should_reset():
            print("[app] Idle timeout — resetting session.")
            ref_buffer.reset()
            engine.reset()
            logical_turn = 0
            session_latencies.clear()
            speaker_for_tts = None
            print("[app] Session reset. Listening for next participant …\n")

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

            # 2. Transcribe
            t_stt = time.perf_counter()
            result = transcriber.transcribe(wav)
            stt_s = time.perf_counter() - t_stt

            if not transcriber.is_speech(result, threshold=no_speech_threshold):
                print(f"[app] Likely non-speech (no_speech_prob={result.no_speech_prob:.2f}), re-prompting.")
                continue

            print(f"[app] Transcript: {result.text!r}")

            # 3. Push to reference buffer
            ref_buffer.push(wav, result, sample_rate=MIC_RATE)
            ref_wav, ref_text = ref_buffer.best_reference()

            if ref_wav is None or not ref_text:
                print("[app] No usable reference audio yet, re-prompting.")
                continue

            # 4. Choose question (Feature 8+9)
            logical_turn_next = logical_turn + 1
            question = engine.next_question()
            print(f"[app] Phase {engine.phase}, turn {logical_turn_next}: {question!r}")

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

            # 8. Record completed turn in engine
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

            if save_audio:
                _save_wav(synth_wav, cloner.sample_rate,
                          f"t{logical_turn:02d}_p{engine.phase}_output", out_dir)

        except KeyboardInterrupt:
            print("\n[app] KeyboardInterrupt — shutting down.")
            break
        except Exception as exc:
            print(f"[app] Error on turn {logical_turn + 1}: {exc!r} — continuing.")
            continue

    # ── session summary ───────────────────────────────────────────────────────
    if session_latencies:
        avg_total = sum(l["total"] for l in session_latencies) / len(session_latencies)
        print(f"\n[app] Session complete — {len(session_latencies)} turns, avg latency {avg_total:.2f}s")
    print("[app] Goodbye.")


def main() -> None:
    """Entry point for `talk2me` CLI command."""
    import argparse

    parser = argparse.ArgumentParser(description="Talk2Me art installation")
    parser.add_argument("--config", default="config/exhibit.yaml")
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--no-save-audio", action="store_true")
    args = parser.parse_args()

    run_loop(
        config_path=args.config,
        max_turns=args.max_turns,
        save_audio=not args.no_save_audio,
    )


if __name__ == "__main__":
    main()
