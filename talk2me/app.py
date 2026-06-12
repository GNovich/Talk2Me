"""Talk2Me orchestration loop.

Feature 5: end-to-end single-turn pipeline —
    record_utterance → Transcriber.transcribe → VoiceCloner.synthesize → Speaker.play

Feature 6 wired: after each turn the participant's utterance is pushed into a
ReferenceBuffer; the next turn's synthesis uses the best accumulated reference.

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


_PLACEHOLDER_QUESTION = "What brought you here today?"
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
    tts_cfg = cfg.get("tts", {})

    out_dir = Path(__file__).parent.parent / "saved_audio"
    out_dir.mkdir(exist_ok=True)

    # ── build components ──────────────────────────────────────────────────────
    from talk2me.audio.io import Microphone, Speaker, SAMPLE_RATE as MIC_RATE
    from talk2me.audio.vad import record_utterance, build_gate
    from talk2me.stt.whisper import Transcriber
    from talk2me.tts.voice_cloner import VoiceCloner
    from talk2me.tts.reference_buffer import ReferenceBuffer

    mic = Microphone(
        device=audio_cfg.get("input_device"),
        gain=audio_cfg.get("input_gain", 1.0),
    )
    speaker = Speaker(device=audio_cfg.get("output_device"))
    transcriber = Transcriber(model=stt_cfg.get("model", "mlx-community/whisper-large-v3-turbo"))
    cloner = VoiceCloner()
    ref_buffer = ReferenceBuffer()

    no_speech_threshold = stt_cfg.get("no_speech_threshold", 0.6)
    max_utterance_s = audio_cfg.get("max_utterance_seconds", 30)
    silence_ms = audio_cfg.get("silence_threshold_ms", 800)
    noise_cal_s = audio_cfg.get("noise_calibration_seconds", 2)

    # ── pre-warm models ───────────────────────────────────────────────────────
    print("[app] Pre-warming Whisper … (transcribe silent input)")
    _ = transcriber.transcribe(np.zeros(16_000, dtype=np.float32))

    print("[app] Pre-warming F5-TTS … (throwaway synthesis)")
    cloner.warm()

    print("[app] Building VAD gate …")
    gate = build_gate(mic, calibration_seconds=noise_cal_s)

    print("[app] Ready.  Listening for participant …\n")

    # ── per-session state ─────────────────────────────────────────────────────
    turn = 0
    session_latencies: list[dict] = []

    # ── main loop ─────────────────────────────────────────────────────────────
    while True:
        if max_turns is not None and turn >= max_turns:
            break

        turn += 1
        print(f"\n── Turn {turn} ────────────────────────────────────────────")

        try:
            # 1. Record
            t_rec = time.perf_counter()
            print("[app] Listening …")
            wav = record_utterance(
                mic,
                max_seconds=max_utterance_s,
                silence_ms=silence_ms,
                _gate=gate,
            )
            rec_s = time.perf_counter() - t_rec

            if len(wav) == 0:
                print("[app] No speech detected, re-prompting.")
                turn -= 1
                continue

            # 2. Transcribe
            t_stt = time.perf_counter()
            result = transcriber.transcribe(wav)
            stt_s = time.perf_counter() - t_stt

            if not transcriber.is_speech(result, threshold=no_speech_threshold):
                print(f"[app] Likely non-speech (no_speech_prob={result.no_speech_prob:.2f}), re-prompting.")
                turn -= 1
                continue

            print(f"[app] Transcript: {result.text!r}")

            # 3. Push to reference buffer
            ref_buffer.push(wav, result, sample_rate=MIC_RATE)
            ref_wav, ref_text = ref_buffer.best_reference()

            # 4. Choose question
            question = _PLACEHOLDER_QUESTION

            # 5. Synthesize
            t_tts = time.perf_counter()
            if ref_wav is not None and ref_text:
                synth_wav = cloner.synthesize(
                    question,
                    reference_wav=ref_wav,
                    reference_text=ref_text,
                    reference_sample_rate=MIC_RATE,
                )
            else:
                # No reference yet (first turn didn't pass quality) — skip playback
                print("[app] No reference audio yet, skipping synthesis.")
                continue
            tts_s = time.perf_counter() - t_tts

            # 6. Play
            t_play = time.perf_counter()
            # Speaker was initialised at MIC_RATE (16 kHz); cloner outputs 24 kHz
            speaker_for_tts = Speaker(
                device=audio_cfg.get("output_device"),
                sample_rate=cloner.sample_rate,
            )
            speaker_for_tts.play(synth_wav, blocking=True)
            play_s = time.perf_counter() - t_play

            total_s = stt_s + tts_s + play_s
            latencies = dict(stt=stt_s, tts=tts_s, play=play_s, total=total_s)
            session_latencies.append(latencies)
            print(
                f"[app] Latency: STT={stt_s:.2f}s | TTS={tts_s:.2f}s | "
                f"play={play_s:.2f}s | total={total_s:.2f}s  "
                f"(tier={ref_buffer.tier}, voiced={ref_buffer.voiced_seconds:.1f}s)"
            )

            if save_audio:
                _save_wav(synth_wav, cloner.sample_rate, f"t{turn:02d}_output", out_dir)

        except KeyboardInterrupt:
            print("\n[app] KeyboardInterrupt — shutting down.")
            break
        except Exception as exc:
            print(f"[app] Error on turn {turn}: {exc!r} — continuing.")
            turn -= 1
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
