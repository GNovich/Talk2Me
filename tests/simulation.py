"""SimulatedParticipant — replays fixture audio through the Talk2Me pipeline.

Feature 15: offline regression testing without a microphone or live session.
Clips from tests/fixtures/voices/ are fed directly to
Transcriber → ReferenceBuffer → VoiceCloner, bypassing record_utterance().
Synthesized output is saved to saved_audio/sim_* for audible review.

Usage (in tests):
    from tests.simulation import SimulatedParticipant
    participant = SimulatedParticipant("speaker_f1")
    result = participant.run_turn(transcriber, ref_buffer, cloner, question)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "voices"
SAMPLE_RATE = 16_000


class SimulatedParticipant:
    """Replays pre-recorded fixture clips through the Talk2Me pipeline.

    Each call to run_turn() uses the next clip in the speaker's fixture set,
    cycling when all clips are exhausted.  Clips are loaded once on construction.
    """

    def __init__(
        self,
        speaker_id: str,
        sample_rate: int = SAMPLE_RATE,
        fixtures_dir: Optional[Path] = None,
    ):
        self.speaker_id = speaker_id
        self.sample_rate = sample_rate
        self._dir = fixtures_dir or FIXTURES_DIR
        self._clips: list[tuple[np.ndarray, str]] = []
        self._clip_idx: int = 0
        self._load_clips()

    # ── loading ───────────────────────────────────────────────────────────────

    def _load_clips(self) -> None:
        import soundfile as sf

        speaker_dir = self._dir / self.speaker_id
        if speaker_dir.is_dir():
            wav_files = sorted(speaker_dir.glob("*.wav"))
        else:
            # Flat layout: speaker_id prefix (e.g. speaker_f1.wav)
            wav_files = sorted(self._dir.glob(f"{self.speaker_id}*.wav"))

        if not wav_files:
            raise FileNotFoundError(
                f"No WAV fixtures for speaker '{self.speaker_id}' in {self._dir}. "
                "Run: uv run python scripts/generate_fixtures.py"
            )

        for wav_path in wav_files:
            txt_path = wav_path.with_suffix(".txt")
            wav, sr = sf.read(str(wav_path), dtype="float32")
            if wav.ndim > 1:
                wav = wav[:, 0]
            if sr != self.sample_rate:
                wav = _resample_to(wav, sr, self.sample_rate)
            transcript = (
                txt_path.read_text(encoding="utf-8").strip()
                if txt_path.exists()
                else ""
            )
            self._clips.append((wav, transcript))

    # ── clip access ───────────────────────────────────────────────────────────

    def next_clip(self) -> tuple[np.ndarray, str]:
        """Return (wav, transcript) for the next fixture clip, cycling on exhaustion."""
        if not self._clips:
            raise RuntimeError(f"No clips loaded for speaker '{self.speaker_id}'")
        wav, text = self._clips[self._clip_idx % len(self._clips)]
        self._clip_idx += 1
        return wav.copy(), text

    def reset(self) -> None:
        """Restart clip rotation from the first clip."""
        self._clip_idx = 0

    @property
    def clip_count(self) -> int:
        return len(self._clips)

    # ── pipeline execution ────────────────────────────────────────────────────

    def run_turn(
        self,
        transcriber,
        ref_buffer,
        cloner,
        question: str,
        out_dir: Optional[Path] = None,
        turn_idx: int = 0,
    ) -> dict:
        """Run one simulated turn through the full pipeline.

        Steps:
            1. Feed next fixture clip to Transcriber
            2. Push transcript + audio to ReferenceBuffer
            3. Synthesize question in cloned voice with VoiceCloner
            4. Optionally save synthesised WAV to out_dir/sim_<speaker>_t<n>.wav

        Returns a dict with timing and transcript info:
            {
                "stt_s": float,          # Whisper latency
                "tts_s": float,          # F5-TTS latency
                "total_s": float,        # stt + tts
                "transcript": str,       # what Whisper heard
                "synth_duration_s": float,  # synthesised audio length
                "tier": int,             # ReferenceBuffer tier after push
                "voiced_s": float,       # total voiced seconds accumulated
            }
        """
        wav, _hint = self.next_clip()

        # 1. Transcribe
        t0 = time.perf_counter()
        result = transcriber.transcribe(wav)
        stt_s = time.perf_counter() - t0

        # 2. Push to reference buffer
        ref_buffer.push(wav, result, sample_rate=self.sample_rate)
        ref_wav, ref_text = ref_buffer.best_reference()

        if ref_wav is None or not ref_text:
            return {
                "stt_s": stt_s, "tts_s": 0.0, "total_s": stt_s,
                "transcript": result.text, "synth_duration_s": 0.0,
                "tier": ref_buffer.tier, "voiced_s": ref_buffer.voiced_seconds,
            }

        # 3. Synthesize
        t1 = time.perf_counter()
        synth_wav = cloner.synthesize(
            question,
            reference_wav=ref_wav,
            reference_text=ref_text,
            reference_sample_rate=self.sample_rate,
        )
        tts_s = time.perf_counter() - t1

        # 4. Save output
        if out_dir is not None and len(synth_wav) > 0:
            try:
                import soundfile as sf
                out_dir = Path(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"sim_{self.speaker_id}_t{turn_idx:02d}.wav"
                sf.write(str(out_path), synth_wav, cloner.sample_rate)
            except Exception as exc:
                print(f"[sim] Warning: could not save WAV — {exc}")

        return {
            "stt_s": stt_s,
            "tts_s": tts_s,
            "total_s": stt_s + tts_s,
            "transcript": result.text,
            "synth_duration_s": len(synth_wav) / cloner.sample_rate if len(synth_wav) > 0 else 0.0,
            "tier": ref_buffer.tier,
            "voiced_s": ref_buffer.voiced_seconds,
        }

    def run_session(
        self,
        transcriber,
        cloner,
        questions: list[str],
        out_dir: Optional[Path] = None,
    ) -> list[dict]:
        """Run a full simulated session through a list of questions.

        Uses a fresh ReferenceBuffer for the session.  Returns a list of per-turn
        result dicts.  Latency numbers are the primary benchmark metric.
        """
        from talk2me.tts.reference_buffer import ReferenceBuffer
        ref_buffer = ReferenceBuffer()

        results: list[dict] = []
        for i, question in enumerate(questions):
            result = self.run_turn(
                transcriber, ref_buffer, cloner,
                question=question, out_dir=out_dir, turn_idx=i,
            )
            result["turn"] = i
            result["question"] = question
            results.append(result)
            print(
                f"[sim] {self.speaker_id} turn {i+1}: "
                f"STT={result['stt_s']:.2f}s  TTS={result['tts_s']:.2f}s  "
                f"total={result['total_s']:.2f}s  tier={result['tier']}  "
                f"voiced={result['voiced_s']:.1f}s"
            )

        return results


# ── helpers ───────────────────────────────────────────────────────────────────

def _resample_to(wav: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return wav
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(src_rate, dst_rate)
    return resample_poly(wav.astype(np.float64), dst_rate // g, src_rate // g).astype(np.float32)
