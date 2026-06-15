#!/usr/bin/env python3
"""Pre-download all model weights for offline exhibit operation.

Run once, while the machine has internet access, before the exhibit Mac goes
dark.  At exhibit time the Mac has no network; this script ensures every model
is present in the local HuggingFace cache.

Usage:
    python scripts/prefetch_models.py            # download everything
    python scripts/prefetch_models.py --dry-run  # list what would be fetched

Exit code 0 = all models cached.  Non-zero = at least one download failed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── model registry ────────────────────────────────────────────────────────────
# Each entry: (label, model_id, fetch_fn)
# fetch_fn(model_id, dry_run) -> bool (True = OK / already cached)

def _fetch_whisper(model_id: str, dry_run: bool) -> bool:
    """Download Whisper weights via mlx_whisper / huggingface_hub."""
    print(f"  [whisper] {model_id}")
    if dry_run:
        return True
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id)
        print(f"  [whisper] OK — cached")
        return True
    except Exception as e:
        print(f"  [whisper] FAILED: {e}", file=sys.stderr)
        return False


def _fetch_f5tts(model_id: str, dry_run: bool) -> bool:
    """Download F5-TTS-MLX weights via huggingface_hub."""
    print(f"  [f5-tts ] {model_id}")
    if dry_run:
        return True
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id)
        print(f"  [f5-tts ] OK — cached")
        return True
    except Exception as e:
        print(f"  [f5-tts ] FAILED: {e}", file=sys.stderr)
        return False


def _fetch_llm(model_id: str, dry_run: bool) -> bool:
    """Download LLM weights (optional) via huggingface_hub."""
    print(f"  [llm    ] {model_id}")
    if dry_run:
        return True
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id)
        print(f"  [llm    ] OK — cached")
        return True
    except Exception as e:
        print(f"  [llm    ] FAILED: {e}", file=sys.stderr)
        return False


def _fetch_silero_vad(url: str, dry_run: bool) -> bool:
    """Download Silero VAD ONNX model to ~/.cache/talk2me/."""
    print(f"  [vad    ] Silero VAD ONNX")
    cache_path = Path.home() / ".cache" / "talk2me" / "silero_vad.onnx"
    if cache_path.exists():
        print(f"  [vad    ] already cached → {cache_path}")
        return True
    if dry_run:
        return True
    try:
        import urllib.request
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [vad    ] downloading … ", end="", flush=True)
        urllib.request.urlretrieve(url, cache_path)
        print("OK")
        return True
    except Exception as e:
        print(f"\n  [vad    ] FAILED: {e}", file=sys.stderr)
        return False


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-fetch Talk2Me model weights")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded without fetching anything.",
    )
    parser.add_argument(
        "--config",
        default="config/exhibit.yaml",
        help="Path to exhibit.yaml (reads model IDs from it).",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Also prefetch the optional LLM model (large download, ~2 GB).",
    )
    args = parser.parse_args()

    # Read model IDs from config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / args.config

    stt_model = "mlx-community/whisper-large-v3-turbo"
    f5tts_model = "lucasnewman/f5-tts-mlx"
    llm_model = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    silero_url = (
        "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
    )

    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            stt_model = cfg.get("stt", {}).get("model", stt_model)
            llm_model = cfg.get("engine", {}).get("llm_model", llm_model)
        except Exception:
            pass  # fallback to defaults above

    print("Talk2Me — model prefetch" + (" (DRY RUN — nothing will be downloaded)" if args.dry_run else ""))
    print()

    tasks = [
        ("STT (Whisper)", stt_model, _fetch_whisper),
        ("TTS (F5-TTS-MLX)", f5tts_model, _fetch_f5tts),
    ]

    # VAD ONNX (Silero)
    print("Models to fetch:")
    for label, model_id, _ in tasks:
        print(f"  • {label}: {model_id}")
    if args.llm:
        print(f"  • LLM (optional): {llm_model}")
    print(f"  • VAD: Silero VAD ONNX")
    print()

    if args.dry_run:
        print("Dry-run complete — no downloads performed.")
        return 0

    print("Downloading:")
    failures = 0
    for label, model_id, fetch_fn in tasks:
        ok = fetch_fn(model_id, dry_run=False)
        if not ok:
            failures += 1

    # Silero VAD
    if not _fetch_silero_vad(silero_url, dry_run=False):
        failures += 1

    # Optional LLM
    if args.llm:
        if not _fetch_llm(llm_model, dry_run=False):
            failures += 1

    print()
    if failures:
        print(f"DONE — {failures} download(s) failed.  Re-run when network is available.")
        return 1

    print("All models cached.  The installation can now run offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
