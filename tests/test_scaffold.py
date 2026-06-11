"""Smoke tests for the MLX scaffold (Feature 1).

These verify package structure and imports — they do not require models or hardware.
"""
import importlib


def test_package_importable():
    mod = importlib.import_module("talk2me")
    assert mod is not None


def test_subpackages_importable():
    for sub in ("talk2me.audio", "talk2me.stt", "talk2me.tts", "talk2me.engine"):
        mod = importlib.import_module(sub)
        assert mod is not None


def test_app_module_importable():
    mod = importlib.import_module("talk2me.app")
    assert hasattr(mod, "main")


def test_config_exists():
    from pathlib import Path
    cfg = Path(__file__).parent.parent / "config" / "exhibit.yaml"
    assert cfg.exists(), "config/exhibit.yaml missing"


def test_saved_audio_dir_exists():
    from pathlib import Path
    d = Path(__file__).parent.parent / "saved_audio"
    assert d.is_dir(), "saved_audio/ directory missing"
