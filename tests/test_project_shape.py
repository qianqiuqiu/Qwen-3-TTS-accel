from __future__ import annotations

from pathlib import Path


def test_optimization_package_exists():
    package_root = Path(__file__).resolve().parents[1] / "qwen3tts_accel"

    assert package_root.is_dir()
    assert (package_root / "preprocess").is_dir()
    assert (package_root / "subtalker").is_dir()
    assert (package_root / "vllm").is_dir()
    assert (package_root / "benchmarks").is_dir()


def test_service_layer_package_removed():
    legacy_root = Path(__file__).resolve().parents[1] / "tts_server"

    assert not legacy_root.exists()
