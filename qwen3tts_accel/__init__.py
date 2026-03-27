from __future__ import annotations

from .pipeline import Qwen3TTSAccelPipeline


def register_qwen3_tts_model() -> bool:
    """Register the vLLM plugin (optional, requires vllm)."""
    try:
        from .vllm import register_qwen3_tts_model as _register
        return _register()
    except ImportError:
        return False


def patch_qwen3tts_config() -> None:
    """Patch Qwen3TTSConfig for vLLM compatibility (optional)."""
    try:
        from .vllm import patch_qwen3tts_config as _patch
        _patch()
    except ImportError:
        pass


__all__ = [
    "Qwen3TTSAccelPipeline",
    "patch_qwen3tts_config",
    "register_qwen3_tts_model",
]
