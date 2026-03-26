from __future__ import annotations

import importlib.util

from ..preprocess import PrefillPayload, Qwen3TTSPreprocessor
from ..state import SequenceState, SequenceStateStore, get_default_sequence_store
from .config import VllmEngineConfig
from .plugin import Qwen3TTSForVLLM
from .runner import AsyncEngineRunner, create_vllm_runner


def patch_qwen3tts_config() -> None:
    """Monkey-patch Qwen3TTSConfig.get_text_config to return talker_config."""
    try:
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    except ImportError:
        return

    if getattr(Qwen3TTSConfig, "_vllm_patched", False):
        return

    def _get_text_config(self, **kwargs):
        return self.talker_config

    Qwen3TTSConfig.get_text_config = _get_text_config
    Qwen3TTSConfig._vllm_patched = True


def register_qwen3_tts_model() -> bool:
    if importlib.util.find_spec("vllm") is None:
        return False

    patch_qwen3tts_config()

    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Qwen3TTSForConditionalGeneration",
        "qwen3tts_accel.vllm.plugin:Qwen3TTSForVLLM",
    )
    return True


def get_cuda_graph_subtalker():
    """Lazy import to avoid CUDA init at module load time."""
    from ..subtalker import CUDAGraphSubTalkerRunner, patch_code_predictor_cuda_graph

    return CUDAGraphSubTalkerRunner, patch_code_predictor_cuda_graph


__all__ = [
    "AsyncEngineRunner",
    "PrefillPayload",
    "Qwen3TTSForVLLM",
    "Qwen3TTSPreprocessor",
    "SequenceState",
    "SequenceStateStore",
    "VllmEngineConfig",
    "create_vllm_runner",
    "get_cuda_graph_subtalker",
    "get_default_sequence_store",
    "patch_qwen3tts_config",
    "register_qwen3_tts_model",
]
