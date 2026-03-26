from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VllmEngineConfig:
    model_path: str
    max_batch_size: int = 8
    max_num_seqs: int = 8
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    sample_rate: int = 24000
    codec_decoder: Any | None = None
    helper_model: Any | None = None
    enable_cuda_graph_subtalker: bool = True
    extra_engine_args: dict[str, object] = field(default_factory=dict)
