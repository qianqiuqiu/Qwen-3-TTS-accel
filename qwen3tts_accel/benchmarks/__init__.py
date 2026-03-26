from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..vllm.runner import collect_sampling_params
from .compare import BenchmarkResult, run_benchmark_case, run_benchmark_suite


@dataclass
class BenchmarkCase:
    name: str
    description: str
    runner: Callable[..., Any]


__all__ = [
    "BenchmarkCase",
    "BenchmarkResult",
    "collect_sampling_params",
    "run_benchmark_case",
    "run_benchmark_suite",
]
