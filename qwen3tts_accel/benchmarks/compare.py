from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass
class BenchmarkResult:
    name: str
    runs: int
    latencies: list[float]

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.latencies) * 1000.0

    @property
    def min_ms(self) -> float:
        return min(self.latencies) * 1000.0

    @property
    def max_ms(self) -> float:
        return max(self.latencies) * 1000.0


def run_benchmark_case(
    name: str,
    runner: Callable[..., Any],
    *,
    warmup: int = 1,
    runs: int = 5,
    kwargs: dict[str, Any] | None = None,
) -> BenchmarkResult:
    call_kwargs = kwargs or {}

    for _ in range(max(warmup, 0)):
        runner(**call_kwargs)

    latencies: list[float] = []
    for _ in range(max(runs, 1)):
        start = time.perf_counter()
        runner(**call_kwargs)
        latencies.append(time.perf_counter() - start)

    return BenchmarkResult(name=name, runs=len(latencies), latencies=latencies)


def run_benchmark_suite(
    cases: Iterable[tuple[str, Callable[..., Any]]],
    *,
    warmup: int = 1,
    runs: int = 5,
) -> list[BenchmarkResult]:
    return [
        run_benchmark_case(name=name, runner=runner, warmup=warmup, runs=runs)
        for name, runner in cases
    ]
