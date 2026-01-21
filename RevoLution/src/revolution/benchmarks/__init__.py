"""Benchmarking framework (protocols, registry, runner)."""

from .protocols import BenchmarkSuiteConfig, BudgetConfig, load_benchmark_suite
from .registry import BenchmarkRegistry
from .runner import run_benchmark_suite

__all__ = [
    "BenchmarkSuiteConfig",
    "BudgetConfig",
    "BenchmarkRegistry",
    "load_benchmark_suite",
    "run_benchmark_suite",
]
