"""Benchmark runner that orchestrates multi-seed experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass

import yaml

from revolution.config import BanditEnvConfig, ExperimentConfig, NeatConfig
from revolution.evolution import run_experiment

from .protocols import (
    BenchmarkSuiteConfig,
    compute_generations_for_budget,
    expected_steps,
    load_benchmark_suite,
)
from .registry import BenchmarkRegistry, default_registry


@dataclass(frozen=True)
class BenchmarkResult:
    algorithm_id: str
    seed: int
    run_dir: str


def run_benchmark_suite(config_path: str) -> list[BenchmarkResult]:
    suite = load_benchmark_suite(config_path)
    registry = default_registry()
    return _run_suite(suite, registry)


def _run_suite(
    suite: BenchmarkSuiteConfig, registry: BenchmarkRegistry
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    env = suite.env
    if env.get("name") != "bandit":
        raise ValueError("Only bandit env is supported in Phase 8.")

    generations = compute_generations_for_budget(
        neat_config_path=suite.neat_config_path,
        episodes_per_genome=suite.episodes_per_genome,
        max_steps=int(env["max_steps"]),
        total_steps=suite.budget.total_steps,
    )

    base_config = ExperimentConfig(
        seed=0,
        generations=generations,
        episodes_per_genome=suite.episodes_per_genome,
        neat=NeatConfig(config_path=suite.neat_config_path),
        env=BanditEnvConfig(
            reward_probs=list(env["reward_probs"]),
            max_steps=int(env["max_steps"]),
        ),
    )

    _assert_budget(suite, generations)

    for algorithm_id in suite.algorithms:
        spec = registry.get(algorithm_id)
        for seed in sorted(suite.seeds):
            config = spec.builder(base_config)
            config = config.model_copy(
                update={
                    "seed": seed,
                    "output_dir": _algo_output_dir(suite, algorithm_id),
                    "run_id": f"seed{seed}",
                }
            )
            run_dir = _run_with_config(config, suite, algorithm_id, seed)
            results.append(
                BenchmarkResult(algorithm_id=algorithm_id, seed=seed, run_dir=run_dir)
            )

    return results


def _run_with_config(
    config: ExperimentConfig, suite: BenchmarkSuiteConfig, algorithm_id: str, seed: int
) -> str:
    output_dir = _algo_output_dir(suite, algorithm_id)
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, f"config_seed{seed}.yaml")
    payload = {"experiment": config.model_dump(mode="json")}
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)

    return run_experiment(config_path)


def _algo_output_dir(suite: BenchmarkSuiteConfig, algorithm_id: str) -> str:
    return os.path.join(suite.output_dir, "benchmarks", suite.name, algorithm_id)


def _assert_budget(suite: BenchmarkSuiteConfig, generations: int) -> None:
    env = suite.env
    expected = expected_steps(
        suite.neat_config_path,
        generations=generations,
        episodes_per_genome=suite.episodes_per_genome,
        max_steps=int(env["max_steps"]),
    )
    if expected != suite.budget.total_steps:
        raise ValueError(
            "Budget mismatch: expected "
            f"{expected} steps but budget is {suite.budget.total_steps}."
        )
