"""Benchmark protocol definitions and budget helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, cast

import neat
import yaml


@dataclass(frozen=True)
class BudgetConfig:
    """Budget for training/evaluation expressed in environment steps."""

    total_steps: int
    eval_every_steps: int
    eval_episodes: int


@dataclass(frozen=True)
class BenchmarkSuiteConfig:
    """Benchmark suite configuration parsed from YAML."""

    name: str
    env: dict[str, Any]
    neat_config_path: str
    episodes_per_genome: int
    seeds: list[int]
    algorithms: list[str]
    budget: BudgetConfig
    output_dir: str = "runs"


def load_benchmark_suite(path: str) -> BenchmarkSuiteConfig:
    base_dir = os.path.dirname(path)
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    suite = raw["suite"]
    budget = suite["budget"]
    neat_path = str(suite["neat_config"])
    if not os.path.isabs(neat_path):
        neat_path = os.path.normpath(os.path.join(base_dir, neat_path))
    return BenchmarkSuiteConfig(
        name=str(suite["name"]),
        env=dict(suite["env"]),
        neat_config_path=neat_path,
        episodes_per_genome=int(suite["episodes_per_genome"]),
        seeds=[int(seed) for seed in suite["seeds"]],
        algorithms=list(suite["algorithms"]),
        budget=BudgetConfig(
            total_steps=int(budget["total_steps"]),
            eval_every_steps=int(budget["eval_every_steps"]),
            eval_episodes=int(budget["eval_episodes"]),
        ),
        output_dir=str(suite.get("output_dir", "runs")),
    )


def compute_generations_for_budget(
    neat_config_path: str,
    episodes_per_genome: int,
    max_steps: int,
    total_steps: int,
) -> int:
    """Compute generations needed to meet the step budget exactly."""

    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        neat_config_path,
    )
    steps_per_gen = cast(int, neat_config.pop_size) * episodes_per_genome * max_steps
    if steps_per_gen <= 0:
        raise ValueError("steps_per_gen must be positive")

    if total_steps % steps_per_gen != 0:
        raise ValueError(
            "total_steps must be divisible by steps_per_gen "
            f"({total_steps} vs {steps_per_gen})."
        )
    return total_steps // steps_per_gen


def expected_steps(
    neat_config_path: str,
    generations: int,
    episodes_per_genome: int,
    max_steps: int,
) -> int:
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        neat_config_path,
    )
    pop_size = cast(int, neat_config.pop_size)
    return pop_size * generations * episodes_per_genome * max_steps
