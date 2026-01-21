"""Baseline registry for benchmark algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from revolution.config import ExperimentConfig, RLConfig, SelectionConfig


@dataclass(frozen=True)
class AlgorithmSpec:
    """Defines how to adapt the base experiment config for a baseline."""

    id: str
    name: str
    builder: Callable[[ExperimentConfig], ExperimentConfig]


class BenchmarkRegistry:
    """Registry for benchmark algorithms."""

    def __init__(self) -> None:
        self._algorithms: dict[str, AlgorithmSpec] = {}

    def register(self, spec: AlgorithmSpec) -> None:
        if spec.id in self._algorithms:
            raise ValueError(f"Algorithm {spec.id} already registered")
        self._algorithms[spec.id] = spec

    def get(self, algorithm_id: str) -> AlgorithmSpec:
        if algorithm_id not in self._algorithms:
            raise KeyError(f"Unknown algorithm id: {algorithm_id}")
        return self._algorithms[algorithm_id]

    def list_ids(self) -> list[str]:
        return sorted(self._algorithms.keys())


def default_registry() -> BenchmarkRegistry:
    registry = BenchmarkRegistry()

    def revolution_full(base: ExperimentConfig) -> ExperimentConfig:
        return base

    def neat_reward_only(base: ExperimentConfig) -> ExperimentConfig:
        selection = SelectionConfig(
            min_reward_enabled=False,
            min_reward_threshold=0.0,
            novelty_weight=0.0,
            reward_weight=1.0,
            penalty_below_threshold=base.selection.penalty_below_threshold,
        )
        return base.model_copy(
            update={"selection": selection, "rl": RLConfig(enabled=False)}
        )

    def neat_novelty_only(base: ExperimentConfig) -> ExperimentConfig:
        selection = SelectionConfig(
            min_reward_enabled=False,
            min_reward_threshold=0.0,
            novelty_weight=1.0,
            reward_weight=0.0,
            penalty_below_threshold=base.selection.penalty_below_threshold,
        )
        return base.model_copy(
            update={"selection": selection, "rl": RLConfig(enabled=False)}
        )

    registry.register(
        AlgorithmSpec(id="revolution_full", name="RevoLution", builder=revolution_full)
    )
    registry.register(
        AlgorithmSpec(
            id="neat_reward_only", name="NEAT Reward Only", builder=neat_reward_only
        )
    )
    registry.register(
        AlgorithmSpec(
            id="neat_novelty_only", name="NEAT Novelty Only", builder=neat_novelty_only
        )
    )
    return registry
