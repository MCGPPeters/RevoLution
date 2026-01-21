"""Lifetime learning rule wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .reinforce import ReinforceConfig, ReinforceLearner


@dataclass(frozen=True)
class NoLearningRule:
    """Placeholder rule that performs no parameter updates."""

    def initialize(self, policy: torch.nn.Module, rng_seed: int) -> None:
        # We set a seed for determinism even though no updates occur.
        torch.manual_seed(rng_seed)

    def start_episode(self) -> None:
        return None

    def on_step(self, log_prob: torch.Tensor, reward: float) -> None:
        return None

    def end_episode(self) -> dict[str, float]:
        return {"loss": 0.0, "grad_norm": 0.0}

    def reset_to_initial(self, policy: torch.nn.Module) -> None:
        return None


def build_default_reinforce() -> ReinforceLearner:
    """Helper to build a default REINFORCE learner."""

    return ReinforceLearner(ReinforceConfig())
