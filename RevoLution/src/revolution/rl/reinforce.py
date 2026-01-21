"""Minimal REINFORCE implementation for lifetime learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, cast

import torch


@dataclass(frozen=True)
class ReinforceConfig:
    """Configuration for the REINFORCE inner loop."""

    lr: float = 1e-2
    gamma: float = 1.0
    use_baseline: bool = False


class ReinforceLearner:
    """REINFORCE learner that updates a policy within an episode.

    Usage:
      - call initialize(policy, rng_seed) once per genome evaluation
      - call start_episode()
      - call on_step(log_prob, reward) for each step
      - call end_episode() to apply the update and get diagnostics
    """

    def __init__(self, config: ReinforceConfig) -> None:
        if config.lr <= 0:
            raise ValueError("lr must be positive")
        if config.gamma <= 0:
            raise ValueError("gamma must be positive")
        self._config = config
        self._optimizer: torch.optim.Optimizer | None = None
        self._initial_state: dict[str, torch.Tensor] | None = None
        self._log_probs: list[torch.Tensor] = []
        self._rewards: list[float] = []

    def initialize(self, policy: torch.nn.Module, rng_seed: int) -> None:
        """Bind to a policy and capture its initial parameters."""

        torch.manual_seed(rng_seed)
        self._optimizer = torch.optim.Adam(policy.parameters(), lr=self._config.lr)
        # Clone parameters so we can reset later exactly.
        self._initial_state = {
            key: value.detach().clone() for key, value in policy.state_dict().items()
        }

    def start_episode(self) -> None:
        """Reset per-episode buffers."""

        self._log_probs.clear()
        self._rewards.clear()

    def on_step(self, log_prob: torch.Tensor, reward: float) -> None:
        """Record the log-probability and reward for a step."""

        self._log_probs.append(log_prob)
        self._rewards.append(reward)

    def end_episode(self) -> dict[str, float]:
        """Compute REINFORCE loss, apply update, and return diagnostics."""

        if self._optimizer is None:
            raise RuntimeError("initialize() must be called before end_episode().")
        if not self._log_probs:
            return {"loss": 0.0, "grad_norm": 0.0}

        returns = _discounted_returns(self._rewards, self._config.gamma)
        returns_tensor = torch.tensor(returns, dtype=self._log_probs[0].dtype)

        if self._config.use_baseline:
            baseline = returns_tensor.mean()
            returns_tensor = returns_tensor - baseline

        loss = -(torch.stack(self._log_probs) * returns_tensor).sum()

        self._optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]

        grad_norm = _grad_l2_norm(self._optimizer.param_groups)
        self._optimizer.step()

        return {"loss": float(loss.detach().cpu().item()), "grad_norm": grad_norm}

    def reset_to_initial(self, policy: torch.nn.Module) -> None:
        """Restore policy parameters to their initial values."""

        if self._initial_state is None:
            raise RuntimeError("initialize() must be called before reset.")
        policy.load_state_dict(self._initial_state, strict=True)


def _discounted_returns(rewards: Iterable[float], gamma: float) -> list[float]:
    """Compute discounted returns (G_t) for REINFORCE."""

    returns: list[float] = []
    running = 0.0
    for reward in reversed(list(rewards)):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


def _grad_l2_norm(param_groups: list[dict[str, object]]) -> float:
    """Compute the L2 norm of all gradients (for logging)."""

    total = 0.0
    for group in param_groups:
        params = cast(Iterable[torch.nn.Parameter], group["params"])
        for param in params:
            if param.grad is None:
                continue
            total += float(param.grad.detach().pow(2).sum().item())
    return float(total**0.5)
