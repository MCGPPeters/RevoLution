"""A minimal Gymnasium-style multi-armed bandit environment.

This environment is intentionally simple so we can validate determinism,
seeding, and descriptor extraction before introducing more complex dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray


@dataclass(frozen=True)
class BanditConfig:
    """Configuration for the bandit environment.

    reward_probs defines the per-arm Bernoulli success probability. This keeps
    the reward distribution transparent and easy to reason about.
    """

    reward_probs: tuple[float, ...]
    max_steps: int = 100


class MultiArmedBanditEnv(gym.Env[NDArray[np.float32], int]):
    """Gymnasium-style bandit with deterministic seeding.

    Observation is a fixed-size zero vector; bandits are stateless, so we use
    a placeholder observation to satisfy the Gymnasium interface.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: BanditConfig):
        super().__init__()
        if not config.reward_probs:
            raise ValueError("reward_probs must not be empty")
        if config.max_steps <= 0:
            raise ValueError("max_steps must be positive")

        # We keep configuration immutable so the environment behavior is stable.
        self._config = config
        # We use a local Generator instance instead of numpy's global RNG.
        self._rng = np.random.default_rng()
        # Track steps so we can truncate episodes deterministically.
        self._step_count = 0

        num_arms = len(config.reward_probs)
        # Gymnasium requires action/observation spaces for validation.
        self.action_space = spaces.Discrete(num_arms)
        # Observations are a fixed-size zero vector; bandits are stateless.
        self.observation_space = spaces.Box(
            low=0.0,
            high=0.0,
            shape=(num_arms,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        # Let Gymnasium record the seed; we also reseed our local RNG.
        super().reset(seed=seed)
        if seed is not None:
            # We explicitly create a new generator to avoid global RNG state.
            self._rng = np.random.default_rng(seed)
        # Reset episode step counter for truncation logic.
        self._step_count = 0
        shape = self.observation_space.shape
        if shape is None:
            raise RuntimeError("Observation space shape must be defined.")
        observation = np.zeros(shape, dtype=np.float32)
        info: dict[str, Any] = {}
        return observation, info

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        # Bernoulli reward: 1 with probability p, else 0.
        reward_prob = self._config.reward_probs[action]
        reward = float(self._rng.random() < reward_prob)

        # Bandits don't terminate on their own; we only truncate by max_steps.
        self._step_count += 1
        truncated = self._step_count >= self._config.max_steps
        terminated = False

        shape = self.observation_space.shape
        if shape is None:
            raise RuntimeError("Observation space shape must be defined.")
        observation = np.zeros(shape, dtype=np.float32)
        # Include step index for debugging/analysis.
        info: dict[str, Any] = {"step": self._step_count}
        return observation, reward, terminated, truncated, info
