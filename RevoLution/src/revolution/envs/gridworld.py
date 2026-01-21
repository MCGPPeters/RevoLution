"""Simple deterministic gridworld environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray


@dataclass(frozen=True)
class GridworldConfig:
    width: int = 5
    height: int = 5
    max_steps: int = 50
    start: tuple[int, int] | None = None


class GridworldEnv(gym.Env[NDArray[np.float32], int]):
    """Discrete gridworld with deterministic transitions."""

    metadata = {"render_modes": []}

    def __init__(self, config: GridworldConfig):
        super().__init__()
        if config.width <= 0 or config.height <= 0:
            raise ValueError("width/height must be positive")
        if config.max_steps <= 0:
            raise ValueError("max_steps must be positive")

        self._config = config
        self._rng = np.random.default_rng()
        self._step_count = 0
        self._pos = np.array([0, 0], dtype=np.int64)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0

        if self._config.start is None:
            x = int(self._rng.integers(0, self._config.width))
            y = int(self._rng.integers(0, self._config.height))
            self._pos = np.array([x, y], dtype=np.int64)
        else:
            self._pos = np.array(self._config.start, dtype=np.int64)

        return self._obs(), {}

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        x, y = self._pos
        if action == 0:  # up
            y = max(0, y - 1)
        elif action == 1:  # down
            y = min(self._config.height - 1, y + 1)
        elif action == 2:  # left
            x = max(0, x - 1)
        elif action == 3:  # right
            x = min(self._config.width - 1, x + 1)

        self._pos = np.array([x, y], dtype=np.int64)

        self._step_count += 1
        truncated = self._step_count >= self._config.max_steps
        terminated = False

        return self._obs(), 0.0, terminated, truncated, {"pos": self._pos.copy()}

    def _obs(self) -> NDArray[np.float32]:
        # Normalize coordinates to [0, 1] for stable scaling.
        x = self._pos[0] / max(1, self._config.width - 1)
        y = self._pos[1] / max(1, self._config.height - 1)
        return np.array([x, y], dtype=np.float32)
