"""Behavior descriptor extractors for environments.

Descriptors should reflect behavior, not reward, and must be deterministic
for a fixed trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class ActionFrequencyDescriptorExtractor:
    """Counts action frequencies over an episode and returns a normalized vector.

    The descriptor is a vector of length `num_actions` where each entry is the
    fraction of times that action was taken. This is deterministic given the
    action sequence.
    """

    num_actions: int

    def __post_init__(self) -> None:
        if self.num_actions <= 0:
            raise ValueError("num_actions must be positive")
        # We keep integer counts for accuracy, then normalize in finalize().
        self._counts = np.zeros(self.num_actions, dtype=np.int64)
        self._total = 0

    def start_episode(self) -> None:
        # Reset counters at the start of each episode.
        self._counts[:] = 0
        self._total = 0

    def on_step(self, action: int) -> None:
        # We only need action history for this descriptor.
        if action < 0 or action >= self.num_actions:
            raise ValueError(f"action {action} out of range")
        self._counts[action] += 1
        self._total += 1

    def end_episode(self) -> None:
        # No-op for this descriptor; hook kept for interface symmetry.
        return None

    def finalize(self) -> NDArray[np.float32]:
        if self._total == 0:
            return np.zeros(self.num_actions, dtype=np.float32)
        # Cast to float32 to keep descriptors compact and consistent.
        return (self._counts / self._total).astype(np.float32)


@dataclass
class GridworldDescriptorExtractor:
    """Descriptor: downsampled visitation heatmap + final position.

    We accumulate a visitation count grid, downsample it to a fixed size,
    then append the final (x, y) position normalized to [0, 1].
    """

    width: int
    height: int
    downsample_width: int = 4
    downsample_height: int = 4

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width/height must be positive")
        if self.downsample_width <= 0 or self.downsample_height <= 0:
            raise ValueError("downsample sizes must be positive")
        self._visits = np.zeros((self.height, self.width), dtype=np.float32)
        self._final_pos = np.array([0, 0], dtype=np.int64)

    def start_episode(self) -> None:
        self._visits[:, :] = 0.0
        self._final_pos = np.array([0, 0], dtype=np.int64)

    def on_step(
        self,
        obs: NDArray[np.floating[Any]],
        action: int,
        reward: float,
        next_obs: NDArray[np.floating[Any]],
        done: bool,
        info: dict[str, object],
    ) -> None:
        _ = action, reward, done, info
        # Observation is normalized position; convert back to grid indices.
        x = int(round(float(next_obs[0]) * (self.width - 1)))
        y = int(round(float(next_obs[1]) * (self.height - 1)))
        x = min(max(x, 0), self.width - 1)
        y = min(max(y, 0), self.height - 1)
        self._visits[y, x] += 1.0
        self._final_pos = np.array([x, y], dtype=np.int64)

    def end_episode(self) -> None:
        return None

    def finalize(self) -> NDArray[np.float32]:
        heatmap = _downsample(
            self._visits, self.downsample_height, self.downsample_width
        )
        heatmap = heatmap / max(1.0, float(self._visits.sum()))

        final_x = self._final_pos[0] / max(1, self.width - 1)
        final_y = self._final_pos[1] / max(1, self.height - 1)
        final_pos = np.array([final_x, final_y], dtype=np.float32)

        combined = np.concatenate([heatmap.flatten(), final_pos], axis=0)
        return np.asarray(combined, dtype=np.float32)


def _downsample(
    grid: NDArray[np.float32], out_h: int, out_w: int
) -> NDArray[np.float32]:
    """Downsample a 2D grid by block averaging."""

    in_h, in_w = grid.shape
    block_h = max(1, in_h // out_h)
    block_w = max(1, in_w // out_w)

    down = np.zeros((out_h, out_w), dtype=np.float32)
    for y in range(out_h):
        for x in range(out_w):
            y0 = y * block_h
            x0 = x * block_w
            y1 = min(y0 + block_h, in_h)
            x1 = min(x0 + block_w, in_w)
            down[y, x] = float(grid[y0:y1, x0:x1].mean())
    return down
