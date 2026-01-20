"""Behavior descriptor extractors for environments.

Descriptors should reflect behavior, not reward, and must be deterministic
for a fixed trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

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
