"""Webots-specific behavior descriptor stubs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class InvertedPendulumDescriptor:
    """Summary stats for inverted pendulum (placeholder)."""

    def finalize(self) -> NDArray[np.float32]:
        return np.zeros(4, dtype=np.float32)


@dataclass
class LineFollowerDescriptor:
    """Descriptor placeholder for line follower task."""

    def finalize(self) -> NDArray[np.float32]:
        return np.zeros(3, dtype=np.float32)


@dataclass
class NavigationMazeDescriptor:
    """Descriptor placeholder for navigation maze task."""

    def finalize(self) -> NDArray[np.float32]:
        return np.zeros(5, dtype=np.float32)
