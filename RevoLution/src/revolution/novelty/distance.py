"""Distance metrics for behavior descriptors."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class DistanceMetric(Protocol):
    """Protocol for distance computation between descriptors."""

    def distance(
        self, a: NDArray[np.floating[Any]], b: NDArray[np.floating[Any]]
    ) -> float:
        """Return the distance between two descriptor vectors."""


class EuclideanDistanceMetric:
    """Standard Euclidean distance implementation."""

    def distance(
        self, a: NDArray[np.floating[Any]], b: NDArray[np.floating[Any]]
    ) -> float:
        if a.shape != b.shape:
            raise ValueError("Descriptors must have the same shape.")
        # numpy.linalg.norm returns a numpy scalar; cast to float for clarity.
        return float(np.linalg.norm(a - b))
