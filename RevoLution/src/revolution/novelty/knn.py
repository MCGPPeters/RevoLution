"""kNN novelty scoring with deterministic ordering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .distance import DistanceMetric


@dataclass(frozen=True)
class _PoolItem:
    """Internal helper for stable tie-breaking in kNN selection."""

    descriptor: NDArray[np.floating[Any]]
    is_population: bool
    index: int


def compute_knn_novelty(
    population: list[NDArray[np.floating[Any]]],
    archive: list[NDArray[np.floating[Any]]],
    k: int,
    metric: DistanceMetric,
) -> list[float]:
    """Compute novelty for each population descriptor using kNN distance.

    We compute novelty as the mean distance to the k nearest neighbors in
    (population + archive), excluding the descriptor itself. Ties are broken
    deterministically using (distance, is_population, index).
    """

    if k <= 0:
        raise ValueError("k must be positive.")
    if not population:
        return []

    pool: list[_PoolItem] = []
    for idx, descriptor in enumerate(population):
        pool.append(_PoolItem(descriptor=descriptor, is_population=True, index=idx))
    for idx, descriptor in enumerate(archive):
        pool.append(_PoolItem(descriptor=descriptor, is_population=False, index=idx))

    novelty_scores: list[float] = []
    for pop_index, pop_descriptor in enumerate(population):
        distances: list[tuple[float, bool, int]] = []
        for item in pool:
            if item.is_population and item.index == pop_index:
                continue
            dist = metric.distance(pop_descriptor, item.descriptor)
            distances.append((dist, item.is_population, item.index))

        if not distances:
            novelty_scores.append(0.0)
            continue

        distances.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
        neighbors = distances[: min(k, len(distances))]
        mean_distance = float(np.mean([entry[0] for entry in neighbors]))
        novelty_scores.append(mean_distance)

    return novelty_scores
