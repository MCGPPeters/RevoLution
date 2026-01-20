"""Novelty archive with deterministic threshold and eviction behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ArchiveThresholdConfig:
    """Threshold adaptation configuration."""

    initial: float = 1.0
    adapt: bool = True
    target_add_rate_per_gen: float = 0.05
    increase_factor: float = 1.05
    decrease_factor: float = 0.95


@dataclass(frozen=True)
class ArchiveCapConfig:
    """Archive size cap configuration."""

    enabled: bool = True
    max_size: int = 5000
    eviction: str = "fifo"


@dataclass(frozen=True)
class ArchiveConfig:
    """Full configuration for the novelty archive."""

    threshold: ArchiveThresholdConfig = ArchiveThresholdConfig()
    cap: ArchiveCapConfig = ArchiveCapConfig()


class NoveltyArchive:
    """Store descriptors that are sufficiently novel.

    We expose start_generation/end_generation to make threshold adaptation
    deterministic and easy to control at the experiment level.
    """

    def __init__(self, config: ArchiveConfig):
        if config.threshold.initial <= 0:
            raise ValueError("threshold.initial must be positive")
        if config.cap.enabled and config.cap.max_size <= 0:
            raise ValueError("cap.max_size must be positive when enabled")
        if config.cap.eviction != "fifo":
            raise ValueError("Only fifo eviction is supported for determinism")

        self._config = config
        self._threshold = config.threshold.initial
        self._descriptors: list[NDArray[np.floating[Any]]] = []
        self._considered = 0
        self._added = 0

    @property
    def threshold(self) -> float:
        """Current novelty threshold."""

        return self._threshold

    def start_generation(self) -> None:
        """Reset per-generation counters for adaptive thresholding."""

        self._considered = 0
        self._added = 0

    def consider_add(
        self,
        descriptor: NDArray[np.floating[Any]],
        novelty: float,
        rng: np.random.Generator | None = None,
    ) -> bool:
        """Consider adding a descriptor to the archive.

        rng is accepted for future stochastic policies but is unused in the
        deterministic threshold policy.
        """

        _ = rng
        self._considered += 1

        if novelty < self._threshold:
            return False

        self._descriptors.append(np.array(descriptor, copy=True))
        self._added += 1

        if self._config.cap.enabled:
            self._enforce_cap()

        return True

    def end_generation(self) -> None:
        """Update the adaptive threshold based on add rate."""

        if not self._config.threshold.adapt or self._considered == 0:
            return

        add_rate = self._added / self._considered
        target = self._config.threshold.target_add_rate_per_gen

        if add_rate > target:
            self._threshold *= self._config.threshold.increase_factor
        elif add_rate < target:
            self._threshold *= self._config.threshold.decrease_factor

        if self._threshold <= 0:
            raise RuntimeError("Archive threshold became non-positive.")

    def query_all(self) -> list[NDArray[np.floating[Any]]]:
        """Return a copy of all descriptors in the archive."""

        return [descriptor.copy() for descriptor in self._descriptors]

    def _enforce_cap(self) -> None:
        while (
            self._config.cap.enabled
            and len(self._descriptors) > self._config.cap.max_size
        ):
            # FIFO eviction is deterministic and easy to reason about.
            self._descriptors.pop(0)
