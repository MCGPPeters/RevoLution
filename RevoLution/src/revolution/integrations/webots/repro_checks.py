"""Determinism check utilities for Webots tasks (stub)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ReproCheckResult:
    seed: int
    drift_metric: float
    reproducible: bool


def compute_drift_metric(values_a: Iterable[float], values_b: Iterable[float]) -> float:
    """Compute a simple L2 drift metric between two sequences."""

    a = list(values_a)
    b = list(values_b)
    if len(a) != len(b):
        raise ValueError("Sequences must have the same length.")
    total = 0.0
    for x, y in zip(a, b, strict=True):
        total += (x - y) ** 2
    return float(total**0.5)
