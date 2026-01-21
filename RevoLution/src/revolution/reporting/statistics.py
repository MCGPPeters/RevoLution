"""Statistical tests and confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class StatTestResult:
    name: str
    p_value: float
    effect_size: float


def bootstrap_ci(
    values: Iterable[float], samples: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values_array = np.array(list(values), dtype=np.float64)
    if values_array.size == 0:
        raise ValueError("No values for bootstrap CI.")

    means = []
    for _ in range(samples):
        resample = rng.choice(values_array, size=values_array.size, replace=True)
        means.append(float(np.mean(resample)))
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def wilcoxon_signed(
    values_a: Iterable[float], values_b: Iterable[float]
) -> StatTestResult:
    a = np.array(list(values_a), dtype=np.float64)
    b = np.array(list(values_b), dtype=np.float64)
    if a.size != b.size:
        raise ValueError("Wilcoxon test requires paired samples of equal length.")
    stat, p_value = stats.wilcoxon(a, b)
    effect = _cliffs_delta(a, b)
    return StatTestResult(
        name="wilcoxon_signed_rank",
        p_value=float(p_value),
        effect_size=effect,
    )


def mann_whitney(
    values_a: Iterable[float], values_b: Iterable[float]
) -> StatTestResult:
    a = np.array(list(values_a), dtype=np.float64)
    b = np.array(list(values_b), dtype=np.float64)
    stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
    effect = _cliffs_delta(a, b)
    return StatTestResult(
        name="mann_whitney_u",
        p_value=float(p_value),
        effect_size=effect,
    )


def _cliffs_delta(
    a: np.ndarray[Any, np.dtype[np.float64]],
    b: np.ndarray[Any, np.dtype[np.float64]],
) -> float:
    total = 0
    for x in a:
        for y in b:
            if x > y:
                total += 1
            elif x < y:
                total -= 1
    return float(total) / float(a.size * b.size)
