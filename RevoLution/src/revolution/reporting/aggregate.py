"""Aggregate run artifacts into aligned learning curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AggregateResult:
    """Aggregated learning curves and final summaries."""

    learning_curves: pd.DataFrame
    final_scores: pd.DataFrame


def aggregate_runs(
    runs: Iterable[pd.DataFrame],
    step_column: str,
    value_column: str,
) -> AggregateResult:
    """Aggregate multiple run dataframes by aligning on step_column."""

    run_list = list(runs)
    if not run_list:
        raise ValueError("No runs provided for aggregation.")

    all_steps = sorted(set().union(*[set(run[step_column]) for run in run_list]))
    aligned_rows: list[dict[str, float]] = []

    for step in all_steps:
        values = []
        for run in run_list:
            subset = run.loc[run[step_column] == step, value_column]
            if not subset.empty:
                values.append(float(subset.iloc[0]))
        if values:
            aligned_rows.append(
                {
                    step_column: float(step),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "iqr": float(np.subtract(*np.percentile(values, [75, 25]))),
                }
            )

    learning_curves = pd.DataFrame(aligned_rows)

    final_scores = pd.DataFrame(
        [
            {
                "final_mean": float(run[value_column].iloc[-1]),
                "final_median": float(run[value_column].iloc[-1]),
            }
            for run in run_list
        ]
    )

    return AggregateResult(learning_curves=learning_curves, final_scores=final_scores)
