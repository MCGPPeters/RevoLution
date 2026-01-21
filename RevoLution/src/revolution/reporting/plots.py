"""Plotting utilities for report figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curve(
    data: pd.DataFrame, step_col: str, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(data[step_col], data["mean"], label="mean")
    ax.fill_between(
        data[step_col],
        data["mean"] - data["iqr"],
        data["mean"] + data["iqr"],
        alpha=0.2,
        label="IQR band",
    )
    ax.set_xlabel(step_col)
    ax.set_ylabel("return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
