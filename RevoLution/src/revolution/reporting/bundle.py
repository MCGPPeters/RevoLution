"""Bundle reporting artifacts into a reproducible directory."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from .aggregate import aggregate_runs
from .io import RunLoader
from .plots import plot_learning_curve
from .tables import write_markdown_table


@dataclass(frozen=True)
class ReportBundle:
    output_dir: Path
    report_path: Path


def generate_report_bundle(runs_dir: str, output_dir: str) -> ReportBundle:
    loader = RunLoader(runs_dir)
    run_dirs = loader.discover()
    if not run_dirs:
        raise ValueError("No runs found for reporting.")

    output = Path(output_dir)
    figures_dir = output / "figures"
    tables_dir = output / "tables"
    agg_dir = output / "agg"
    output.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    agg_dir.mkdir(parents=True, exist_ok=True)

    runs = [loader.load_run(path) for path in run_dirs]
    generations_frames = [run.generations for run in runs]

    aggregate = aggregate_runs(
        runs=generations_frames,
        step_column="generation",
        value_column="best_reward",
    )

    learning_curve_path = agg_dir / "learning_curves.csv"
    aggregate.learning_curves.to_csv(learning_curve_path, index=False)

    final_scores_path = agg_dir / "final_scores.csv"
    aggregate.final_scores.to_csv(final_scores_path, index=False)

    plot_learning_curve(
        aggregate.learning_curves,
        "generation",
        figures_dir / "learning_curve.png",
    )

    write_markdown_table(aggregate.final_scores, tables_dir / "final_scores.md")

    report_path = output / "report.md"
    _render_report(report_path, aggregate.learning_curves, aggregate.final_scores)

    metadata_path = output / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "runs_dir": os.path.abspath(runs_dir),
                "num_runs": len(runs),
            },
            handle,
            indent=2,
        )

    return ReportBundle(output_dir=output, report_path=report_path)


def _render_report(
    report_path: Path, learning_curves: pd.DataFrame, final_scores: pd.DataFrame
) -> None:
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=False)
    template = env.get_template("report.md.j2")
    content = template.render(
        num_points=len(learning_curves),
        num_runs=len(final_scores),
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(content)
