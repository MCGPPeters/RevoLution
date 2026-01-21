"""Run artifact loading for reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RunArtifacts:
    """Container for run artifacts loaded from disk."""

    run_dir: Path
    episodes: pd.DataFrame
    generations: pd.DataFrame
    archive: pd.DataFrame
    config_path: Path
    metadata_path: Path


class RunLoader:
    """Discover and load run artifacts from a runs directory."""

    def __init__(self, runs_dir: str | Path):
        self._runs_dir = Path(runs_dir)

    def discover(self) -> list[Path]:
        if not self._runs_dir.exists():
            return []
        return sorted([path for path in self._runs_dir.iterdir() if path.is_dir()])

    def load_run(self, run_dir: str | Path) -> RunArtifacts:
        run_path = Path(run_dir)
        episodes = pd.read_csv(run_path / "raw" / "episodes.csv")
        generations = pd.read_csv(run_path / "raw" / "generations.csv")
        archive = pd.read_csv(run_path / "raw" / "archive.csv")
        config_path = run_path / "config.yaml"
        metadata_path = run_path / "metadata.json"

        return RunArtifacts(
            run_dir=run_path,
            episodes=episodes,
            generations=generations,
            archive=archive,
            config_path=config_path,
            metadata_path=metadata_path,
        )
