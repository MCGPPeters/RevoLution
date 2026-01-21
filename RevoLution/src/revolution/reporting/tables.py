"""Table exports for reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(df.to_markdown(index=False))
