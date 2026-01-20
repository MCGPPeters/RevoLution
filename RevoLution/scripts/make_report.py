"""Entry point for report generation.

Phase 0 placeholder: CLI wiring only.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a report bundle from stored runs (placeholder)."
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Directory containing run artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to write report bundles.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    _args = parser.parse_args()
    # Phase 0: no execution logic yet. We exit cleanly after parsing.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
