"""Entry point for running a benchmark suite across algorithms.

Phase 0 placeholder: CLI wiring only.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a benchmark suite (placeholder)."
    )
    parser.add_argument(
        "--suite-config",
        type=str,
        required=False,
        help="Path to a benchmark suite YAML config.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    _args = parser.parse_args()
    # Phase 0: no execution logic yet. We exit cleanly after parsing.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
