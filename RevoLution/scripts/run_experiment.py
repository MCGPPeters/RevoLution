"""Entry point for a single algorithm experiment run.

Phase 0 placeholder: we only provide CLI wiring and help text.
"""

from __future__ import annotations

import argparse

from revolution.evolution import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a single RevoLution experiment (placeholder)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config file.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run_dir = run_experiment(args.config)
    print(f"Run artifacts stored in: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
