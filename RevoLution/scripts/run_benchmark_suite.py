"""Entry point for running a benchmark suite across algorithms."""

from __future__ import annotations

import argparse

from revolution.benchmarks import run_benchmark_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a benchmark suite."
    )
    parser.add_argument(
        "--suite-config",
        type=str,
        required=True,
        help="Path to a benchmark suite YAML config.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    results = run_benchmark_suite(args.suite_config)
    print(f"Completed {len(results)} runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
