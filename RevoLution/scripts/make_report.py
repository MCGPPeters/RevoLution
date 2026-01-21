"""Entry point for report generation."""

from __future__ import annotations

import argparse

from revolution.reporting import generate_report_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a report bundle from stored runs."
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
    args = parser.parse_args()
    bundle = generate_report_bundle(args.runs_dir, args.output_dir)
    print(f"Report generated: {bundle.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
