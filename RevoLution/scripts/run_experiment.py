"""Entry point for a single algorithm experiment run.

Phase 0 placeholder: we only provide CLI wiring and help text.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a single RevoLution experiment (placeholder)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to a YAML config file.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    _args = parser.parse_args()
    # Phase 0: no execution logic yet. We exit cleanly after parsing.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
