from pathlib import Path

import pytest

from revolution.benchmarks.protocols import (
    compute_generations_for_budget,
    expected_steps,
)


def test_budget_matches_expected_steps() -> None:
    neat_config = Path(__file__).parent / "fixtures" / "neat_test.cfg"
    generations = compute_generations_for_budget(
        neat_config_path=str(neat_config),
        episodes_per_genome=2,
        max_steps=5,
        total_steps=50,
    )
    assert generations == 1

    steps = expected_steps(
        neat_config_path=str(neat_config),
        generations=generations,
        episodes_per_genome=2,
        max_steps=5,
    )
    assert steps == 50


def test_budget_rejects_mismatched_steps() -> None:
    neat_config = Path(__file__).parent / "fixtures" / "neat_test.cfg"
    with pytest.raises(ValueError):
        compute_generations_for_budget(
            neat_config_path=str(neat_config),
            episodes_per_genome=2,
            max_steps=5,
            total_steps=51,
        )
