import json
from pathlib import Path

import yaml

from revolution.evolution import run_experiment


def _write_config(path: Path, neat_cfg: Path, run_id: str) -> None:
    data = {
        "experiment": {
            "seed": 42,
            "generations": 2,
            "episodes_per_genome": 1,
            "neat": {"config_path": str(neat_cfg)},
            "env": {"reward_probs": [0.2, 0.8], "max_steps": 5},
            "novelty": {"k": 2, "archive_max_size": 10},
            "selection": {
                "min_reward_enabled": True,
                "min_reward_threshold": 0.0,
                "novelty_weight": 1.0,
                "reward_weight": 0.05,
                "penalty_below_threshold": 1000000.0,
            },
            "output_dir": str(path.parent),
            "run_id": run_id,
        }
    }
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)


def _read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def test_engine_deterministic_outputs(tmp_path: Path) -> None:
    neat_cfg = Path(__file__).parent / "fixtures" / "neat_test.cfg"

    config_one = tmp_path / "config_one.yaml"
    config_two = tmp_path / "config_two.yaml"

    _write_config(config_one, neat_cfg, run_id="run-one")
    _write_config(config_two, neat_cfg, run_id="run-two")

    run_dir_one = Path(run_experiment(str(config_one)))
    run_dir_two = Path(run_experiment(str(config_two)))

    for filename in ["raw/episodes.csv", "raw/generations.csv", "raw/archive.csv"]:
        text_one = _read_text(run_dir_one / filename)
        text_two = _read_text(run_dir_two / filename)
        assert text_one == text_two

    metadata_one = json.loads(_read_text(run_dir_one / "metadata.json"))
    metadata_two = json.loads(_read_text(run_dir_two / "metadata.json"))
    assert metadata_one == metadata_two
