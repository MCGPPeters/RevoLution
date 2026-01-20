"""Configuration models for experiment runs.

We use Pydantic so YAML configs are validated and defaulted consistently.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BanditEnvConfig(BaseModel):
    """Configuration for the multi-armed bandit environment."""

    reward_probs: list[float]
    max_steps: int = 200


class NeatConfig(BaseModel):
    """NEAT configuration wrapper (points to neat-python config file)."""

    config_path: str


class NoveltyConfig(BaseModel):
    """Novelty scoring and archive configuration."""

    k: int = 10
    threshold_initial: float = 1.0
    threshold_adapt: bool = True
    target_add_rate_per_gen: float = 0.05
    increase_factor: float = 1.05
    decrease_factor: float = 0.95
    archive_max_size: int = 5000


class SelectionConfig(BaseModel):
    """Selection score configuration (novelty primary, reward secondary)."""

    min_reward_enabled: bool = True
    min_reward_threshold: float = 0.0
    novelty_weight: float = 1.0
    reward_weight: float = 0.05
    penalty_below_threshold: float = 1_000_000.0


class ExperimentConfig(BaseModel):
    """Top-level experiment settings."""

    seed: int = 123
    generations: int = 5
    episodes_per_genome: int = 3
    neat: NeatConfig
    env: BanditEnvConfig
    novelty: NoveltyConfig = Field(default_factory=NoveltyConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    output_dir: str = "runs"
    run_id: str | None = None


class RootConfig(BaseModel):
    """Root wrapper to make the YAML shape explicit."""

    experiment: ExperimentConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RootConfig":
        return cls.model_validate(data)
