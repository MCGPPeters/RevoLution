"""Phase 6 evolution engine (NEAT + novelty + lifetime RL)."""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, cast

import neat
import numpy as np
import torch
import yaml
from gymnasium import spaces
from numpy.typing import NDArray

from revolution.config import ExperimentConfig, RootConfig
from revolution.envs.bandit import BanditConfig, MultiArmedBanditEnv
from revolution.envs.descriptors import ActionFrequencyDescriptorExtractor
from revolution.neat_adapter.genome_bridge import build_feedforward_policy
from revolution.novelty.archive import (
    ArchiveCapConfig,
    ArchiveConfig,
    ArchiveThresholdConfig,
    NoveltyArchive,
)
from revolution.novelty.distance import EuclideanDistanceMetric
from revolution.novelty.knn import compute_knn_novelty
from revolution.rl import NoLearningRule, ReinforceLearner
from revolution.rl.reinforce import ReinforceConfig
from revolution.seeding import derive_seed
from revolution.utils import stable_sorted


@dataclass(frozen=True)
class _GenomeEvalResult:
    genome_id: int
    reward_mean: float
    descriptor_mean: NDArray[np.floating[Any]]


def run_experiment(config_path: str) -> str:
    """Run a NEAT + novelty experiment and return the run directory."""

    with open(config_path, "r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    root_config = RootConfig.from_dict(raw_config)
    config = root_config.experiment

    run_dir = _prepare_run_dir(config, config_path)
    _write_metadata(run_dir)

    # Seed all stochastic sources used by neat-python and numpy.
    random.seed(config.seed)
    np.random.seed(config.seed)

    neat_config_path = _resolve_path(
        os.path.dirname(config_path), config.neat.config_path
    )
    neat_config = _load_neat_config(neat_config_path)

    bandit_config = BanditConfig(
        reward_probs=tuple(config.env.reward_probs),
        max_steps=config.env.max_steps,
    )
    _validate_bandit_against_neat(bandit_config, neat_config)

    novelty_archive = NoveltyArchive(
        ArchiveConfig(
            threshold=ArchiveThresholdConfig(
                initial=config.novelty.threshold_initial,
                adapt=config.novelty.threshold_adapt,
                target_add_rate_per_gen=config.novelty.target_add_rate_per_gen,
                increase_factor=config.novelty.increase_factor,
                decrease_factor=config.novelty.decrease_factor,
            ),
            cap=ArchiveCapConfig(
                enabled=True,
                max_size=config.novelty.archive_max_size,
                eviction="fifo",
            ),
        )
    )

    population = neat.Population(neat_config)
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    evaluator = _build_evaluator(
        config=config,
        bandit_config=bandit_config,
        novelty_archive=novelty_archive,
        population=population,
        run_dir=run_dir,
    )

    population.run(evaluator, n=config.generations)

    return run_dir


def _build_evaluator(
    config: ExperimentConfig,
    bandit_config: BanditConfig,
    novelty_archive: NoveltyArchive,
    population: neat.Population,
    run_dir: str,
) -> Callable[[list[tuple[int, neat.DefaultGenome]], neat.Config], None]:
    """Build the eval_genomes callback for neat-python."""

    episodes_path = os.path.join(run_dir, "raw", "episodes.csv")
    generations_path = os.path.join(run_dir, "raw", "generations.csv")
    archive_path = os.path.join(run_dir, "raw", "archive.csv")

    _init_csv(
        episodes_path,
        [
            "generation",
            "genome_id",
            "episode",
            "reward",
            "steps",
            "descriptor",
            "loss",
            "grad_norm",
        ],
    )
    _init_csv(
        generations_path,
        [
            "generation",
            "best_reward",
            "median_reward",
            "best_novelty",
            "archive_size",
            "num_species",
            "threshold",
        ],
    )
    _init_csv(archive_path, ["generation", "archive_size", "threshold"])

    metric = EuclideanDistanceMetric()

    generation_counter = 0

    def eval_genomes(
        genomes: list[tuple[int, neat.DefaultGenome]],
        neat_config: neat.Config,
    ) -> None:
        # neat-python passes genomes as list[(genome_id, genome)].
        nonlocal generation_counter
        generation = generation_counter
        novelty_archive.start_generation()

        # Sort genomes to ensure deterministic evaluation order.
        ordered = stable_sorted(genomes, key=lambda pair: pair[0])

        eval_results: list[_GenomeEvalResult] = []
        raw_rewards: list[float] = []

        for genome_id, genome in ordered:
            policy = build_feedforward_policy(genome, neat_config)
            learner = _build_lifetime_learner(config)
            init_seed = derive_seed(config.seed, generation, genome_id, 0, 999)
            learner.initialize(policy, rng_seed=init_seed)

            reward_mean, descriptor_mean, episode_logs = _evaluate_genome(
                config=config,
                bandit_config=bandit_config,
                genome_id=genome_id,
                policy=policy,
                learner=learner,
                generation=generation,
            )

            eval_results.append(
                _GenomeEvalResult(
                    genome_id=genome_id,
                    reward_mean=reward_mean,
                    descriptor_mean=descriptor_mean,
                )
            )
            raw_rewards.append(reward_mean)

            _append_rows(episodes_path, episode_logs)

        novelty_scores = compute_knn_novelty(
            population=[result.descriptor_mean for result in eval_results],
            archive=novelty_archive.query_all(),
            k=config.novelty.k,
            metric=metric,
        )

        # Assign fitness scores based on novelty + reward.
        best_novelty = 0.0
        for result, novelty in zip(eval_results, novelty_scores, strict=True):
            genome = _lookup_genome(genomes, result.genome_id)
            selection_score = _selection_score(novelty, result.reward_mean, config)
            genome.fitness = selection_score
            best_novelty = max(best_novelty, novelty)

            novelty_archive.consider_add(result.descriptor_mean, novelty=novelty)

        novelty_archive.end_generation()

        best_reward = max(raw_rewards) if raw_rewards else 0.0
        median_reward = float(np.median(raw_rewards)) if raw_rewards else 0.0

        _append_rows(
            generations_path,
            [
                {
                    "generation": generation,
                    "best_reward": best_reward,
                    "median_reward": median_reward,
                    "best_novelty": best_novelty,
                    "archive_size": len(novelty_archive.query_all()),
                    "num_species": len(population.species.species),
                    "threshold": novelty_archive.threshold,
                }
            ],
        )

        _append_rows(
            archive_path,
            [
                {
                    "generation": generation,
                    "archive_size": len(novelty_archive.query_all()),
                    "threshold": novelty_archive.threshold,
                }
            ],
        )

        generation_counter += 1

    return eval_genomes


def _evaluate_genome(
    config: ExperimentConfig,
    bandit_config: BanditConfig,
    genome_id: int,
    policy: torch.nn.Module,
    learner: ReinforceLearner | NoLearningRule,
    generation: int,
) -> tuple[float, NDArray[np.floating[Any]], list[dict[str, object]]]:
    """Evaluate a genome across multiple episodes and return aggregated results."""

    env = MultiArmedBanditEnv(bandit_config)
    action_space = cast(spaces.Discrete[np.integer[Any]], env.action_space)
    if not isinstance(action_space, spaces.Discrete):
        raise RuntimeError("Bandit action space must be Discrete.")
    num_actions = cast(int, action_space.n)

    episode_rewards: list[float] = []
    episode_descriptors: list[NDArray[np.floating[Any]]] = []
    episode_logs: list[dict[str, object]] = []

    for episode in range(config.episodes_per_genome):
        seed = derive_seed(config.seed, generation, genome_id, episode)
        _obs, _info = env.reset(seed=seed)
        torch.manual_seed(seed)

        extractor = ActionFrequencyDescriptorExtractor(num_actions=num_actions)
        extractor.start_episode()
        learner.start_episode()

        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action, log_prob = _sample_action(policy, _obs)
            _obs, reward, _terminated, truncated, _info = env.step(action)
            extractor.on_step(action)
            learner.on_step(log_prob, reward)
            total_reward += reward
            steps += 1
            done = truncated

        extractor.end_episode()
        descriptor = extractor.finalize()
        diagnostics = learner.end_episode()

        episode_rewards.append(total_reward)
        episode_descriptors.append(descriptor)
        episode_logs.append(
            {
                "generation": generation,
                "genome_id": genome_id,
                "episode": episode,
                "reward": total_reward,
                "steps": steps,
                "descriptor": json.dumps(descriptor.tolist()),
                "loss": diagnostics["loss"],
                "grad_norm": diagnostics["grad_norm"],
            }
        )

    reward_mean = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    descriptor_mean = (
        np.mean(episode_descriptors, axis=0).astype(np.float32)
        if episode_descriptors
        else np.zeros(num_actions, dtype=np.float32)
    )

    learner.reset_to_initial(policy)
    return reward_mean, descriptor_mean, episode_logs


def _sample_action(
    policy: torch.nn.Module, obs: NDArray[np.floating[Any]]
) -> tuple[int, torch.Tensor]:
    """Sample a discrete action from torch policy logits."""

    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    logits = policy(obs_tensor).squeeze(0)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()  # type: ignore[no-untyped-call]
    return int(action.item()), dist.log_prob(action)  # type: ignore[no-untyped-call]


def _build_lifetime_learner(
    config: ExperimentConfig,
) -> ReinforceLearner | NoLearningRule:
    if not config.rl.enabled:
        return NoLearningRule()
    return ReinforceLearner(
        ReinforceConfig(
            lr=config.rl.lr,
            gamma=config.rl.gamma,
            use_baseline=config.rl.use_baseline,
        )
    )


def _selection_score(novelty: float, reward: float, config: ExperimentConfig) -> float:
    """Compute scalar selection score for neat-python fitness."""

    selection = config.selection
    if selection.min_reward_enabled and reward < selection.min_reward_threshold:
        return -selection.penalty_below_threshold

    return selection.novelty_weight * novelty + selection.reward_weight * reward


def _lookup_genome(
    genomes: list[tuple[int, neat.DefaultGenome]], genome_id: int
) -> neat.DefaultGenome:
    for gid, genome in genomes:
        if gid == genome_id:
            return genome
    raise KeyError(f"Genome id {genome_id} not found")


def _load_neat_config(config_path: str) -> neat.Config:
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )


def _resolve_path(base_dir: str, target_path: str) -> str:
    if os.path.isabs(target_path):
        return target_path
    return os.path.normpath(os.path.join(base_dir, target_path))


def _validate_bandit_against_neat(
    bandit_config: BanditConfig, neat_config: neat.Config
) -> None:
    num_actions = len(bandit_config.reward_probs)
    genome_config = neat_config.genome_config

    if num_actions != genome_config.num_inputs:
        raise ValueError(
            "Bandit action count must match NEAT num_inputs. "
            f"Got {num_actions} vs {genome_config.num_inputs}."
        )
    if num_actions != genome_config.num_outputs:
        raise ValueError(
            "Bandit action count must match NEAT num_outputs. "
            f"Got {num_actions} vs {genome_config.num_outputs}."
        )


def _prepare_run_dir(config: ExperimentConfig, config_path: str) -> str:
    run_id = config.run_id or _default_run_id(config.seed)
    run_dir = os.path.join(config.output_dir, run_id)
    os.makedirs(os.path.join(run_dir, "raw"), exist_ok=True)

    # Store an immutable snapshot of the config for reproducibility.
    shutil.copy2(config_path, os.path.join(run_dir, "config.yaml"))

    return run_dir


def _default_run_id(seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-seed{seed}"


def _write_metadata(run_dir: str) -> None:
    metadata = {
        "git_hash": _git_hash(),
        "python_version": _python_version(),
    }
    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _python_version() -> str:
    import sys

    return sys.version


def _init_csv(path: str, fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def _append_rows(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writerows(rows)
