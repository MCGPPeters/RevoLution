"""Sample random rollout in the bandit environment.

This script is a minimal sanity check to show how an environment + descriptor
work together. It does not run evolution or learning yet.
"""

from __future__ import annotations

import argparse

from revolution.envs.bandit import BanditConfig, MultiArmedBanditEnv
from revolution.envs.descriptors import ActionFrequencyDescriptorExtractor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a random bandit rollout.")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed.")
    parser.add_argument("--steps", type=int, default=10, help="Steps to run.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Define a simple 3-arm bandit with increasing reward probabilities.
    config = BanditConfig(reward_probs=(0.1, 0.5, 0.9), max_steps=args.steps)
    env = MultiArmedBanditEnv(config)
    extractor = ActionFrequencyDescriptorExtractor(num_actions=env.action_space.n)

    # Reset with a seed so the reward stream is reproducible.
    obs, _info = env.reset(seed=args.seed)
    extractor.start_episode()

    done = False
    total_reward = 0.0
    while not done:
        # Random policy: sample uniformly from the action space.
        action = env.action_space.sample()
        obs, reward, _terminated, truncated, _info = env.step(action)
        extractor.on_step(action)
        total_reward += reward
        done = truncated

    extractor.end_episode()
    descriptor = extractor.finalize()

    print("Total reward:", total_reward)
    print("Action frequency descriptor:", descriptor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
