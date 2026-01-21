import numpy as np

from revolution.envs.descriptors import (
    ActionFrequencyDescriptorExtractor,
    GridworldDescriptorExtractor,
)
from revolution.envs.gridworld import GridworldConfig, GridworldEnv


def test_action_frequency_descriptor_matches_histogram() -> None:
    extractor = ActionFrequencyDescriptorExtractor(num_actions=3)
    extractor.start_episode()

    actions = [0, 2, 2, 1, 2]
    for action in actions:
        extractor.on_step(action)

    extractor.end_episode()
    descriptor = extractor.finalize()

    expected = np.array([1 / 5, 1 / 5, 3 / 5], dtype=np.float32)
    assert np.allclose(descriptor, expected)


def test_action_frequency_descriptor_deterministic_for_same_actions() -> None:
    actions = [1, 1, 0, 2, 1, 0]

    extractor_a = ActionFrequencyDescriptorExtractor(num_actions=3)
    extractor_a.start_episode()
    for action in actions:
        extractor_a.on_step(action)
    descriptor_a = extractor_a.finalize()

    extractor_b = ActionFrequencyDescriptorExtractor(num_actions=3)
    extractor_b.start_episode()
    for action in actions:
        extractor_b.on_step(action)
    descriptor_b = extractor_b.finalize()

    assert np.array_equal(descriptor_a, descriptor_b)


def test_gridworld_descriptor_deterministic() -> None:
    env = GridworldEnv(GridworldConfig(width=4, height=4, max_steps=5, start=(0, 0)))
    extractor = GridworldDescriptorExtractor(
        width=4, height=4, downsample_width=2, downsample_height=2
    )

    obs, info = env.reset(seed=123)
    extractor.start_episode()

    actions = [3, 3, 1, 1, 2]  # right, right, down, down, left
    for action in actions:
        next_obs, reward, _terminated, truncated, info = env.step(action)
        extractor.on_step(obs, action, reward, next_obs, truncated, info)
        obs = next_obs
        if truncated:
            break

    extractor.end_episode()
    descriptor_one = extractor.finalize()

    # Repeat with the same trajectory to confirm determinism.
    env.reset(seed=123)
    extractor.start_episode()
    obs = np.array([0.0, 0.0], dtype=np.float32)
    for action in actions:
        next_obs, reward, _terminated, truncated, info = env.step(action)
        extractor.on_step(obs, action, reward, next_obs, truncated, info)
        obs = next_obs
        if truncated:
            break
    extractor.end_episode()
    descriptor_two = extractor.finalize()

    assert np.array_equal(descriptor_one, descriptor_two)
