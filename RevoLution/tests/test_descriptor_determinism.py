import numpy as np

from revolution.envs.descriptors import ActionFrequencyDescriptorExtractor


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
