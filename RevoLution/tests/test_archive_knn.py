import numpy as np

from revolution.novelty.archive import (
    ArchiveCapConfig,
    ArchiveConfig,
    ArchiveThresholdConfig,
    NoveltyArchive,
)
from revolution.novelty.distance import EuclideanDistanceMetric
from revolution.novelty.knn import compute_knn_novelty


def test_knn_novelty_matches_hand_calculation() -> None:
    metric = EuclideanDistanceMetric()
    population = [
        np.array([0.0], dtype=np.float32),
        np.array([2.0], dtype=np.float32),
        np.array([5.0], dtype=np.float32),
    ]

    novelty = compute_knn_novelty(population, archive=[], k=2, metric=metric)

    assert np.allclose(novelty, [3.5, 2.5, 4.0])


def test_knn_novelty_is_stable_for_equal_distances() -> None:
    metric = EuclideanDistanceMetric()
    population = [
        np.array([1.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    ]

    novelty = compute_knn_novelty(population, archive=[], k=1, metric=metric)

    assert novelty == [0.0, 0.0, 0.0]


def test_archive_threshold_adaptation_and_fifo_eviction() -> None:
    config = ArchiveConfig(
        threshold=ArchiveThresholdConfig(
            initial=1.0,
            adapt=True,
            target_add_rate_per_gen=0.5,
            increase_factor=2.0,
            decrease_factor=0.5,
        ),
        cap=ArchiveCapConfig(enabled=True, max_size=2, eviction="fifo"),
    )
    archive = NoveltyArchive(config)

    archive.start_generation()
    archive.consider_add(np.array([0.0], dtype=np.float32), novelty=1.5)
    archive.consider_add(np.array([1.0], dtype=np.float32), novelty=2.0)
    archive.end_generation()

    assert archive.threshold == 2.0

    archive.start_generation()
    archive.consider_add(np.array([2.0], dtype=np.float32), novelty=2.1)
    archive.consider_add(np.array([3.0], dtype=np.float32), novelty=2.2)
    archive.consider_add(np.array([4.0], dtype=np.float32), novelty=2.3)
    archive.end_generation()

    descriptors = archive.query_all()
    assert len(descriptors) == 2
    assert np.allclose(descriptors[0], np.array([3.0], dtype=np.float32))
    assert np.allclose(descriptors[1], np.array([4.0], dtype=np.float32))
