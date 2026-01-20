import pytest

from revolution.seeding import SeedSplitter, derive_seeds
from revolution.utils import stable_sorted


def test_seed_splitter_deterministic_for_same_master_seed() -> None:
    splitter_a = SeedSplitter(12345)
    splitter_b = SeedSplitter(12345)

    seeds_a = splitter_a.spawn(5)
    seeds_b = splitter_b.spawn(5)

    assert seeds_a == seeds_b


def test_seed_splitter_rejects_non_positive_count() -> None:
    splitter = SeedSplitter(1)

    with pytest.raises(ValueError):
        splitter.spawn(0)


def test_derive_seeds_is_label_order_independent() -> None:
    labels_one = ["generation", "torch", "env"]
    labels_two = ["env", "generation", "torch"]

    seeds_one = derive_seeds(999, labels_one)
    seeds_two = derive_seeds(999, labels_two)

    assert seeds_one == seeds_two


def test_stable_sorted_tie_breaks_deterministically() -> None:
    items = {"b", "a"}

    ordered = stable_sorted(items, key=lambda _: 1)

    assert ordered == ["a", "b"]
