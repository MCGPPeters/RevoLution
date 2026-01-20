"""Deterministic seed utilities.

We avoid global RNG state by generating all derived seeds from a master seed.
This mirrors the idea of explicit RNG streams in C# or other languages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SeedSplitter:
    """Deterministically splits a master seed into child seeds.

    We rely on numpy's SeedSequence to provide reproducible seed spawning.
    The class is immutable (frozen) to prevent accidental mutation.

    Typical usage mirrors creating independent Random instances in C#:
        splitter = SeedSplitter(1234)
        seeds = splitter.spawn(3)
    """

    master_seed: int

    def spawn(self, count: int) -> list[int]:
        """Return `count` deterministic child seeds.

        We explicitly generate 32-bit unsigned values because most RNGs in
        Python ecosystems accept 32-bit seeds, similar to System.Random.
        """

        if count <= 0:
            raise ValueError("count must be positive")

        # SeedSequence is pure; spawning is deterministic for the same master seed.
        root = np.random.SeedSequence(self.master_seed)
        children = root.spawn(count)

        # generate_state returns uint32 array; convert to Python int for portability.
        return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]


def derive_seeds(master_seed: int, labels: Iterable[str]) -> dict[str, int]:
    """Derive named seeds from a master seed in a stable, deterministic order.

    We sort the labels to prevent accidental nondeterminism from set iteration.
    This is analogous to ordering keys before generating RNG streams in C#.
    """

    label_list = sorted(labels)
    splitter = SeedSplitter(master_seed)
    child_seeds = splitter.spawn(len(label_list))
    return dict(zip(label_list, child_seeds, strict=True))


def derive_seed(master_seed: int, *tokens: int) -> int:
    """Derive a single deterministic seed from a master seed + tokens.

    This is useful when you need per-episode or per-genome seeds that are
    stable across runs. The tokens should be integers (e.g., generation id,
    genome id, episode index).
    """

    seed_sequence = np.random.SeedSequence([master_seed, *tokens])
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
