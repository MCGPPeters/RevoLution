"""Small deterministic helpers.

We keep ordering helpers here to avoid nondeterministic iteration (dict/set).
"""

from __future__ import annotations

from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


def stable_sorted(
    items: Iterable[T],
    key: Callable[[T], object] | None = None,
) -> list[T]:
    """Return a deterministically ordered list from any iterable.

    If multiple items compare equal under `key`, we apply a stable tie-breaker
    using `repr(item)` so that the output is deterministic even for set inputs.
    """

    item_list = list(items)
    if key is None:
        return sorted(item_list, key=lambda item: repr(item))

    return sorted(item_list, key=lambda item: (key(item), repr(item)))
