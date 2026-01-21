"""Webots environment base class (stub)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym


@dataclass(frozen=True)
class WebotsEnvConfig:
    world: str
    basic_time_step: int
    headless: bool = True
    rendering: bool = False
    reset_mode: str = "supervisor_reset"


class WebotsEnvBase(gym.Env[Any, Any]):
    """Gymnasium-compatible base for Webots tasks.

    This is a stub to keep optional dependency boundaries. Subclasses should
    implement reset/step and ensure determinism by feeding seeds into the
    Webots controller.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: WebotsEnvConfig):
        super().__init__()
        self._config = config

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError("WebotsEnvBase is a stub. Implement in task env.")

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        raise NotImplementedError("WebotsEnvBase is a stub. Implement in task env.")
