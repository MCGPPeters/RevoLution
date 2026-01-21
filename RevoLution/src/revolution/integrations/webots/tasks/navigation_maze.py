"""Navigation maze Webots task (stub)."""

from __future__ import annotations

from revolution.integrations.webots.webots_env_base import (
    WebotsEnvBase,
    WebotsEnvConfig,
)


class NavigationMazeEnv(WebotsEnvBase):
    """Placeholder env for the Webots navigation maze task."""

    def __init__(self, config: WebotsEnvConfig):
        super().__init__(config)
