"""Line follower Webots task (stub)."""

from __future__ import annotations

from revolution.integrations.webots.webots_env_base import (
    WebotsEnvBase,
    WebotsEnvConfig,
)


class LineFollowerEnv(WebotsEnvBase):
    """Placeholder env for the Webots line follower task."""

    def __init__(self, config: WebotsEnvConfig):
        super().__init__(config)
