"""Optional Webots integration (stub).

This package provides interfaces for Webots-based environments but does not
import Webots at module import time. Users must install Webots separately.
"""

from .installation import INSTALLATION_TEXT
from .webots_env_base import WebotsEnvBase, WebotsEnvConfig

__all__ = ["INSTALLATION_TEXT", "WebotsEnvBase", "WebotsEnvConfig"]
