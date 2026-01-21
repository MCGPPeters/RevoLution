"""Bridge between neat-python genomes and torch policies."""

from .genome_bridge import build_feedforward_policy
from .phenotype_torch import TorchFeedForwardPolicy

__all__ = ["TorchFeedForwardPolicy", "build_feedforward_policy"]
