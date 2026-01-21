"""Torch phenotype for NEAT genomes (feedforward only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class ConnectionSpec:
    """Identifies a connection in the genome."""

    in_key: int
    out_key: int

    def parameter_key(self) -> str:
        return f"{self.in_key}->{self.out_key}"


class TorchFeedForwardPolicy(nn.Module):
    """Feedforward policy whose parameters map 1:1 to a NEAT genome.

    We implement a direct, explicit forward pass:
      - Inputs are copied from the observation.
      - Each non-input node sums incoming weighted activations + bias.
      - Node activation is applied using NEAT's configured function.

    This mirrors how neat-python evaluates feedforward graphs, but exposes the
    weights/biases as trainable torch Parameters for Phase 4+.
    """

    def __init__(
        self,
        input_keys: list[int],
        output_keys: list[int],
        eval_order: list[int],
        incoming: dict[int, list[ConnectionSpec]],
        activations: dict[int, Callable[[torch.Tensor], torch.Tensor]],
        biases: dict[int, float],
        weights: dict[ConnectionSpec, float],
    ) -> None:
        super().__init__()
        self._input_keys = input_keys
        self._output_keys = output_keys
        self._eval_order = eval_order
        self._incoming = incoming
        self._activations = activations

        self.node_bias = nn.ParameterDict(
            {
                str(node_key): nn.Parameter(torch.tensor(bias, dtype=torch.float32))
                for node_key, bias in biases.items()
            }
        )
        self.conn_weight = nn.ParameterDict(
            {
                spec.parameter_key(): nn.Parameter(
                    torch.tensor(weight, dtype=torch.float32)
                )
                for spec, weight in weights.items()
            }
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute action logits for a single observation or batch."""

        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if obs.ndim != 2:
            raise ValueError("obs must be shape (features,) or (batch, features)")

        if obs.shape[1] != len(self._input_keys):
            raise ValueError("obs feature count does not match input_keys")

        # Activation cache: node_key -> tensor(batch,)
        activations: dict[int, torch.Tensor] = {}
        for idx, key in enumerate(self._input_keys):
            activations[key] = obs[:, idx]

        for node_key in self._eval_order:
            incoming = self._incoming.get(node_key, [])
            if not incoming:
                summed = torch.zeros(obs.shape[0], dtype=obs.dtype, device=obs.device)
            else:
                terms = []
                for spec in incoming:
                    weight = self.conn_weight[spec.parameter_key()]
                    terms.append(activations[spec.in_key] * weight)
                summed = torch.stack(terms, dim=0).sum(dim=0)

            bias = self.node_bias[str(node_key)]
            pre_activation = summed + bias
            activations[node_key] = self._activations[node_key](pre_activation)

        outputs = torch.stack([activations[key] for key in self._output_keys], dim=1)
        return outputs

    def apply_to_genome(self, genome) -> None:  # type: ignore[no-untyped-def]
        """Write current torch parameters back into a neat-python genome."""

        for (in_key, out_key), conn in genome.connections.items():
            key = ConnectionSpec(in_key, out_key).parameter_key()
            if key in self.conn_weight:
                conn.weight = float(self.conn_weight[key].detach().cpu().item())

        for node_key, node in genome.nodes.items():
            bias_key = str(node_key)
            if bias_key in self.node_bias:
                node.bias = float(self.node_bias[bias_key].detach().cpu().item())

    def load_from_genome(self, genome) -> None:  # type: ignore[no-untyped-def]
        """Overwrite torch parameters with genome values."""

        for (in_key, out_key), conn in genome.connections.items():
            key = ConnectionSpec(in_key, out_key).parameter_key()
            if key in self.conn_weight:
                with torch.no_grad():
                    self.conn_weight[key].copy_(torch.tensor(conn.weight))

        for node_key, node in genome.nodes.items():
            bias_key = str(node_key)
            if bias_key in self.node_bias:
                with torch.no_grad():
                    self.node_bias[bias_key].copy_(torch.tensor(node.bias))

    @property
    def input_keys(self) -> list[int]:
        return list(self._input_keys)

    @property
    def output_keys(self) -> list[int]:
        return list(self._output_keys)

    def connection_specs(self) -> Iterable[ConnectionSpec]:
        specs: list[ConnectionSpec] = []
        for items in self._incoming.values():
            specs.extend(items)
        return specs
