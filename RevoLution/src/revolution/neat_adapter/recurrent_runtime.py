"""Recurrent runtime for NEAT-derived torch policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .phenotype_torch import ConnectionSpec


@dataclass(frozen=True)
class RecurrentConnectionSpec:
    """Identifies a recurrent connection in the genome."""

    in_key: int
    out_key: int

    def parameter_key(self) -> str:
        return f"{self.in_key}->{self.out_key}"


class TorchRecurrentPolicy(nn.Module):
    """Recurrent policy that uses previous node activations as state.

    We treat all non-input sources as recurrent. That means:
      - inputs come from the current observation
      - non-input nodes use the previous activation snapshot
    This matches a simple Elman-style recurrence and keeps determinism explicit.
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

        # Recurrent state is stored per-node for non-input nodes.
        self._state: dict[int, torch.Tensor] = {}
        self.reset_recurrent_state()

    def reset_recurrent_state(self) -> None:
        """Reset recurrent state to zeros (deterministic)."""

        self._state = {key: torch.tensor(0.0) for key in self._eval_order}

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute action logits using current obs + previous state."""

        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if obs.ndim != 2:
            raise ValueError("obs must be shape (features,) or (batch, features)")

        if obs.shape[1] != len(self._input_keys):
            raise ValueError("obs feature count does not match input_keys")

        # Current activations for this step.
        activations: dict[int, torch.Tensor] = {}
        for idx, key in enumerate(self._input_keys):
            activations[key] = obs[:, idx]

        # Snapshot previous state to avoid in-place feedback in this step.
        prev_state: dict[int, torch.Tensor] = {}
        for key in self._eval_order:
            value = self._state[key].to(obs.device).to(obs.dtype)
            # Expand scalar state to the current batch size.
            if value.ndim == 0:
                value = value.expand(obs.shape[0])
            prev_state[key] = value

        for node_key in self._eval_order:
            incoming = self._incoming.get(node_key, [])
            if not incoming:
                summed = torch.zeros(obs.shape[0], dtype=obs.dtype, device=obs.device)
            else:
                terms = []
                for spec in incoming:
                    weight = self.conn_weight[spec.parameter_key()]
                    if spec.in_key in self._input_keys:
                        source = activations[spec.in_key]
                    else:
                        source = prev_state[spec.in_key]
                    terms.append(source * weight)
                summed = torch.stack(terms, dim=0).sum(dim=0)

            bias = self.node_bias[str(node_key)]
            pre_activation = summed + bias
            activations[node_key] = self._activations[node_key](pre_activation)

        # Update recurrent state at the end of the step.
        for node_key in self._eval_order:
            self._state[node_key] = activations[node_key].detach()

        outputs = torch.stack([activations[key] for key in self._output_keys], dim=1)
        return outputs
