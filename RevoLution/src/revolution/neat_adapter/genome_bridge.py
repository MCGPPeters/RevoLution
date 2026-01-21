"""Build torch policies from neat-python genomes (feedforward only)."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Callable

import neat
import torch

from .phenotype_torch import ConnectionSpec, TorchFeedForwardPolicy


def build_feedforward_policy(
    genome: neat.DefaultGenome, neat_config: neat.Config
) -> TorchFeedForwardPolicy:
    """Construct a torch policy from a neat-python genome.

    The returned policy has parameters that map 1:1 to genome biases/weights.
    """

    genome_config = neat_config.genome_config
    if not genome_config.feed_forward:
        raise ValueError("Only feed-forward genomes are supported in Phase 4.")

    input_keys = list(genome_config.input_keys)
    output_keys = list(genome_config.output_keys)

    enabled_connections = [
        (key, conn)
        for key, conn in genome.connections.items()
        if conn.enabled
    ]

    # Build incoming connection lists for each node.
    incoming: dict[int, list[ConnectionSpec]] = defaultdict(list)
    for (in_key, out_key), _conn in enabled_connections:
        incoming[out_key].append(ConnectionSpec(in_key, out_key))

    eval_order = _topological_order(input_keys, output_keys, enabled_connections)

    activations = _build_activation_map(genome)
    biases = {node_key: node.bias for node_key, node in genome.nodes.items()}
    weights = {
        ConnectionSpec(in_key, out_key): conn.weight
        for (in_key, out_key), conn in enabled_connections
    }

    return TorchFeedForwardPolicy(
        input_keys=input_keys,
        output_keys=output_keys,
        eval_order=eval_order,
        incoming=incoming,
        activations=activations,
        biases=biases,
        weights=weights,
    )


def _topological_order(
    input_keys: list[int],
    output_keys: list[int],
    enabled_connections: list[tuple[tuple[int, int], Any]],
) -> list[int]:
    """Return a deterministic topological order for non-input nodes."""

    nodes = set(output_keys)
    for (in_key, out_key), _conn in enabled_connections:
        nodes.add(in_key)
        nodes.add(out_key)

    all_nodes = sorted(nodes)
    in_degree = {node: 0 for node in all_nodes}
    adjacency: dict[int, list[int]] = defaultdict(list)

    for (in_key, out_key), _conn in enabled_connections:
        adjacency[in_key].append(out_key)
        if out_key in in_degree:
            in_degree[out_key] += 1

    queue = deque(sorted(node for node, deg in in_degree.items() if deg == 0))
    order: list[int] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in sorted(adjacency.get(node, [])):
            if neighbor not in in_degree:
                continue
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(all_nodes):
        raise ValueError("Genome contains a cycle or disconnected nodes.")

    return [node for node in order if node not in input_keys]


def _build_activation_map(
    genome: neat.DefaultGenome,
) -> dict[int, Callable[[torch.Tensor], torch.Tensor]]:
    """Map NEAT activation names to torch functions."""

    def identity(x: torch.Tensor) -> torch.Tensor:
        return x

    activation_map: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "relu": torch.relu,
        "identity": identity,
        "linear": identity,
    }

    activations: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
    for node_key, node in genome.nodes.items():
        activation_name = getattr(node, "activation", "sigmoid")
        if activation_name not in activation_map:
            raise ValueError(f"Unsupported activation: {activation_name}")
        activations[node_key] = activation_map[activation_name]

    return activations
