import math
from pathlib import Path

import neat
import torch
from neat.innovation import InnovationTracker

from revolution.neat_adapter.genome_bridge import build_feedforward_policy
from revolution.neat_adapter.phenotype_torch import ConnectionSpec


def _load_neat_config() -> neat.Config:
    cfg_path = Path(__file__).parent / "fixtures" / "neat_test.cfg"
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(cfg_path),
    )


def test_feedforward_policy_matches_manual_sigmoid() -> None:
    neat_config = _load_neat_config()
    neat_config.genome_config.innovation_tracker = InnovationTracker()
    genome = neat.DefaultGenome(0)
    genome.configure_new(neat_config.genome_config)

    input_keys = list(neat_config.genome_config.input_keys)
    output_keys = list(neat_config.genome_config.output_keys)

    # Zero all biases for predictable outputs.
    for node in genome.nodes.values():
        node.bias = 0.0

    # Set weights to route input[0] -> output[0], input[1] -> output[1].
    for (in_key, out_key), conn in genome.connections.items():
        if in_key == input_keys[0] and out_key == output_keys[0]:
            conn.weight = 1.0
        elif in_key == input_keys[1] and out_key == output_keys[1]:
            conn.weight = 1.0
        else:
            conn.weight = 0.0

    policy = build_feedforward_policy(genome, neat_config)

    obs = torch.tensor([1.0, -1.0], dtype=torch.float32)
    output = policy(obs).squeeze(0)

    expected = torch.tensor(
        [1.0 / (1.0 + math.exp(-1.0)), 1.0 / (1.0 + math.exp(1.0))],
        dtype=torch.float32,
    )
    assert torch.allclose(output, expected, atol=1e-5)

    # Determinism check: repeated forwards should match exactly.
    output_again = policy(obs).squeeze(0)
    assert torch.equal(output, output_again)


def test_policy_parameter_roundtrip() -> None:
    neat_config = _load_neat_config()
    neat_config.genome_config.innovation_tracker = InnovationTracker()
    genome = neat.DefaultGenome(1)
    genome.configure_new(neat_config.genome_config)

    policy = build_feedforward_policy(genome, neat_config)

    input_keys = list(neat_config.genome_config.input_keys)
    output_keys = list(neat_config.genome_config.output_keys)
    target_conn = ConnectionSpec(input_keys[0], output_keys[0]).parameter_key()

    with torch.no_grad():
        policy.conn_weight[target_conn].copy_(torch.tensor(0.25))
        policy.node_bias[str(output_keys[0])].copy_(torch.tensor(0.5))

    policy.apply_to_genome(genome)

    assert genome.connections[(input_keys[0], output_keys[0])].weight == 0.25
    assert genome.nodes[output_keys[0]].bias == 0.5
