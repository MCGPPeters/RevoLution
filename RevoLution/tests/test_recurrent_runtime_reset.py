import torch

from revolution.neat_adapter.phenotype_torch import ConnectionSpec
from revolution.neat_adapter.recurrent_runtime import TorchRecurrentPolicy


def test_recurrent_state_reset_is_deterministic() -> None:
    input_keys = [-1]
    output_keys = [0]
    eval_order = [0]

    incoming = {
        0: [
            ConnectionSpec(-1, 0),
            ConnectionSpec(0, 0),
        ]
    }
    activations = {0: torch.sigmoid}
    biases = {0: 0.0}
    weights = {ConnectionSpec(-1, 0): 1.0, ConnectionSpec(0, 0): 1.0}

    policy = TorchRecurrentPolicy(
        input_keys=input_keys,
        output_keys=output_keys,
        eval_order=eval_order,
        incoming=incoming,
        activations=activations,
        biases=biases,
        weights=weights,
    )

    obs = torch.tensor([1.0], dtype=torch.float32)
    out_one = policy(obs)
    out_two = policy(obs)

    # Without reset, recurrence should make the output change.
    assert not torch.equal(out_one, out_two)

    policy.reset_recurrent_state()
    out_reset = policy(obs)
    assert torch.equal(out_one, out_reset)
