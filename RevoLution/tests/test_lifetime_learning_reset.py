import torch

from revolution.rl import NoLearningRule, ReinforceLearner
from revolution.rl.reinforce import ReinforceConfig


def _policy() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Linear(2, 2, bias=True)


def _step_policy(policy: torch.nn.Module) -> torch.Tensor:
    obs = torch.tensor([1.0, -1.0], dtype=torch.float32)
    logits = policy(obs)
    dist = torch.distributions.Categorical(logits=logits)
    return dist.log_prob(torch.tensor(0))


def test_params_change_within_lifetime_and_reset() -> None:
    policy = _policy()
    initial = {k: v.detach().clone() for k, v in policy.state_dict().items()}

    learner = ReinforceLearner(ReinforceConfig(lr=0.1, gamma=1.0))
    learner.initialize(policy, rng_seed=123)
    learner.start_episode()
    for _ in range(3):
        log_prob = _step_policy(policy)
        learner.on_step(log_prob, reward=1.0)
    diagnostics = learner.end_episode()

    assert diagnostics["grad_norm"] > 0.0
    changed = any(
        not torch.equal(param, initial[name])
        for name, param in policy.state_dict().items()
    )
    assert changed

    learner.reset_to_initial(policy)
    for name, param in policy.state_dict().items():
        assert torch.equal(param, initial[name])


def test_no_learning_rule_keeps_params_fixed() -> None:
    policy = _policy()
    initial = {k: v.detach().clone() for k, v in policy.state_dict().items()}

    learner = NoLearningRule()
    learner.initialize(policy, rng_seed=321)
    learner.start_episode()
    for _ in range(2):
        log_prob = _step_policy(policy)
        learner.on_step(log_prob, reward=1.0)
    learner.end_episode()

    for name, param in policy.state_dict().items():
        assert torch.equal(param, initial[name])
