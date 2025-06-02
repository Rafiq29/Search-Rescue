import pytest
import torch
from tensordict import TensorDict

from src.envs.env import SearchAndRescueEnv


def test_env_reset():
    env = SearchAndRescueEnv()
    td = env.reset()
    assert "observation" in td
    assert td["observation"].shape == (env.num_agents, 8)  # Fixed: should be 8 features, not 6


def test_env_step():
    env = SearchAndRescueEnv()
    td = env.reset()
    action = torch.randint(0, 5, (env.num_agents,), dtype=torch.long)
    step_input = TensorDict({"action": action}, batch_size=[])
    next_td = env.step(step_input)
    assert "reward" in next_td
    assert "done" in next_td
    assert next_td["reward"].shape == (env.num_agents,)
    assert next_td["done"].shape == (1,)


def test_env_action_boundaries():
    env = SearchAndRescueEnv()
    td = env.reset()

    # Test all valid actions
    for action_val in range(5):
        action = torch.full((env.num_agents,), action_val, dtype=torch.long)
        step_input = TensorDict({"action": action}, batch_size=[])
        next_td = env.step(step_input)
        assert "reward" in next_td
        assert "done" in next_td


def test_env_specs():
    env = SearchAndRescueEnv()
    # Test that environment has proper specs
    assert env.observation_spec is not None
    assert env.action_spec is not None
    assert env.reward_spec is not None
    assert env.done_spec is not None


def test_env_multiple_steps():
    env = SearchAndRescueEnv(max_steps=10)
    td = env.reset()

    for _ in range(15):  # Test more than max_steps
        action = torch.randint(0, 5, (env.num_agents,), dtype=torch.long)
        step_input = TensorDict({"action": action}, batch_size=[])
        td = env.step(step_input)
        if td["done"].item():
            break

    # Should be done after max_steps
    assert td["done"].item() == True