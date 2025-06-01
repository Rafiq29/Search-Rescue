import pytest
import torch

from src.envs.env import SearchAndRescueEnv

def test_env_reset():
    env = SearchAndRescueEnv()
    td = env.reset()
    assert "observation" in td
    assert td["observation"].shape == (env.num_agents, 6)

def test_env_step():
    env = SearchAndRescueEnv()
    td = env.reset()
    td["action"] = torch.zeros(env.num_agents, dtype=torch.int32)
    next_td = env.step(td)
    assert "reward" in next_td
    assert "done" in next_td