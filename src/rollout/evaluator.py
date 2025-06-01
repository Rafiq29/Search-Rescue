import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
import numpy as np

def evaluate(env: EnvBase, policy, num_episodes=10):
    metrics = {"rescue_success_rate": [], "time_to_complete": [], "collisions": []}
    for _ in range(num_episodes):
        td = env.reset()
        done = False
        steps = 0
        collisions = 0
        rescued = 0
        while not done:
            td = policy(td)
            td = env.step(td)
            rewards = td["reward"].numpy()
            collisions += sum(1 for r in rewards if r < 0)
            rescued += sum(1 for r in rewards if r > 5)  # Assuming rescue reward > 5
            steps += 1
            done = td["done"].all().item()
        metrics["rescue_success_rate"].append(rescued / env.num_victims)
        metrics["time_to_complete"].append(steps)
        metrics["collisions"].append(collisions)
    return {
        "rescue_success_rate": np.mean(metrics["rescue_success_rate"]),
        "time_to_complete": np.mean(metrics["time_to_complete"]),
        "collisions": np.mean(metrics["collisions"])
    }