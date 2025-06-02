import os
import importlib
import torch
import hydra
from omegaconf import DictConfig
from envs.env import SearchAndRescueEnv
from agents.ppo_agent import PPOAgent

# Debug the loaded env.py path
env_module = importlib.import_module("envs.env")
print(f"Loaded env.py from: {os.path.abspath(env_module.__file__)}")

@hydra.main(config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    # Extract configurations
    env_params = cfg.env
    agent_params = cfg.agent
    algo_params = cfg.algo.algo
    experiment_params = cfg.experiment.experiment

    # Set random seed
    torch.manual_seed(experiment_params.seed)
    np.random.seed(experiment_params.seed)

    # Initialize environment
    env = SearchAndRescueEnv(**env_params)
    print("Environment initialized with", env_params.num_agents, "agents and", env_params.num_victims, "victims")

    # Debug print of action spec
    print(f"Action spec in train.py: {env.action_spec}")

    # Initialize agent
    try:
        agent = PPOAgent(env, **agent_params)
    except Exception as e:
        print(f"Error in training function: {e}")
        raise

    # Training loop (simplified for debugging)
    total_frames = algo_params.total_frames
    frames_per_batch = algo_params.frames_per_batch
    num_steps = total_frames // frames_per_batch

    print("Starting training loop...")
    for step in range(num_steps):
        # Simplified loop to confirm execution
        obs = env.reset()
        action_tensordict = agent(obs)
        print(f"Step {step}, Observation shape: {obs['observation'].shape}")
        break  # Break after one step for debugging

if __name__ == "__main__":
    train()