import os
import sys
import torch
import numpy as np
import hydra
from omegaconf import DictConfig

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from envs.env import SearchAndRescueEnv
from agents.ppo_agent import PPOAgent
from rollout.evaluator import evaluate
from rollout.visualizer import render_episode


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def evaluate_model(cfg: DictConfig) -> None:
    """Evaluate a trained model"""

    # Extract configurations
    env_params = cfg.env
    agent_params = cfg.agent
    experiment_params = cfg.experiment

    # Set random seed
    seed = experiment_params.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize environment
    env = SearchAndRescueEnv(
        map_size=env_params.map_size,
        num_agents=env_params.num_agents,
        num_victims=env_params.num_victims,
        max_steps=env_params.max_steps,
        obstacle_density=env_params.obstacle_density
    )

    # Initialize agent
    agent = PPOAgent(
        env,
        learning_rate=agent_params.learning_rate,
        gamma=agent_params.gamma
    )

    # Load trained model if available
    model_path = f"models/ppo/policy_seed_{seed}.pt"
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        agent.policy.load_state_dict(torch.load(model_path, map_location="cpu"))
        agent.policy.eval()
    else:
        print(f"No trained model found at {model_path}, using random policy")

    # Evaluate the agent
    print("Evaluating agent...")
    metrics = evaluate(env, agent, num_episodes=10)

    print("\nEvaluation Results:")
    print(f"Rescue Success Rate: {metrics['rescue_success_rate']:.3f}")
    print(f"Average Time to Complete: {metrics['time_to_complete']:.1f} steps")
    print(f"Average Collisions: {metrics['collisions']:.1f}")

    # Generate visualization
    print("\nGenerating visualization...")
    output_dir = experiment_params.get('output_dir', 'outputs/')
    os.makedirs(f"{output_dir}/gifs", exist_ok=True)

    try:
        render_episode(env, agent, f"{output_dir}/gifs/evaluation_episode.gif")
        print(f"Visualization saved to {output_dir}/gifs/evaluation_episode.gif")
    except Exception as e:
        print(f"Visualization failed: {e}")


if __name__ == "__main__":
    evaluate_model()