import os
import sys
import importlib
import torch
import numpy as np
import hydra
from omegaconf import DictConfig
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torch.utils.tensorboard import SummaryWriter
import wandb

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from envs.env import SearchAndRescueEnv
from agents.ppo_agent import PPOAgent

# Debug the loaded env.py path
try:
    env_module = importlib.import_module("envs.env")
    print(f"Loaded env.py from: {os.path.abspath(env_module.__file__)}")
except ImportError as e:
    print(f"Import error: {e}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    print("Starting training with configuration:")
    print(cfg)

    # Extract configurations
    env_params = cfg.env
    agent_params = cfg.agent
    algo_params = cfg.algo
    experiment_params = cfg.experiment

    # Set random seed
    seed = experiment_params.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize logging
    log_dir = experiment_params.get('log_dir', 'logs/')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Initialize WandB (optional)
    try:
        wandb.init(
            project="search-and-rescue-marl",
            config=dict(cfg),
            name=f"run_{seed}"
        )
        use_wandb = True
    except Exception as e:
        print(f"Warning: WandB initialization failed: {e}")
        use_wandb = False

    # Create environment function for collector
    def env_fn():
        return SearchAndRescueEnv(
            map_size=env_params.map_size,
            num_agents=env_params.num_agents,
            num_victims=env_params.num_victims,
            max_steps=env_params.max_steps,
            obstacle_density=env_params.obstacle_density
        )

    # Initialize environment for spec checking
    env = env_fn()
    print(f"Environment initialized with {env_params.num_agents} agents and {env_params.num_victims} victims")
    print(f"Action spec: {env.action_spec}")
    print(f"Observation spec: {env.observation_spec}")

    # Initialize agent
    try:
        agent = PPOAgent(
            env,
            learning_rate=agent_params.learning_rate,
            gamma=agent_params.gamma,
            clip_eps=algo_params.clip_epsilon
        )
        print("Agent initialized successfully")
    except Exception as e:
        print(f"Error initializing agent: {e}")
        raise

    # Create data collector
    collector = SyncDataCollector(
        create_env_fn=env_fn,
        policy=agent,
        frames_per_batch=algo_params.frames_per_batch,
        total_frames=algo_params.total_frames,
        device="cpu"
    )

    # Create replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=algo_params.frames_per_batch),
        batch_size=256  # Mini-batch size for PPO updates
    )

    # Create PPO loss
    loss_module = ClipPPOLoss(
        actor_network=agent.policy,
        critic_network=agent.critic,
        clip_epsilon=algo_params.clip_epsilon,
        entropy_coef=algo_params.entropy_coef,
        normalize_advantage=True
    )

    # Training loop
    total_frames = algo_params.total_frames
    frames_per_batch = algo_params.frames_per_batch
    ppo_epochs = algo_params.ppo_epochs

    print(f"Starting training for {total_frames} total frames...")

    step = 0
    for batch in collector:
        print(f"Collected batch {step}, batch size: {batch.batch_size}")

        # Add batch to replay buffer
        replay_buffer.extend(batch)

        # PPO updates
        for epoch in range(ppo_epochs):
            # Sample from replay buffer
            sampled_batch = replay_buffer.sample()

            # Compute loss
            loss_dict = loss_module(sampled_batch)
            total_loss = loss_dict["loss_objective"] + loss_dict["loss_critic"] + loss_dict["loss_entropy"]

            # Backward pass
            agent.actor_optimizer.zero_grad()
            agent.critic_optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_norm=0.5)

            agent.actor_optimizer.step()
            agent.critic_optimizer.step()

        # Logging
        if step % 10 == 0:
            mean_reward = batch["reward"].mean().item()
            print(f"Step {step}: Mean reward = {mean_reward:.4f}")

            # TensorBoard logging
            writer.add_scalar("reward/mean", mean_reward, step)
            writer.add_scalar("loss/total", total_loss.item(), step)

            # WandB logging
            if use_wandb:
                wandb.log({
                    "reward/mean": mean_reward,
                    "loss/total": total_loss.item(),
                    "step": step
                })

        step += 1

        # Check if we've collected enough frames
        if collector.total_frames >= total_frames:
            break

    # Save final model
    model_dir = "models/ppo/"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(agent.policy.state_dict(), f"{model_dir}/policy_seed_{seed}.pt")
    torch.save(agent.critic.state_dict(), f"{model_dir}/critic_seed_{seed}.pt")

    print("Training completed!")

    # Cleanup
    writer.close()
    if use_wandb:
        wandb.finish()
    collector.shutdown()


if __name__ == "__main__":
    train()