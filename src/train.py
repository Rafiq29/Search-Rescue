import hydra
from omegaconf import DictConfig
from envs.env import SearchAndRescueEnv
import torch
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from agents.ppo_agent import PPOAgent
from torchrl.objectives import ClipPPOLoss
from torch.utils.tensorboard import SummaryWriter
import imageio
import numpy as np
from tensordict import TensorDict
import os

@hydra.main(config_path="../configs", config_name="base", version_base="1.2")
def train(cfg: DictConfig) -> None:
    # Ensure output directories exist
    os.makedirs(cfg.experiment.experiment.log_dir, exist_ok=True)
    os.makedirs(cfg.experiment.experiment.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cpu")  # Use CPU for consistency; change to "cuda" if GPU is available

    # Initialize logging
    writer = SummaryWriter(log_dir=cfg.experiment.experiment.log_dir)

    # Print configuration
    print(f"Configuration: {cfg}")

    # Set seed for reproducibility
    torch.manual_seed(cfg.experiment.experiment.seed)
    np.random.seed(cfg.experiment.experiment.seed)

    # Extract environment parameters
    env_params = {
        "map_size": cfg.env.map_size,
        "num_agents": cfg.env.num_agents,
        "num_victims": cfg.env.num_victims,
        "max_steps": cfg.env.max_steps,
        "obstacle_density": cfg.env.obstacle_density,
        "terrain_difficulty": cfg.env.get("terrain_difficulty", 0.1),
        "device": device
    }

    # Initialize environment
    env = SearchAndRescueEnv(**env_params)

    # Initialize PPO agent
    agent_params = {
        "learning_rate": cfg.agent.learning_rate,
        "gamma": cfg.agent.gamma,
        "clip_eps": cfg.algo.algo.clip_epsilon
    }
    agent = PPOAgent(env, **agent_params)

    # Move policy and critic to device
    agent.policy.to(device)
    agent.critic.to(device)

    # Debug: Test policy output
    obs = env.reset()
    print(f"Observation shape: {obs['observation'].shape}, Batch size: {obs.batch_size}")
    action = agent.policy(obs)  # Pass the entire TensorDict
    print(f"Action shape: {action.shape}")

    # Set up loss module (PPO-Clip)
    loss_module = ClipPPOLoss(
        actor=agent.policy,
        critic=agent.critic,
        clip_epsilon=cfg.algo.algo.clip_epsilon,
        entropy_coef=cfg.algo.algo.entropy_coef
    )
    loss_module.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        list(agent.policy.parameters()) + list(agent.critic.parameters()),
        lr=cfg.algo.algo.lr
    )

    # Set up data collector
    collector = SyncDataCollector(
        env,
        agent.policy,
        frames_per_batch=cfg.algo.algo.frames_per_batch,
        total_frames=cfg.algo.algo.total_frames,
        device=device
    )

    # Set up replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.algo.algo.frames_per_batch),
        batch_size=32
    )

    # Metrics tracking
    total_rewards = []
    rescue_success_rate = 0
    collisions = 0
    total_loss = torch.tensor(0.0, device=device)  # Initialize to avoid unassigned variable warning

    # Training loop
    for i, data in enumerate(collector):
        # Collect data
        replay_buffer.extend(data)

        # Sample batch and train
        for _ in range(cfg.algo.algo.ppo_epochs):
            batch = replay_buffer.sample()
            batch = batch.to(device)
            loss_vals = loss_module(batch)
            total_loss = loss_vals["loss_actor"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Log metrics
        mean_reward = data["reward"].mean().item()
        total_rewards.append(mean_reward)
        writer.add_scalar("reward/mean", mean_reward, i)

        # Update metrics
        num_successful_rescues = 0
        for _ in range(10):
            obs = env.reset()
            done = torch.zeros(1, dtype=torch.bool, device=device)
            while not done.item():  # Check global done
                action = agent.policy(obs)
                next_state = env.step(TensorDict({"action": action}, batch_size=[]))
                obs = next_state["observation"]
                done = next_state["done"]
            num_successful_rescues += np.sum(env.victims_rescued)
        rescue_success_rate = num_successful_rescues / (10 * env.num_victims)
        collisions += data["reward"].lt(0).sum().item()
        writer.add_scalar("metrics/rescue_success_rate", rescue_success_rate, i)
        writer.add_scalar("metrics/collisions", collisions, i)

        print(f"Step {i}, Loss: {total_loss.item()}, Mean Reward: {mean_reward}")

    # Save GIF
    frames = []
    env.reset()
    for _ in range(100):
        obs = TensorDict({"observation": env._get_obs()}, batch_size=[])
        action = agent.policy(obs)
        next_state = env.step(TensorDict({"action": action}, batch_size=[]))
        frame = np.zeros((env.map_size * 10, env.map_size * 10, 3), dtype=np.uint8)
        for sz_idx, sz in enumerate(env.safe_zones):
            x, y = int(sz[0] * 10), int(sz[1] * 10)
            if sz_idx == 0: frame[x:x+10, y:y+10] = [255, 0, 0]
            elif sz_idx == 1: frame[x:x+10, y:y+10] = [0, 255, 0]
            elif sz_idx == 2: frame[x:x+10, y:y+10] = [0, 0, 255]
            elif sz_idx == 3: frame[x:x+10, y:y+10] = [255, 255, 0]
        for v_idx, v in enumerate(env.victim_pos):
            x, y = int(v[0] * 10), int(v[1] * 10)
            frame[x:x+10, y:y+10] = [0, 0, 0]
        frames.append(frame)
    imageio.mimsave(f"{cfg.experiment.experiment.output_dir}/training.gif", frames, duration=0.1)

    print("Training completed!")
    writer.close()

if __name__ == "__main__":
    train()