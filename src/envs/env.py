import numpy as np
import torch
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs
from tensordict import TensorDict
from torchrl.data import Unbounded, Composite, Categorical

class SearchAndRescueEnv(EnvBase):
    def __init__(self, map_size=10, num_agents=5, num_victims=3, max_steps=100, obstacle_density=0.2, terrain_difficulty=0.1, device="cpu"):
        super().__init__(device=device)
        self.map_size = map_size
        self.num_agents = num_agents
        self.num_victims = num_victims
        self.max_steps = max_steps
        self.obstacle_density = obstacle_density
        self.terrain_difficulty = terrain_difficulty

        # Define observation and action specs
        self.observation_spec = Composite(
            observation=Unbounded(
                shape=torch.Size([num_agents, 8]),  # 8 features: agent pos (2), nearest victim (2), nearest safe zone (2), has victim (1), step count (1)
                device=self.device
            ),
            shape=()
        )
        self.action_spec = Composite(
            action=Categorical(
                n=5,  # 5 actions: up, down, left, right, rescue
                shape=torch.Size([num_agents]),
                device=self.device
            ),
            shape=()
        )
        self.reward_spec = Unbounded(
            shape=torch.Size([num_agents]),
            device=self.device
        )
        self.done_spec = Unbounded(
            shape=torch.Size([1]),  # Global done signal
            device=self.device,
            dtype=torch.bool
        )

        # Initialize environment
        self.grid = self._generate_grid()
        self.agent_pos = self._place_agents()
        self.victim_pos = self._place_victims()
        self.safe_zones = self._place_safe_zones()
        self.victims_rescued = np.zeros(self.num_victims, dtype=np.bool)
        self.step_count = 0

    def _generate_grid(self):
        grid = np.zeros((self.map_size, self.map_size))
        num_obstacles = int(self.map_size * self.map_size * self.obstacle_density)
        obstacle_indices = np.random.choice(self.map_size * self.map_size, num_obstacles, replace=False)
        for idx in obstacle_indices:
            x, y = divmod(idx, self.map_size)
            grid[x, y] = 1  # 1 represents an obstacle
        return grid

    def _place_agents(self):
        positions = []
        for _ in range(self.num_agents):
            while True:
                pos = np.random.randint(0, self.map_size, 2)
                if self.grid[pos[0], pos[1]] == 0 and not any((pos == p).all() for p in positions):
                    positions.append(pos)
                    break
        return np.array(positions)

    def _place_victims(self):
        positions = []
        for _ in range(self.num_victims):
            while True:
                pos = np.random.randint(0, self.map_size, 2)
                if self.grid[pos[0], pos[1]] == 0 and not any((pos == p).all() for p in positions + list(self.agent_pos)):
                    positions.append(pos)
                    break
        return np.array(positions)

    def _place_safe_zones(self):
        safe_zones = []
        for _ in range(4):  # 4 safe zones
            while True:
                pos = np.random.randint(0, self.map_size, 2)
                if self.grid[pos[0], pos[1]] == 0 and not any((pos == p).all() for p in safe_zones + list(self.agent_pos) + list(self.victim_pos)):
                    safe_zones.append(pos)
                    break
        return np.array(safe_zones)

    def _get_victim_at(self, pos):
        for idx, victim in enumerate(self.victim_pos):
            if (victim == pos).all():
                return idx
        return None

    def _get_safe_zone_at(self, pos):
        for idx, safe_zone in enumerate(self.safe_zones):
            if (safe_zone == pos).all():
                return idx
        return None

    def _get_obs(self):
        obs = np.zeros((self.num_agents, 8), dtype=np.float32)  # 8 features
        for agent_idx in range(self.num_agents):
            pos = self.agent_pos[agent_idx]
            # Agent position
            obs[agent_idx, 0:2] = pos / self.map_size  # Normalize position
            # Nearest victim
            alive_victims = [v for i, v in enumerate(self.victim_pos) if not self.victims_rescued[i]]
            if alive_victims:
                nearest_victim = min(alive_victims, key=lambda v: np.linalg.norm(v - pos))
                obs[agent_idx, 2:4] = (nearest_victim / self.map_size).astype(np.float32)
            else:
                obs[agent_idx, 2:4] = (pos / self.map_size).astype(np.float32)
            # Nearest safe zone
            nearest_safe_zone = min(self.safe_zones, key=lambda sz: np.linalg.norm(sz - pos))
            obs[agent_idx, 4:6] = (nearest_safe_zone / self.map_size).astype(np.float32)
            # Has victim nearby
            victim_idx = self._get_victim_at(pos)
            obs[agent_idx, 6] = 1.0 if victim_idx is not None and not self.victims_rescued[victim_idx] else 0.0
            # Step count
            obs[agent_idx, 7] = self.step_count / self.max_steps
        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    def _set_seed(self, seed):
        """Set the seed for the environment's random number generators."""
        np.random.seed(seed)
        # Reset the environment to ensure the new seed takes effect
        self.grid = self._generate_grid()
        self.agent_pos = self._place_agents()
        self.victim_pos = self._place_victims()
        self.safe_zones = self._place_safe_zones()
        self.victims_rescued = np.zeros(self.num_victims, dtype=np.bool)
        self.step_count = 0
        return seed

    def _reset(self, tensordict=None, **kwargs):
        self.grid = self._generate_grid()
        self.agent_pos = self._place_agents()
        self.victim_pos = self._place_victims()
        self.safe_zones = self._place_safe_zones()
        self.victims_rescued = np.zeros(self.num_victims, dtype=np.bool)
        self.step_count = 0
        obs = self._get_obs()
        return TensorDict(
            {
                "observation": obs,
                "done": torch.tensor([False], dtype=torch.bool, device=self.device),  # Global done
                "reward": torch.zeros(self.num_agents, dtype=torch.float32, device=self.device)
            },
            batch_size=[]  # No batch dimension for the environment
        )

    def _step(self, tensordict):
        actions = tensordict["action"].numpy()
        rewards = np.zeros(self.num_agents, dtype=np.float32)

        for agent_idx in range(self.num_agents):
            action = actions[agent_idx]
            agent_pos = self.agent_pos[agent_idx]
            new_pos = agent_pos.copy()

            if action == 0:  # Up
                new_pos[1] -= 1
            elif action == 1:  # Down
                new_pos[1] += 1
            elif action == 2:  # Left
                new_pos[0] -= 1
            elif action == 3:  # Right
                new_pos[0] += 1
            elif action == 4:  # Rescue
                victim_idx = self._get_victim_at(agent_pos)
                if victim_idx is not None and not self.victims_rescued[victim_idx]:
                    self.victims_rescued[victim_idx] = True
                    rewards[agent_idx] = 10.0  # Reward for rescuing a victim
                    safe_zone_idx = self._get_safe_zone_at(agent_pos)
                    if safe_zone_idx is not None:
                        rewards[agent_idx] += 5.0  # Additional reward for being in a safe zone

            # Check boundaries and obstacles
            if not (0 <= new_pos[0] < self.map_size and 0 <= new_pos[1] < self.map_size and
                    self.grid[new_pos[0], new_pos[1]] != 1):
                rewards[agent_idx] -= 1.0  # Penalty for hitting a wall or obstacle

            self.agent_pos[agent_idx] = new_pos

        self.step_count += 1
        # Global episode done if max steps reached or all victims rescued
        global_done = (self.step_count >= self.max_steps) or np.all(self.victims_rescued)

        return TensorDict(
            {
                "observation": self._get_obs(),
                "reward": torch.tensor(rewards, dtype=torch.float32, device=self.device),
                "done": torch.tensor([global_done], dtype=torch.bool, device=self.device),  # Global done
                "next_observation": self._get_obs(),
                "action": tensordict["action"]
            },
            batch_size=[]  # No batch dimension for the environment
        )

if __name__ == "__main__":
    # Test the environment
    env = SearchAndRescueEnv(map_size=10, num_agents=5, num_victims=3)
    check_env_specs(env)
    print("Environment specs checked successfully!")
    obs = env.reset()
    print(f"Initial observation shape: {obs['observation'].shape}")
    action = torch.randint(0, 5, (5,))
    tensordict = TensorDict({"action": action}, batch_size=[])
    next_state = env.step(tensordict)
    print(f"Next state observation shape: {next_state['observation'].shape}")
    print(f"Reward shape: {next_state['reward'].shape}")
    print(f"Done shape: {next_state['done'].shape}")