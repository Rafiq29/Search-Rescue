import numpy as np
import torch
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs
from tensordict import TensorDict
from torchrl.data import Unbounded, Composite, Categorical

# Debug print to confirm this file is loaded
print("Loading env.py from SearchAndRescueEnv module")

class SearchAndRescueEnv(EnvBase):
    def __init__(self, map_size=10, num_agents=5, num_victims=3, max_steps=100, obstacle_density=0.2,
                 terrain_difficulty=0.1, device="cpu"):
        super().__init__(device=device)
        self.map_size = map_size
        self.num_agents = num_agents
        self.num_victims = num_victims
        self.max_steps = max_steps
        self.obstacle_density = obstacle_density
        self.terrain_difficulty = terrain_difficulty

        # Define observation and action specs as CompositeSpec for consistency
        self.observation_spec = Composite(
            observation=Unbounded(
                shape=(num_agents, 8),
                device=self.device
            )
        )
        self.action_spec = Composite(
            action=Categorical(
                n=5,  # 5 discrete actions: Up, Down, Left, Right, Rescue
                shape=(num_agents,),
                device=self.device
            )
        )
        self.reward_spec = Composite(
            reward=Unbounded(
                shape=(num_agents,),
                device=self.device
            )
        )
        self.done_spec = Composite(
            done=Unbounded(
                shape=(1,),
                device=self.device,
                dtype=torch.bool
            ),
            terminated=Unbounded(
                shape=(1,),
                device=self.device,
                dtype=torch.bool
            ),
            truncated=Unbounded(
                shape=(1,),
                device=self.device,
                dtype=torch.bool
            )
        )

        # Debug prints to verify specs
        print(f"Observation spec: {self.observation_spec}")
        print(f"Action spec: {self.action_spec}")
        print(f"Reward spec: {self.reward_spec}")
        print(f"Done spec: {self.done_spec}")

        # Initialize environment state
        self._initialize_state()

    def _initialize_state(self):
        """Initialize or reset the environment state"""
        self.grid = self._generate_grid()
        self.agent_pos = self._place_agents()
        self.victim_pos = self._place_victims()
        self.safe_zones = self._place_safe_zones()
        self.victims_rescued = np.zeros(self.num_victims, dtype=bool)
        self.step_count = 0

    def _generate_grid(self):
        grid = np.zeros((self.map_size, self.map_size))
        num_obstacles = int(self.map_size * self.map_size * self.obstacle_density)
        if num_obstacles > 0:
            obstacle_indices = np.random.choice(self.map_size * self.map_size, num_obstacles, replace=False)
            for idx in obstacle_indices:
                x, y = divmod(idx, self.map_size)
                grid[x, y] = 1  # 1 represents an obstacle
        return grid

    def _place_agents(self):
        positions = []
        for _ in range(self.num_agents):
            attempts = 0
            while attempts < 100:  # Prevent infinite loops
                pos = np.random.randint(0, self.map_size, 2)
                if self.grid[pos[0], pos[1]] == 0 and not any((pos == p).all() for p in positions):
                    positions.append(pos)
                    break
                attempts += 1
            if attempts >= 100:
                # Fallback: place agent at first available position
                for x in range(self.map_size):
                    for y in range(self.map_size):
                        pos = np.array([x, y])
                        if self.grid[x, y] == 0 and not any((pos == p).all() for p in positions):
                            positions.append(pos)
                            break
                    if len(positions) == len(positions):
                        break
        return np.array(positions)

    def _place_victims(self):
        positions = []
        occupied_positions = list(self.agent_pos)

        for _ in range(self.num_victims):
            attempts = 0
            while attempts < 100:
                pos = np.random.randint(0, self.map_size, 2)
                if (self.grid[pos[0], pos[1]] == 0 and
                        not any((pos == p).all() for p in positions + occupied_positions)):
                    positions.append(pos)
                    break
                attempts += 1
        return np.array(positions)

    def _place_safe_zones(self):
        safe_zones = []
        occupied_positions = list(self.agent_pos) + list(self.victim_pos)

        for _ in range(4):  # 4 safe zones
            attempts = 0
            while attempts < 100:
                pos = np.random.randint(0, self.map_size, 2)
                if (self.grid[pos[0], pos[1]] == 0 and
                        not any((pos == p).all() for p in safe_zones + occupied_positions)):
                    safe_zones.append(pos)
                    break
                attempts += 1
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
        obs = np.zeros((self.num_agents, 8), dtype=np.float32)

        for agent_idx in range(self.num_agents):
            pos = self.agent_pos[agent_idx]

            # Agent position (normalized)
            obs[agent_idx, 0:2] = pos / self.map_size

            # Nearest victim (only unrescued ones)
            alive_victims = [v for i, v in enumerate(self.victim_pos) if not self.victims_rescued[i]]
            if alive_victims:
                distances = [np.linalg.norm(v - pos) for v in alive_victims]
                nearest_victim_idx = np.argmin(distances)
                nearest_victim = alive_victims[nearest_victim_idx]
                obs[agent_idx, 2:4] = (nearest_victim / self.map_size).astype(np.float32)
            else:
                obs[agent_idx, 2:4] = (pos / self.map_size).astype(np.float32)

            # Nearest safe zone
            if len(self.safe_zones) > 0:
                distances = [np.linalg.norm(sz - pos) for sz in self.safe_zones]
                nearest_safe_zone_idx = np.argmin(distances)
                nearest_safe_zone = self.safe_zones[nearest_safe_zone_idx]
                obs[agent_idx, 4:6] = (nearest_safe_zone / self.map_size).astype(np.float32)
            else:
                obs[agent_idx, 4:6] = (pos / self.map_size).astype(np.float32)

            # Has victim nearby (at same position)
            victim_idx = self._get_victim_at(pos)
            obs[agent_idx, 6] = 1.0 if victim_idx is not None and not self.victims_rescued[victim_idx] else 0.0

            # Step count (normalized)
            obs[agent_idx, 7] = self.step_count / self.max_steps if self.max_steps > 0 else 0.0

        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    def _set_seed(self, seed):
        """Set the seed for the environment's random number generators."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Reset the environment to ensure the new seed takes effect
        self._initialize_state()
        return seed

    def _reset(self, tensordict=None, **kwargs):
        """Reset the environment and return initial state"""
        self._initialize_state()
        obs = self._get_obs()

        return TensorDict(
            {
                "observation": obs,
                "done": torch.tensor([False], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([False], dtype=torch.bool, device=self.device),
                "truncated": torch.tensor([False], dtype=torch.bool, device=self.device),
                "reward": torch.zeros(self.num_agents, dtype=torch.float32, device=self.device)
            },
            batch_size=torch.Size([])
        )

    def _step(self, tensordict):
        actions = tensordict["action"].numpy()
        print(f"actions shape: {actions.shape}, actions: {actions}")  # Debug print
        rewards = np.zeros(self.num_agents, dtype=np.float32)

        for agent_idx in range(self.num_agents):
            action = actions[agent_idx]  # Scalar action for the current agent
            print(f"agent_idx: {agent_idx}, action type: {type(action)}, action: {action}")  # Debug print
            agent_pos = self.agent_pos[agent_idx].copy()  # Ensure we work with a copy
            new_pos = agent_pos.copy()

            # Apply action for the current agent
            if action == 0:  # Up
                new_pos[1] = max(0, new_pos[1] - 1)
            elif action == 1:  # Down
                new_pos[1] = min(self.map_size - 1, new_pos[1] + 1)
            elif action == 2:  # Left
                new_pos[0] = max(0, new_pos[0] - 1)
            elif action == 3:  # Right
                new_pos[0] = min(self.map_size - 1, new_pos[0] + 1)
            elif action == 4:  # Rescue
                victim_idx = self._get_victim_at(agent_pos)
                if victim_idx is not None and not self.victims_rescued[victim_idx]:
                    self.victims_rescued[victim_idx] = True
                    rewards[agent_idx] = 10.0  # Reward for rescuing a victim
                    safe_zone_idx = self._get_safe_zone_at(agent_pos)
                    if safe_zone_idx is not None:
                        rewards[agent_idx] += 5.0  # Additional reward for being in a safe zone
                continue  # Don't move when rescuing

            # Check boundaries and obstacles (movement actions only)
            if action < 4:  # Only for movement actions
                if (0 <= new_pos[0] < self.map_size and
                        0 <= new_pos[1] < self.map_size and
                        self.grid[new_pos[0], new_pos[1]] != 1):
                    self.agent_pos[agent_idx] = new_pos
                else:
                    rewards[agent_idx] -= 1.0  # Penalty for hitting a wall or obstacle

        self.step_count += 1
        # Global episode done if max steps reached or all victims rescued
        global_done = (self.step_count >= self.max_steps) or np.all(self.victims_rescued)
        terminated = global_done
        truncated = self.step_count >= self.max_steps

        # Get next observation (after state update)
        next_obs = self._get_obs()

        # Validate that no values are None
        for key, value in zip(
            ["observation", "action", "reward", "done", "terminated", "truncated", "next_observation"],
            [next_obs, actions, rewards, global_done, terminated, truncated, next_obs]
        ):
            if value is None:
                raise ValueError(f"Found None value for key {key}")

        # Return flat TensorDict with all keys at top level
        return TensorDict(
            {
                "observation": next_obs,
                "action": torch.tensor(actions, device=self.device),
                "reward": torch.tensor(rewards, dtype=torch.float32, device=self.device),
                "done": torch.tensor([global_done], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([terminated], dtype=torch.bool, device=self.device),
                "truncated": torch.tensor([truncated], dtype=torch.bool, device=self.device),
                "next_observation": next_obs
            },
            batch_size=torch.Size([])
        )

if __name__ == "__main__":
    # Test the environment
    env = SearchAndRescueEnv(map_size=10, num_agents=5, num_victims=3)
    try:
        check_env_specs(env)
        print("Environment specs checked successfully!")
    except Exception as e:
        print(f"Environment spec check failed: {e}")

    # Test reset and step
    obs = env.reset()
    print(f"Initial observation shape: {obs['observation'].shape}")

    # Test step
    action = torch.randint(0, 5, (5,))
    tensordict = TensorDict({"action": action}, batch_size=torch.Size([]))
    next_state = env.step(tensordict)
    print(f"Next state observation shape: {next_state['observation'].shape}")
    print(f"Reward shape: {next_state['reward'].shape}")
    print(f"Done shape: {next_state['done'].shape}")
    print("Environment test completed successfully!")