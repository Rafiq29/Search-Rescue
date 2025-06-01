import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.modules import ProbabilisticActor, ValueOperator
from tensordict.nn import TensorDictModule
from torch.optim import Adam

class PPOAgent:
    def __init__(self, env, learning_rate=0.001, gamma=0.99, clip_eps=0.2):
        self.env = env
        self.gamma = gamma
        self.clip_eps = clip_eps

        # Get dimensions
        base_env = env
        # Handle action_spec (might be Composite or directly Categorical)
        if hasattr(base_env.action_spec, "keys"):
            action_spec = base_env.action_spec["action"]
        else:
            action_spec = base_env.action_spec  # Directly Categorical
        if hasattr(action_spec, "n"):
            action_dim = action_spec.n  # Should be 5
        else:
            raise ValueError(f"Action spec does not have 'n' attribute: {action_spec}")

        # Handle observation_spec (might be Composite or directly Unbounded)
        if hasattr(base_env.observation_spec, "keys"):
            obs_spec = base_env.observation_spec["observation"]
        else:
            obs_spec = base_env.observation_spec  # Directly Unbounded
        obs_shape = obs_spec.shape
        if len(obs_shape) >= 2:  # Expect (num_agents, 8)
            obs_dim = obs_shape[-1]  # Last dimension should be 8
        elif len(obs_shape) == 1:  # Handle case where batch dim is flattened
            obs_dim = obs_shape[0]
        else:
            raise ValueError(f"Unexpected observation spec shape: {obs_shape}")

        # Create policy and critic
        self.policy, self.critic = create_policy_and_critic(
            num_agents=base_env.num_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_spec=action_spec
        )

        # Optimizers
        self.actor_optimizer = Adam(self.policy.parameters(), lr=learning_rate)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=learning_rate)

    def policy(self, obs):
        if not isinstance(obs, TensorDict):
            # Wrap the observation tensor into a TensorDict with the correct batch size
            obs = TensorDict({"observation": obs}, batch_size=[self.env.num_agents])
        # Run the policy and get the action
        policy_output = self.policy(obs.clone())  # Clone to avoid modifying the input
        action = policy_output["action"]
        expected_batch_size = (self.env.num_agents,)
        if action.shape != expected_batch_size:
            raise ValueError(f"Action shape {action.shape} does not match expected batch size {expected_batch_size}")
        return action

    def value(self, obs):
        if not isinstance(obs, TensorDict):
            obs = TensorDict({"observation": obs}, batch_size=[self.env.num_agents])
        return self.critic(obs)

    def update(self, batch):
        obs = batch["observation"]
        actions = batch["action"]
        rewards = batch["reward"]
        next_obs = batch.get("next_observation", obs)
        dones = batch["done"]

        # Compute returns and advantages
        values = self.value(obs).squeeze(-1)
        next_values = self.value(next_obs).squeeze(-1)
        returns = rewards + self.gamma * next_values * (1 - dones)
        advantages = returns - values

        # Actor loss
        log_probs = self.policy.log_prob(obs, actions)
        old_log_probs = log_probs.detach()
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Critic loss
        critic_loss = nn.MSELoss()(values, returns)

        # Optimize
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss + 0.5 * critic_loss

def create_policy_and_critic(num_agents, obs_dim, action_dim, action_spec):
    # Define custom networks
    class PolicyNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
            )

        def forward(self, x):
            # x shape: (batch_size, obs_dim) = (5, 8)
            return self.net(x)  # Output shape: (5, action_dim) = (5, 5)

    class CriticNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim * num_agents, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        def forward(self, x):
            # x shape: (batch_size, obs_dim) = (5, 8)
            # Flatten for centralized critic
            x = x.view(-1, obs_dim * num_agents)  # Shape: (1, 40)
            return self.net(x)  # Output shape: (1, 1)

    # Create policy module (CTDE: decentralized execution with local observations)
    policy_net = PolicyNetwork()
    policy_module = TensorDictModule(
        policy_net,
        in_keys=["observation"],
        out_keys=["logits"]
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=["logits"],  # Only logits are needed for Categorical distribution
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True  # Ensure log_prob is available
    )

    # Create critic module (CTDE: centralized training with global info)
    critic_net = CriticNetwork()
    critic = ValueOperator(
        module=critic_net,
        in_keys=["observation"]
    )

    return policy, critic