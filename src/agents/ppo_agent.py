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

    def get_action(self, obs):
        """Get action from policy - this is what the collector will call"""
        if not isinstance(obs, TensorDict):
            obs = TensorDict({"observation": obs}, batch_size=[self.env.num_agents])

        # Run the policy and get the action
        policy_output = self.policy(obs.clone())
        return policy_output["action"]

    def value(self, obs):
        if not isinstance(obs, TensorDict):
            obs = TensorDict({"observation": obs}, batch_size=[self.env.num_agents])
        return self.critic(obs)

    def __call__(self, tensordict):
        """Make the agent callable for the collector"""
        return self.policy(tensordict)


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
            # Use individual agent observations for simplicity
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        def forward(self, x):
            # x shape: (batch_size, obs_dim) = (5, 8)
            # Process each agent's observation individually
            batch_size = x.shape[0]
            values = []
            for i in range(batch_size):
                agent_obs = x[i:i + 1]  # Keep batch dimension
                value = self.net(agent_obs)
                values.append(value)
            return torch.cat(values, dim=0)  # Shape: (5, 1)

    # Create policy module (CTDE: decentralized execution with local observations)
    policy_net = PolicyNetwork()
    policy_module = TensorDictModule(
        policy_net,
        in_keys=["observation"],
        out_keys=["logits"]
    )

    # Create the probabilistic actor
    policy = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True
    )

    # Create critic module
    critic_net = CriticNetwork()
    critic = ValueOperator(
        module=critic_net,
        in_keys=["observation"]
    )

    return policy, critic