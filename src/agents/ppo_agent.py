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
        self.num_agents = env.num_agents

        # Get dimensions from environment specs
        base_env = env

        # Handle action_spec (might be Composite or directly Categorical)
        if hasattr(base_env.action_spec, "keys"):
            action_spec = base_env.action_spec["action"]
        else:
            action_spec = base_env.action_spec

        if hasattr(action_spec, "n"):
            action_dim = action_spec.n  # Should be 5
        else:
            raise ValueError(f"Action spec does not have 'n' attribute: {action_spec}")

        # Handle observation_spec (might be Composite or directly Unbounded)
        if hasattr(base_env.observation_spec, "keys"):
            obs_spec = base_env.observation_spec["observation"]
        else:
            obs_spec = base_env.observation_spec

        obs_shape = obs_spec.shape
        if len(obs_shape) >= 2:  # Expect (num_agents, 8)
            obs_dim = obs_shape[-1]  # Last dimension should be 8
        elif len(obs_shape) == 1:  # Handle case where batch dim is flattened
            obs_dim = obs_shape[0]
        else:
            raise ValueError(f"Unexpected observation spec shape: {obs_shape}")

        print(f"Creating PPO agent with obs_dim={obs_dim}, action_dim={action_dim}, num_agents={self.num_agents}")

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
            obs = TensorDict({"observation": obs}, batch_size=[self.num_agents])

        # Run the policy and get the action
        with torch.no_grad():
            policy_output = self.policy(obs.clone())
        return policy_output["action"]

    def value(self, obs):
        """Get value estimate from critic"""
        if not isinstance(obs, TensorDict):
            obs = TensorDict({"observation": obs}, batch_size=[self.num_agents])
        with torch.no_grad():
            return self.critic(obs)

    def __call__(self, tensordict):
        """Make the agent callable for the collector"""
        # Ensure we have the right input format
        if "observation" not in tensordict:
            raise ValueError("Expected 'observation' key in tensordict")

        # Forward pass through policy
        policy_output = self.policy(tensordict)

        # Add value estimate
        value_output = self.critic(tensordict)

        # Combine outputs
        result = tensordict.clone()
        result.update(policy_output)
        result.update(value_output)

        return result


def create_policy_and_critic(num_agents, obs_dim, action_dim, action_spec):
    """Create policy and critic networks"""

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
            # x shape: (batch_size, obs_dim) where batch_size = num_agents
            return self.net(x)

    class CriticNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            # For CTDE, critic can use global information
            # Here we'll use individual observations for simplicity
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        def forward(self, x):
            # x shape: (batch_size, obs_dim) where batch_size = num_agents
            return self.net(x)

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
        return_log_prob=True,
        default_interaction_type="random"
    )

    # Create critic module
    critic_net = CriticNetwork()
    critic = ValueOperator(
        module=critic_net,
        in_keys=["observation"],
        out_keys=["state_value"]
    )

    return policy, critic