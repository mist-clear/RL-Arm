import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random

# Define a named tuple for experience replay
Rollout = namedtuple('Rollout', ['state', 'action', 'reward', 'next_state', 'done'])

# Define the neural network
class Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# SAC Agent
class SACAgent:
    def __init__(self, env, lr=3e-4, gamma=0.99, tau=0.005, memory_size=100000, batch_size=64, hidden_size=256, log_std_range=[-20, 2]):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.log_std_min, self.log_std_max = log_std_range

        # Dimensions of state and action spaces
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # Initialize networks
        self.actor = Network(self.state_dim, self.action_dim * 2, hidden_size).to(self.device)
        self.critic1 = Network(self.state_dim + self.action_dim, 1, hidden_size).to(self.device)
        self.critic2 = Network(self.state_dim + self.action_dim, 1, hidden_size).to(self.device)
        self.target_critic1 = Network(self.state_dim + self.action_dim, 1, hidden_size).to(self.device)
        self.target_critic2 = Network(self.state_dim + self.action_dim, 1, hidden_size).to(self.device)

        # Copy parameters to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Target entropy
        self.target_entropy = -np.prod(env.action_space.shape).item()
        self.alpha = self.log_alpha.exp()

        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)

    def store_rollout(self, rollout):
        self.memory.append(rollout)

    def sample_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32).to(self.device),
            torch.tensor(actions, dtype=torch.float32).to(self.device),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device),
            torch.tensor(next_states, dtype=torch.float32).to(self.device),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(self.device),
        )

    def get_action(self, state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, log_std = self.actor(state).chunk(2, dim=-1)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            if deterministic:
                action = mean
            else:
                action = normal.rsample()
            action = torch.tanh(action)
        return action.cpu().numpy()[0]

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_batch()

        # Update Critic
        with torch.no_grad():
            next_actions, next_log_probs = self.sample_action_and_log_prob(next_states)
            target_q1 = self.target_critic1(torch.cat([next_states, next_actions], dim=-1))
            target_q2 = self.target_critic2(torch.cat([next_states, next_actions], dim=-1))
            target_q = rewards + (1 - dones) * self.gamma * (torch.min(target_q1, target_q2) - self.alpha * next_log_probs)

        current_q1 = self.critic1(torch.cat([states, actions], dim=-1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=-1))
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update Actor
        actions, log_probs = self.sample_action_and_log_prob(states)
        q1 = self.critic1(torch.cat([states, actions], dim=-1))
        q2 = self.critic2(torch.cat([states, actions], dim=-1))
        actor_loss = (self.alpha * log_probs - torch.min(q1, q2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Alpha
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Soft update target networks
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)

    def sample_action_and_log_prob(self, states):
        mean, log_std = self.actor(states).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
