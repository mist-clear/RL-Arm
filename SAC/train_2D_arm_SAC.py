import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import torch
from torch import nn, optim
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm_2D import Arm_2D
from utility import Network, Memory

Rollout = namedtuple('Rollout', ['state', 'action', 'reward', 'next', 'done'])

class DiscreteSACAgent:
    def __init__(self, env, lr=1e-3, gamma=0.95, tau=0.005, memory_size=5000, hidden_size=512):
        self.env = env
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.lr = lr
        self.gamma = gamma
        self.tau = tau

        self.critic1 = Network(self.n_states, self.n_actions, hidden_size)
        self.critic2 = Network(self.n_states, self.n_actions, hidden_size)
        self.target_critic1 = Network(self.n_states, self.n_actions, hidden_size)
        self.target_critic2 = Network(self.n_states, self.n_actions, hidden_size)

        self.actor = Network(self.n_states, self.n_actions, hidden_size)

        self.target_entropy = -np.log(1.0 / self.n_actions)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()

        self.critic_optim1 = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic_optim2 = optim.Adam(self.critic2.parameters(), lr=self.lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr)

        self.criterion = nn.MSELoss()
        self.memory = Memory(capacity=memory_size)

        self.update_target_networks(tau=1)

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau
        for t_p, p in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            t_p.data.copy_(tau * p.data + (1 - tau) * t_p.data)
        for t_p, p in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            t_p.data.copy_(tau * p.data + (1 - tau) * t_p.data)

    def choose_action(self, state, evaluate=False):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits = self.actor(state)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().item() if not evaluate else torch.argmax(logits).item()

    def unpack_batch(self, samples):
        states, actions, rewards, next_states, dones = map(list, zip(*samples))
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        )

    def learn(self, samples):
        states, actions, rewards, next_states, dones = self.unpack_batch(samples)

        with torch.no_grad():
            next_logits = self.actor(next_states)
            next_dist = torch.distributions.Categorical(logits=next_logits)
            next_probs = next_dist.probs
            next_log_probs = next_dist.logits - torch.logsumexp(next_dist.logits, dim=1, keepdim=True)
            next_q1 = self.target_critic1(next_states)
            next_q2 = self.target_critic2(next_states)
            next_min_q = torch.min(next_q1, next_q2)
            next_v = (next_probs * (next_min_q - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = rewards + (1 - dones) * self.gamma * next_v

        q1 = self.critic1(states).gather(1, actions)
        q2 = self.critic2(states).gather(1, actions)
        critic_loss1 = self.criterion(q1, target_q)
        critic_loss2 = self.criterion(q2, target_q)

        self.critic_optim1.zero_grad()
        self.critic_optim2.zero_grad()
        critic_loss1.backward()
        critic_loss2.backward()
        self.critic_optim1.step()
        self.critic_optim2.step()

        logits = self.actor(states)
        dist = torch.distributions.Categorical(logits=logits)
        probs = dist.probs
        log_probs = dist.logits - torch.logsumexp(dist.logits, dim=1, keepdim=True)
        q1_pi = self.critic1(states)
        q2_pi = self.critic2(states)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (probs * (self.alpha * log_probs - min_q_pi)).sum(dim=1).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha * (log_probs.sum(dim=1, keepdim=True) + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        self.update_target_networks()
        return critic_loss1.item(), actor_loss.item(), alpha_loss.item()

    def train(self, n_episode=1000, batch_size=256):
        self.memory.initialise_memory(self.env, size=5000)
        results = []
        for ep in range(n_episode):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.memory.store_rollout(Rollout(state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                if len(self.memory.memory) >= batch_size:
                    samples = self.memory.sample_batch(batch_size)
                    self.learn(samples)

            results.append(total_reward)
            # if ep % 10 == 0:
            print(f"Episode {ep}, Reward: {total_reward:.2f}")

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epos', type=int, default=1000)
    args = parser.parse_args()

    env = Arm_2D()
    agent = DiscreteSACAgent(env)
    rewards = agent.train(n_episode=args.epos)

    os.makedirs("models", exist_ok=True)
    torch.save(agent.actor.state_dict(), "models/sac_2D_arm.pt")
    plt.plot(rewards)
    plt.title("Discrete SAC on 2D Arm")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
