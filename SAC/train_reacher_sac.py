import os
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import torch
from custom_reacher_env import CustomReacherEnv
import gymnasium as gym
from utility import Network, Memory
from torch import nn, optim
import argparse

# Information unit
Rollout = namedtuple('Rollout', ['state', 'action', 'reward', 'next', 'done'])

class SACAgent:
    def __init__(self, env, lr=1e-3, gamma=0.99, soft_update_tau=0.01,
                 memory_size=2000, hidden_size=128, log_std_range=[-20,2]):
        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        self.lr = lr
        self.gamma = gamma
        self.tau = soft_update_tau
        self.hidden_size = hidden_size
        self.min_clamp = log_std_range[0]
        self.max_clamp = log_std_range[-1]

        # 初始化网络和优化器
        self._initialise_model()
        self.update_target_networks(tau=1)
        self.criterion = nn.MSELoss()
        self.memory = Memory(capacity=memory_size)
        
    def _initialise_model(self):
        self.critic1 = Network(self.n_states + self.n_actions, 1, self.hidden_size).to(self.device)
        self.critic2 = Network(self.n_states + self.n_actions, 1, self.hidden_size).to(self.device)
        self.target_critic1 = Network(self.n_states + self.n_actions, 1, self.hidden_size).to(self.device)
        self.target_critic2 = Network(self.n_states + self.n_actions, 1, self.hidden_size).to(self.device)
        self.actor = Network(self.n_states, self.n_actions*2, self.hidden_size).to(self.device)
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.critic_optim1 = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic_optim2 = optim.Adam(self.critic2.parameters(), lr=self.lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr, eps=1e-4)
        
    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau
        for local_param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(tau * local_param.data + (1-tau) * target_param.data)
        for local_param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(tau * local_param.data + (1-tau) * target_param.data)
        
    def get_action_prob(self, state, epsilon=1e-6):
        state = state.float().to(self.device)
        output = self.actor(state)
        mean, log_std = output[..., :self.n_actions], output[..., self.n_actions:]
        log_std = torch.clamp(log_std, self.min_clamp, self.max_clamp)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob
    
    def critic_loss(self, states, actions, rewards, nextstates, done):
        with torch.no_grad():
            next_actions, next_log_probs = self.get_action_prob(nextstates)
            next_q1 = self.target_critic1(torch.cat((nextstates, next_actions), dim=1))
            next_q2 = self.target_critic2(torch.cat((nextstates, next_actions), dim=1))
            min_next_q = torch.min(next_q1, next_q2)
            soft_state = min_next_q - self.alpha * next_log_probs
            target_q = rewards + (1 - done) * self.gamma * soft_state
        pred_q1 = self.critic1(torch.cat((states, actions), dim=1))
        pred_q2 = self.critic2(torch.cat((states, actions), dim=1))
        loss1 = self.criterion(pred_q1, target_q)
        loss2 = self.criterion(pred_q2, target_q)
        return loss1, loss2
        
    def actor_loss(self, states):
        actions, log_prob = self.get_action_prob(states)
        q_values1 = self.critic1(torch.cat((states, actions), dim=1))
        q_values2 = self.critic2(torch.cat((states, actions), dim=1))
        min_q_values = torch.min(q_values1, q_values2)
        policy_loss = -(min_q_values-self.alpha * log_prob).mean()
        return policy_loss, log_prob
    
    def temperature_loss(self, log_prob):
        loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        return loss
        
    def _choose_action(self, state, random=False):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state)
        if random:
            actions = self.env.action_space.sample()            
        else:
            with torch.no_grad():
                actions, _ = self.get_action_prob(state)
        actions = np.array(actions.cpu()).reshape(-1)
        actions = actions * 0.5 
        return actions
        
    def unpack_batch(self, samples):
        batch_data = list(map(list, zip(*samples)))
        batch_states = torch.tensor(np.array(batch_data[0])).float().to(self.device)
        batch_actions = torch.tensor(np.array(batch_data[1])).float().to(self.device)
        batch_rewards = torch.tensor(np.array(batch_data[2])).float().unsqueeze(-1).to(self.device)
        batch_nextstates = torch.tensor(np.array(batch_data[3])).float().to(self.device)
        batch_done = torch.tensor(np.array(batch_data[4])).float().unsqueeze(-1).to(self.device)
        return batch_states, batch_actions, batch_rewards, batch_nextstates, batch_done

    def learn(self, samples):
        batch_states, batch_actions, batch_rewards, batch_nextstates, batch_done = self.unpack_batch(samples)
        self.critic_optim1.zero_grad()
        self.critic_optim2.zero_grad()
        critic_loss1, critic_loss2 = self.critic_loss(
            batch_states, batch_actions, batch_rewards, batch_nextstates, batch_done)
        critic_loss1.backward()
        critic_loss2.backward()
        self.critic_optim1.step()
        self.critic_optim2.step()
        self.actor_optim.zero_grad()         
        actor_loss, log_probs = self.actor_loss(batch_states)
        actor_loss.backward()
        self.actor_optim.step()
        self.alpha_optim.zero_grad()               
        alpha_loss = self.temperature_loss(log_probs)
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()
        self.update_target_networks(tau=self.tau)
        return torch.min(critic_loss1, critic_loss2).item(), actor_loss.item(), alpha_loss.item()
    
    def train(self, n_episode=250, initial_memory=None,
          report_freq=10, batch_size=32):
        if initial_memory is None:
            initial_memory = batch_size*10
        self.memory.initialise_memory(self.env, size=initial_memory)
        results = []
        for i in range(n_episode):
            state, _ = self.env.reset()
            done, truncated = False, False
            eps_reward = 0
            while not (done or truncated):
                action = self._choose_action(state)
                nextstate, reward, done, truncated, _ = self.env.step(action)
                roll = Rollout(state, action, reward, nextstate, done)
                self.memory.store_rollout(roll)
                state = nextstate
                samples = self.memory.sample_batch(batch_size)
                critic_loss, actor_loss, alpha_loss = self.learn(samples)
                eps_reward += reward
                if eps_reward < -30:
                    done = True
            results.append(eps_reward)
            if i % report_freq == 0:
                print(f'Episode {i}/{n_episode} \t Reward: {eps_reward:.4f} \t Critic Loss: {critic_loss:.3f}\t '+
                    f'Actor Loss: {actor_loss:.3f}\t Alpha Loss: {alpha_loss:.3f}\t Alpha: {self.alpha.item():.4f}')
            if (i + 1) % 200 == 0:
                actor_path = os.path.join('models', f'actor_{i+1}_reward_{int(eps_reward)}.pt')
                torch.save(self.actor.state_dict(), actor_path)
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epos', type=int, default=2000000, required=False)
    args = parser.parse_args()
    epos = args.epos

    env = CustomReacherEnv(render_mode=None)
    agent = SACAgent(env, lr=3e-4, gamma=0.99, memory_size=20000, hidden_size=256)
    learning_data = agent.train(n_episode=epos, batch_size=64, report_freq=100)
    actor = os.path.join('models/actor_' + str(epos) + '.pt')
    torch.save(agent.actor.state_dict(), actor)

    plt.plot(learning_data, label='Reward over episodes')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
