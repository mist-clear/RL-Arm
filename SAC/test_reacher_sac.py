import numpy as np
import torch
import gymnasium as gym
from utility import Network
from custom_reacher_env import CustomReacherEnv
import argparse

class SACAgentEvaluator:
    def __init__(self, env, model_path, random, n_episode):
        self.env = env
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        self.n_episode = n_episode
        self.random = random
        self.model = Network(self.n_states, self.n_actions * 2, hidden_dim=256)
        if not self.random:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
    
    def choose_action(self, state):
        if self.random:
            action = self.env.action_space.sample()    
        else:
            state = torch.tensor(state).float()
            with torch.no_grad():
                output = self.model(state)
                mean, log_std = output[..., :self.n_actions], output[..., self.n_actions:]
                log_std = torch.clamp(log_std, min=-20, max=2)  # 加这一行
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                z = normal.sample()
                action = torch.tanh(z)
            action = action.detach().numpy()
            action = action * 0.5
        return action
    
    def evaluate(self):
        for i in range(self.n_episode):
            state, _ = self.env.reset()
            done, truncated = False, False
            eps_reward = 0
            n_steps = 0
            while not (done or truncated):
                self.env.render()
                action = self.choose_action(state)
                nextstate, reward, done, truncated, _ = self.env.step(action)
                state = nextstate
                eps_reward += reward
                n_steps += 1
            print(f"Episode {i+1}: Reward={eps_reward:.2f}, Steps={n_steps}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="models/actor_5000.pt")
    parser.add_argument('--rand', type=bool, default=False)
    parser.add_argument('--epos', type=int, default=50)
    args = parser.parse_args()

    # 用自定义环境加载xml
    env = CustomReacherEnv(render_mode='human')
    agent_evaluator = SACAgentEvaluator(env, args.model, args.rand, args.epos)
    agent_evaluator.evaluate()

if __name__ == '__main__':
    main()
