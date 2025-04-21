import os
import sys
import numpy as np
import torch
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm_2D import Arm_2D
from utility import Network

def choose_action(model, state, evaluate=True):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    logits = model(state)
    dist = torch.distributions.Categorical(logits=logits)
    return torch.argmax(logits).item() if evaluate else dist.sample().item()

def test_agent(model_path, episodes=5, render=True):
    env = Arm_2D()
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    model = Network(n_states, n_actions, hidden_dim=256)  # hidden_dim must match train.py
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        print(f"\n--- Episode {ep+1} ---")
        while not done:
            if render:
                env.render()
            action = choose_action(model, state, evaluate=True)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
        print(f"Episode {ep+1} Reward: {total_reward:.2f}")

    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/sac_2D_arm.pt', help='Path to trained model')
    parser.add_argument('--epos', type=int, default=5, help='Number of test episodes')
    parser.add_argument('--no_render', action='store_true', help='Disable rendering')
    args = parser.parse_args()

    test_agent(args.model, episodes=args.epos, render=not args.no_render)

if __name__ == '__main__':
    main()
