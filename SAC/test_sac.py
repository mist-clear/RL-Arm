import os
import time
import torch
import numpy as np
from custom_reacher_env import CustomReacherEnv
from sac import Network

def test_sac(model_path, episodes=10, render_mode="human"):
    env = CustomReacherEnv(render_mode=render_mode)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = Network(state_dim, action_dim * 2, hidden_size=256).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()

    for episode in range(episodes):
        state, _ = env.reset()
        done, truncated = False, False
        episode_reward = 0
        step_count = 0

        while not (done or truncated):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                mean, log_std = actor(state_tensor).chunk(2, dim=-1)
                action = torch.tanh(mean).cpu().numpy()[0]
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            step_count += 1
            
            env.render()
            time.sleep(0.015)

        print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward:.2f}, Steps: {step_count}")
    env.close()

if __name__ == "__main__":
    model_path = os.path.join("models/best_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    test_sac(model_path, episodes=20)
