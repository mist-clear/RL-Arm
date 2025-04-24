import torch
import numpy as np
import time
from env.custom_reacher_env import CustomReacherEnv
from td3 import Actor

env = CustomReacherEnv(render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

actor = Actor(state_dim, action_dim, max_action).cpu()
actor.load_state_dict(torch.load("td3/results_td3/best_actor.pth", map_location="cpu"))
actor.eval()

n_episodes = 20
for ep in range(n_episodes):
    state, _ = env.reset()
    ep_reward = 0
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).cpu()
        action = actor(state_tensor).cpu().data.numpy().flatten()
        state, reward, done, truncated, _ = env.step(action)
        ep_reward += reward
        env.render()
        time.sleep(0.015)

        if done or truncated:
            break
    print(f"Episode {ep}: Total Reward = {ep_reward:.2f}")
    
env.close()
