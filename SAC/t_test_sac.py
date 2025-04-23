import os
import time
import torch
import numpy as np
from custom_reacher_env import CustomReacherEnv
from t_sac import Network

def test_sac(model_path, episodes=10, render_mode="human"):
    # 创建环境
    env = CustomReacherEnv(render_mode=render_mode)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    actor = Network(state_dim, action_dim * 2, hidden_size=256).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()

    for episode in range(episodes):
        state, _ = env.reset()
        done, truncated = False, False
        episode_reward = 0
        step_count = 0  # 初始化步数计数器

        while not (done or truncated):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                mean, log_std = actor(state_tensor).chunk(2, dim=-1)
                action = torch.tanh(mean).cpu().numpy()[0]  # 取均值作为动作
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            step_count += 1  # 每步加 1
            
            env.render()
            time.sleep(0.01)

        print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward:.2f}, Steps: {step_count}")
    env.close()

if __name__ == "__main__":
    # 模型路径
    model_path = os.path.join("test_model/sac_7000.pt")  # 修改为你的模型路径
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # 测试模型
    test_sac(model_path, episodes=50)
