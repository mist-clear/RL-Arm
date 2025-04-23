import os
import matplotlib.pyplot as plt
import torch
from custom_reacher_env import CustomReacherEnv
from t_sac import SACAgent

def train_sac(env, agent, episodes=1000, max_steps=200, save_freq=100):
    rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.store_rollout((state, action, reward, next_state, done))
            agent.update()
            state = next_state
            episode_reward += reward
            if done or truncated:
                break
        rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward:.2f}")

        if (episode + 1) % save_freq == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(agent.actor.state_dict(), f"test_model/sac_{episode + 1}.pt")

    return rewards

if __name__ == "__main__":
    env = CustomReacherEnv(render_mode=None)
    agent = SACAgent(env)
    rewards = train_sac(env, agent, episodes=50000)

    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("SAC Training Rewards")
    plt.grid()
    plt.show()
