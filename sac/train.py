import os
import matplotlib.pyplot as plt
import torch
from env.custom_reacher_env import CustomReacherEnv
from sac import SACAgent

def train_sac(env, agent, episodes=1000, max_steps=200, save_freq=100, plot_freq=1000, start_episode=0, best_reward=0, avg_window=100):
    rewards = []
    recent_rewards = []  # Store rewards for calculating the moving average
    best_reward = float('-inf')  # Initialize the best reward

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
        recent_rewards.append(episode_reward)

        # Keep only the last `avg_window` rewards
        if len(recent_rewards) > avg_window:
            recent_rewards.pop(0)

        # Update model every `avg_window` episodes
        if (episode + 1) % avg_window == 0:
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}")

            # Save the model if the average reward is the best so far
            if avg_reward > best_reward:
                best_reward = avg_reward
                os.makedirs("models", exist_ok=True)
                torch.save(agent.actor.state_dict(), "sac/models/best_model.pt")
                print(f"New best model saved with avg reward: {best_reward:.2f}")

        # Periodically save the model
        if (episode + 1) % save_freq == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(agent.actor.state_dict(), f"sac/models/sac_{episode + 1}.pt")

        # Periodically plot and save rewards
        if (episode + 1) % plot_freq == 0:
            os.makedirs("plots", exist_ok=True)  # Ensure the directory exists
            plt.plot(rewards)
            plt.xlabel("Episodes")
            plt.ylabel("Reward")
            plt.title("SAC Training Rewards")
            plt.grid()
            plt.savefig(f"sac/plots/sac_rewards_up_to_episode_{episode + 1}.png")  # Save the plot
            plt.close()  # Close the plot to free memory

    return rewards, best_reward

if __name__ == "__main__":
    # Create environment
    env = CustomReacherEnv(render_mode=None)

    # Initialize SAC agent
    agent = SACAgent(env)
    start_episode = 0
    best_reward = float('-inf')

    # Continue training
    rewards, best_reward = train_sac(env, agent, episodes=1000000, max_steps=100, save_freq=100, plot_freq=1000, start_episode=start_episode, best_reward=best_reward, avg_window=100)

    # Final plot and save the rewards
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("SAC Training Rewards")
    plt.grid()
    os.makedirs("plots", exist_ok=True)  # Ensure the directory exists
    plt.savefig("sac/plots/sac_training_rewards.png")  # Save the plot as a PNG file
    plt.show()
