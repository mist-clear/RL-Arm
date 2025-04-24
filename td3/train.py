import os
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit
from env.custom_reacher_env import CustomReacherEnv
from td3 import TD3
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

raw_env = CustomReacherEnv(render_mode=None, only_first_phase=False, max_steps=50)
env = TimeLimit(raw_env, max_episode_steps=100)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer(state_dim, action_dim)

max_timesteps = 2_000_000
start_timesteps = 10_000
batch_size = 256
expl_noise = 0.1 * max_action

eps_start = 0.1
eps_end = 0.01
eps_decay = max_timesteps

discount = 0.99
tau = 0.005
policy_noise = 0.2 * max_action
noise_clip = 0.5 * max_action
policy_freq = 2

eval_freq = 5_000
log_freq = 1_000

save_dir = "results_td3"
os.makedirs(save_dir, exist_ok=True)

best_eval = -np.inf
reward_history = []
eval_log = []

ep_num = 0
ep_steps = 0
ep_reward = 0.0

state, _ = env.reset()
print(f"Episode {ep_num} start, red at {raw_env.data.site_xpos[raw_env.target1_id][:2]}")

for t in range(1, max_timesteps + 1):
    ep_steps += 1

    if t < start_timesteps:
        action = env.action_space.sample()
    else:
        eps = eps_end + (eps_start - eps_end) * max(0, (eps_decay - t) / eps_decay)
        if np.random.rand() < eps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(np.array(state))
            action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                env.action_space.low, env.action_space.high
            )

    next_state, reward, done, truncated, _ = env.step(action)
    replay_buffer.add(state, action, next_state, reward, float(done or truncated))
    state = next_state
    ep_reward += reward

    if t >= start_timesteps:
        agent.train(
            replay_buffer,
            batch_size,
            discount=discount,
            tau=tau,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_freq=policy_freq,
        )

    if done or truncated:
        print(f"Episode {ep_num} | Steps: {ep_steps} | Reward: {ep_reward:.2f}")
        reward_history.append(ep_reward)
        state, _ = env.reset()
        print(f"Episode {ep_num + 1} start, red at {raw_env.data.site_xpos[raw_env.target1_id][:2]}")
        ep_num += 1
        ep_steps = 0
        ep_reward = 0.0

    if t % log_freq == 0:
        ft = raw_env.data.site_xpos[raw_env.s_fingertip][:2]
        t1 = raw_env.data.site_xpos[raw_env.target1_id][:2]
        t2 = raw_env.data.site_xpos[raw_env.target2_id][:2]
        print(f"Time {t} | EpR {ep_reward:.2f} | d1={np.linalg.norm(ft - t1):.3f}, d2={np.linalg.norm(ft - t2):.3f}")

    if t % eval_freq == 0:
        avg_r = 0.0
        for _ in range(5):
            s, _ = env.reset()
            done = False
            while not done:
                a = agent.select_action(np.array(s))
                s, r, done, *_ = env.step(a)
                avg_r += r
        avg_r /= 5
        print(f"Eval {t}: avg_r={avg_r:.2f}")
        eval_log.append((t, avg_r))
        if avg_r > best_eval:
            best_eval = avg_r
            print("New best, saving models")
            torch.save(agent.actor.state_dict(), f"{save_dir}/best_actor.pth")
            torch.save(agent.critic.state_dict(), f"{save_dir}/best_critic.pth")
        torch.save(agent.actor.state_dict(), f"{save_dir}/actor_{t}.pth")
        torch.save(agent.critic.state_dict(), f"{save_dir}/critic_{t}.pth")

np.save(os.path.join(save_dir, "reward_history.npy"), np.array(reward_history))
np.save(os.path.join(save_dir, "eval_log.npy"), np.array(eval_log))

plt.figure(figsize=(8,5))
plt.plot(reward_history, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("TD3 Training Reward Convergence")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "learning_curve.png"))

data = np.array(eval_log)
ts, avg_r = data[:,0], data[:,1]
plt.figure(figsize=(8,5))
plt.plot(ts, avg_r, '-o', label='Avg Eval Reward')
plt.xlabel("Timestep")
plt.ylabel("Average Eval Reward")
plt.title("Evaluation Performance Over Training")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "eval_curve.png"))

print(f"Saved reward_history.npy, learning_curve.png, eval_log.npy, eval_curve.png in {save_dir}")
