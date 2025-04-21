import random
from collections import deque
from torch import nn


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def store_rollout(self, rollout):
        self.memory.append(rollout)

    def sample_batch(self, batch_size):
        return random.sample(self.memory, batch_size)

    def initialise_memory(self, env, size):
        result = env.reset()
        if isinstance(result, tuple):
            state = result[0]
        else:
            state = result
        for _ in range(size):
            action = env.action_space.sample()
            step_result = env.step(action)
            # 兼容 step 返回4或5个值
            if isinstance(step_result, tuple) and len(step_result) >= 4:
                next_state, reward, done = step_result[:3]
                # 若有truncated等，done = done or truncated
                if len(step_result) > 4:
                    truncated = step_result[3]
                    done = done or truncated
            else:
                next_state, reward, done = step_result
            rollout = (state, action, reward, next_state, done)
            self.store_rollout(rollout)
            if done:
                result = env.reset()
                if isinstance(result, tuple):
                    state = result[0]
                else:
                    state = result
            else:
                state = next_state


class Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(Network, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        
    def forward(self, x):
        x = self.network(x)
        return x
