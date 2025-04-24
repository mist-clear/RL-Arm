# RL-Arm: 2D Robotic Arm Control

This repository implements reinforcement learning algorithms for controlling a robotic arm in a simulated environment. The project uses the Soft Actor-Critic (SAC) and Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithms to train agents in a custom Mujoco-based environment.

## Features

- **Custom Environment**: A Mujoco-based environment for robotic arm control.
- **Reinforcement Learning Algorithms**:
  - Soft Actor-Critic (SAC)
  - Twin Delayed Deep Deterministic Policy Gradient (TD3)
- **Model Saving and Visualization**:
  - Periodic saving of models during training.
  - Reward plots for monitoring training progress.

---

## Installation
```bash
git clone https://github.com/mist-clear/RL-Arm.git
cd RL-Arm
conda create -n rl-arm python=3.8 -y
conda activate rl-arm
pip install -r requirements.txt
```
Ensure Mujoco is properly installed and configured. Refer to the Mujoco installation guide for details.


**Training**
To train the SAC/TD3 agent:
```python
python train.py
```

**Testing**

To test a trained SAC/TD3 model:
```python
python test.py
```

