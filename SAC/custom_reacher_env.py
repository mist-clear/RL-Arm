import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv

class CustomReacherEnv(MujocoEnv, gym.utils.EzPickle):
    def __init__(self, render_mode=None, only_first_phase=False, max_steps=100):
        self.only_first_phase = only_first_phase
        xml_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../SAC/custom_reacher.xml")
        )
        self.frame_skip = 2
        self.max_steps = max_steps

        self.observation_space = Box(-np.inf, np.inf, (10,), np.float32)
        self.action_space      = Box(-0.5, 0.5, (2,), np.float32)

        gym.utils.EzPickle.__init__(self)
        MujocoEnv.__init__(
            self,
            model_path=xml_path,
            frame_skip=self.frame_skip,
            observation_space=self.observation_space,
            render_mode=render_mode,
        )

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.phase = 0
        self.current_step = 0

        self.j_target1_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target1_x")
        self.j_target1_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target1_y")
        self.j_target2_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target2_x")
        self.j_target2_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target2_y")

        self.s_fingertip = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,  "fingertip")
        self.s_target1   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target1_site")
        self.s_target2   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target2_site")

        self.fingertip_id = self.s_fingertip
        self.target1_id   = self.s_target1
        self.target2_id   = self.s_target2

        self._target1_pos = np.zeros(2, dtype=np.float32)
        self.prev_dist1   = None
        self.prev_dist2   = None
        self.prev_action  = None
        self.prev_ft_pos  = None

    def get_obs(self):
        qpos      = self.data.qpos[:2].copy()
        qvel      = self.data.qvel[:2].copy()
        fingertip = self.data.site_xpos[self.s_fingertip][:2].copy()
        t1        = self.data.site_xpos[self.s_target1][:2].copy()
        t2        = self.data.site_xpos[self.s_target2][:2].copy()
        return np.concatenate([qpos, qvel, fingertip, t1, t2])

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        self.do_simulation(action, self.frame_skip)

        # Pin targets
        self.data.qpos[self.j_target1_x:self.j_target1_y+1] = self._target1_pos
        self.data.qvel[self.j_target1_x:self.j_target1_y+1] = 0.0
        self.data.qpos[self.j_target2_x:self.j_target2_y+1] = self.init_qpos[self.j_target2_x:self.j_target2_y+1]
        self.data.qvel[self.j_target2_x:self.j_target2_y+1] = 0.0
        mujoco.mj_forward(self.model, self.data)

        obs = self.get_obs()
        self.current_step += 1
        fingertip = obs[4:6]
        reward = 0.0
        done = False

        # Stage 1: approach red ball
        if self.phase == 0:
            dist1 = np.linalg.norm(fingertip - obs[6:8])
            dist2 = np.linalg.norm(self.data.site_xpos[self.s_fingertip][:2]
                               - self.data.site_xpos[self.s_target2][:2])

            if self.prev_dist1 is None:
                self.prev_dist1 = dist1
            K1, max_inc1 = 2.0, 0.1
            delta1 = self.prev_dist1 - dist1
            inc1 = np.clip(max(0.0, delta1) * K1, 0.0, max_inc1)
            reward += inc1
            self.prev_dist1 = dist1
            if dist1 < 0.02:
                # print("Hit red ball, transitioning to green ball phase")
                reward += 20.0
                self.phase = 1
                self.prev_dist2 = np.linalg.norm(fingertip - obs[8:10])
                if self.only_first_phase:
                    return obs, reward, True, False, {}
        # Stage 2: approach green ball
        elif self.phase == 1:
            dist2 = np.linalg.norm(fingertip - obs[8:10])
            reward -= 0.5 * dist2
            K2, max_inc2 = 2.0, 0.1
            delta2 = self.prev_dist2 - dist2
            inc2 = np.clip(max(0.0, delta2) * K2, 0.0, max_inc2)
            reward += inc2
            self.prev_dist2 = dist2
            if dist2 < 0.04:
                print("Hit green ball, ending episode")
                reward += 30.0
                return obs, reward, True, False, {}

        # Penalties
        reward -= 0.005
        reward -= 0.02 * np.sum(np.square(action))
        reward -= 0.005 * np.sum(np.square(self.data.qvel[:2]))
        if self.prev_action is not None:
            reward -= np.linalg.norm(action - self.prev_action)
        self.prev_action = action.copy()
        if self.prev_ft_pos is not None:
            move_dist = np.linalg.norm(fingertip - self.prev_ft_pos)
            if move_dist < 0.01:
                reward -= 1.0
        self.prev_ft_pos = fingertip.copy()

        # Timeout
        if self.current_step >= self.max_steps:
            done = True

        return obs, reward, done, False, {}

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = np.zeros_like(self.init_qvel)
        min_a, max_a = 0.1, 1.2
        while True:
            a0 = self.np_random.uniform(-max_a, max_a)
            a1 = self.np_random.uniform(-max_a, max_a)
            if abs(a0) > min_a or abs(a1) > min_a:
                break
        qpos[0], qpos[1] = a0, a1

        low, high = -0.27, 0.27
        r_min, r_max = 0.1, 0.22
        while True:
            x = self.np_random.uniform(low, high)
            y = self.np_random.uniform(low, high)
            r = np.linalg.norm([x, y])
            if r_min <= r <= r_max:
                break
        qpos[self.j_target1_x], qpos[self.j_target1_y] = x, y
        self._target1_pos[:] = [x, y]

        qpos[self.j_target2_x:self.j_target2_y+1] = self.init_qpos[self.j_target2_x:self.j_target2_y+1]

        self.set_state(qpos, qvel)
        self.phase = 0
        self.current_step = 0
        self.prev_dist1 = None
        self.prev_dist2 = None
        self.prev_action = None
        self.prev_ft_pos = None
        obs = self.get_obs()
        self.prev_dist1 = np.linalg.norm(obs[4:6] - obs[6:8])
        return obs
