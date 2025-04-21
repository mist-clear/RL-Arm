import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv

class CustomReacherEnv(MujocoEnv, gym.utils.EzPickle):
    def __init__(self, render_mode=None, only_first_phase=True):
        self.only_first_phase = only_first_phase
        xml_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "custom_reacher.xml")
        )
        self.frame_skip = 2

        self.observation_space = Box(-np.inf, np.inf, (10,), np.float32)
        self.action_space      = Box(-0.5, 0.5,   (2,), np.float32)

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
        self.max_steps = 500

        # 只查找 site id
        self.s_fingertip = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,  "fingertip")
        self.s_object    = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,  "object")
        self.s_target    = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,  "target")

    def get_obs(self):
        qpos      = self.data.qpos[:2].copy()
        qvel      = self.data.qvel[:2].copy()
        fingertip = self.data.site_xpos[self.s_fingertip][:2].copy()
        obj       = self.data.site_xpos[self.s_object][:2].copy()
        tgt       = self.data.site_xpos[self.s_target][:2].copy()
        return np.concatenate([qpos, qvel, fingertip, obj, tgt])

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        self.do_simulation(action, self.frame_skip)
        obs = self.get_obs()
        self.current_step += 1

        fingertip = obs[4:6]
        done = False

        if self.phase == 0:
            dist_obj = np.linalg.norm(fingertip - obs[6:8])
            reward = -10.0 * dist_obj
            if dist_obj < 0.05:
                reward += 20.0
                if self.only_first_phase:
                    done = True
                    return obs, reward, done, False, {}
                self.phase = 1
        else:
            dist_tgt = np.linalg.norm(fingertip - obs[8:10])
            reward = -10.0 * dist_tgt
            if dist_tgt < 0.05:
                reward += 20.0
                done = True

        reward -= 0.005 * np.sum(np.square(action))
        if self.current_step >= self.max_steps:
            reward -= 5.0
            done = True

        return obs, reward, done, False, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(-0.1, 0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(-0.1, 0.1, size=self.model.nv)
        max_radius = 0.18

        # 随机生成 object
        r1 = self.np_random.uniform(0.05, max_radius)
        theta1 = self.np_random.uniform(0, 2 * np.pi)
        obj_x = r1 * np.cos(theta1)
        obj_y = r1 * np.sin(theta1)

        # 随机生成 target，确保和object距离不太近
        while True:
            r2 = self.np_random.uniform(0.05, max_radius)
            theta2 = self.np_random.uniform(0, 2 * np.pi)
            tgt_x = r2 * np.cos(theta2)
            tgt_y = r2 * np.sin(theta2)
            if np.linalg.norm([tgt_x - obj_x, tgt_y - obj_y]) > 0.05:
                break

        # 直接设置 site 的位置
        self.model.site_pos[self.s_object][:2] = [obj_x, obj_y]
        self.model.site_pos[self.s_object][2] = 0.01
        self.model.site_pos[self.s_target][:2] = [tgt_x, tgt_y]
        self.model.site_pos[self.s_target][2] = 0.01
        self.set_state(qpos, qvel)
        self.phase = 0
        self.current_step = 0
        return self.get_obs()
