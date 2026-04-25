# single_script.py
import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import time
import sys

# --- ENVIRONMENT CLASS ---
class BipedalStandEnv(gym.Env):
    def __init__(self, xml_path='robot/scene.xml', max_episode_steps=500):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.max_episode_steps = max_episode_steps
        self.dt = self.model.opt.timestep
        self.action_space = spaces.Box(low=-100.0, high=100.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.model.nq + self.model.nv,), dtype=np.float32)
        self._episode_step = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_step = 0
        mujoco.mj_resetData(self.model, self.data)
        self.data.qvel[:] = 0
        return self._get_obs(), {}
    
    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -100, 100)
        mujoco.mj_step(self.model, self.data)
        self._episode_step += 1
        reward = self._compute_reward()
        terminated = self._check_termination()
        truncated = self._episode_step >= self.max_episode_steps
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _get_obs(self):
        return np.concatenate([self.data.qpos[:], self.data.qvel[:]]).astype(np.float32)
    
    def _compute_reward(self):
        torso_height = self.data.qpos[2]
        target_height = 0.8
        height_reward = min(torso_height / target_height, 1.0) * 1.0
        quat = self.data.qpos[3:7]
        upright_reward = 1.0 - abs(quat[2])
        vel_penalty = -0.01 * np.sum(self.data.qvel ** 2)
        energy_penalty = -0.0001 * np.sum(self.data.ctrl ** 2)
        alive_bonus = 0.1
        return height_reward + upright_reward + vel_penalty + energy_penalty + alive_bonus
    
    def _check_termination(self):
        torso_height = self.data.qpos[2]
        if torso_height < 0.3:
            return True
        quat = self.data.qpos[3:7]
        if abs(quat[2]) > 0.7:
            return True
        return False

# --- TRAINING FUNCTION ---
def train():
    env = DummyVecEnv([lambda: BipedalStandEnv(xml_path='robot/scene.xml')])
    model = PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, verbose=1)
    model.learn(total_timesteps=100_000)
    model.save("bipedal_stand_ppo")
    print("Training complete!")

# --- VISUALIZATION FUNCTION ---
def visualize():
    model = PPO.load("bipedal_stand_ppo")
    mujoco_model = mujoco.MjModel.from_xml_path('robot/scene.xml')
    mujoco_data = mujoco.MjData(mujoco_model)
    mujoco.mj_resetData(mujoco_model, mujoco_data)
    
    with mujoco.viewer.launch_passive(mujoco_model, mujoco_data) as viewer:
        while viewer.is_running():
            obs = np.concatenate([mujoco_data.qpos[:], mujoco_data.qvel[:]])
            action, _ = model.predict(obs, deterministic=True)
            mujoco_data.ctrl[:] = np.clip(action, -100, 100)
            mujoco.mj_step(mujoco_model, mujoco_data)
            viewer.sync()
            time.sleep(0.01)

# --- MAIN ---
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--v":
        visualize()
    else:
        train()