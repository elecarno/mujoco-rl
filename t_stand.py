import os
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import TimeLimit

class StandUpRobotEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(StandUpRobotEnv, self).__init__()
        
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        scene_path = os.path.join(curr_dir, "robot/scene.xml")
        
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.renderer = None
        
        # Action space: -1 to 1. We will map this to the joint ranges in step()
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        
        obs_shape = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

        # Pre-identify IDs
        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.allowed_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "leg_lower_l"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "leg_lower_r")
        ]

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def step(self, action):
        # --- Action Scaling ---
        # Map PPO's -1 to 1 range into the actual joint limits defined in XML
        # This helps the agent learn much faster
        ctrlrange = self.model.actuator_ctrlrange
        denorm_action = ctrlrange[:, 0] + 0.5 * (action + 1) * (ctrlrange[:, 1] - ctrlrange[:, 0])
        self.data.ctrl[:] = denorm_action 
        
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        body_height = float(self.data.qpos[2]) 
        
        # --- Strict Failure Conditions ---
        illegal_contact = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == self.floor_id or contact.geom2 == self.floor_id:
                g_id = contact.geom1 if contact.geom2 == self.floor_id else contact.geom2
                if self.model.geom_bodyid[g_id] not in self.allowed_body_ids:
                    illegal_contact = True
                    break

        # NEW: Fail if too high (prevents escaping simulation) or too low
        out_of_bounds = bool(body_height < 0.1 or body_height > 0.3)
        terminated = bool(illegal_contact or out_of_bounds)
        
        # --- Reward Engineering ---
        target_h = 0.24
        # 1. Precise Height Reward (using a narrower curve)
        height_reward = np.exp(-100 * (body_height - target_h)**2)
        
        # 2. Uprightness (Body should be oriented vertically)
        upright_reward = float(self.data.qpos[3]) # w-quat
        
        # 3. Survival Bonus (encourages staying alive)
        survival_reward = 0.5 if not terminated else 0.0
        
        # 4. Energy/Stability Penalty (prevents jittering/shaking)
        effort_penalty = -0.01 * np.sum(np.square(action))
        velocity_penalty = -0.001 * np.sum(np.square(self.data.qvel))
        
        reward = float(height_reward + (0.5 * upright_reward) + survival_reward + effort_penalty + velocity_penalty)
        
        if terminated:
            reward -= 50.0 # Heavy penalty for failing early

        return obs, reward, terminated, False, {"height": body_height}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Start at target height to show it what success looks like
        self.data.qpos[2] = 0.26
        # Narrower noise for stability
        self.data.qpos[7:] += np.random.uniform(-0.02, 0.02, size=self.model.nq - 7)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            if self.renderer is None:
                from mujoco import viewer
                self.renderer = viewer.launch_passive(self.model, self.data)
            self.renderer.sync()

def train():
    num_envs = 8 # Ryzen 9 6000 is great for 8 envs
    env = make_vec_env(lambda: TimeLimit(StandUpRobotEnv(), 1000), n_envs=num_envs)
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4, # Standard PPO rate
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01, # Keep exploration up
        tensorboard_log="./ppo_robot_tensorboard/",
        device="cpu"
    )
    
    print("Training with Strict Height Bounds and Survival Bonus...")
    model.learn(total_timesteps=150000)
    model.save("robot_stand_up_model")

def visualize(only_final=False):
    base_env = StandUpRobotEnv(render_mode="human")
    env = TimeLimit(base_env, max_episode_steps=1000)
    model = PPO.load("robot_stand_up_model")
    obs, _ = env.reset()
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                if only_final:
                    while True: env.render(); time.sleep(0.1)
                time.sleep(0.5)
                obs, _ = env.reset()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v", action="store_true")
    parser.add_argument("--f", action="store_true")
    args = parser.parse_args()
    if args.v or args.f:
        visualize(only_final=args.f)
    else:
        train()