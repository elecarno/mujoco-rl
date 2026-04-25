import os
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import TimeLimit

class WalkRobotEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(WalkRobotEnv, self).__init__()
        
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        # Update this to your local path for the new robot xml
        scene_path = os.path.join(curr_dir, "robot/scene.xml")
        
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.renderer = None
        
        # 6 Actuators: hip1, hip2, knee for both legs
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        
        obs_shape = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        # Allowed contacts: leg_lower_l and leg_lower_r
        self.allowed_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "leg_lower_l"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "leg_lower_r")
        ]

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def step(self, action):
        # Store previous Y position to calculate velocity/progress
        pos_before = self.data.qpos[1]
        
        # Action Scaling to XML ranges
        ctrlrange = self.model.actuator_ctrlrange
        denorm_action = ctrlrange[:, 0] + 0.5 * (action + 1) * (ctrlrange[:, 1] - ctrlrange[:, 0])
        self.data.ctrl[:] = denorm_action 
        
        mujoco.mj_step(self.model, self.data)
        
        pos_after = self.data.qpos[1]
        body_height = self.data.qpos[2]
        obs = self._get_obs()
        
        # --- Failure Conditions ---
        illegal_contact = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == self.floor_id or contact.geom2 == self.floor_id:
                g_id = contact.geom1 if contact.geom2 == self.floor_id else contact.geom2
                if self.model.geom_bodyid[g_id] not in self.allowed_body_ids:
                    illegal_contact = True
                    break

        # Bounds check: Fail if body falls below 0.18m or flips
        terminated = bool(illegal_contact or body_height < 0.18 or body_height > 0.35)
        
        # --- Walking Reward Engineering ---
        # 1. Forward Progress: Reward movement in NEGATIVE Y direction
        # (pos_before - pos_after) is positive if moving in negative Y
        progress_reward = (pos_before - pos_after) * 200.0 
        
        # 2. Healthy Height Reward: Maintain ~0.25m height
        height_reward = np.exp(-40 * (body_height - 0.25)**2)
        
        # 3. Uprightness: Using w-quaternion to ensure body is not tilted
        upright_reward = self.data.qpos[3] 
        
        # 4. Straight Line Penalty: Penalize X deviation and Yaw
        lateral_penalty = -5.0 * np.abs(self.data.qpos[0])
        
        # 5. Energy and Smoothness
        effort_penalty = -0.05 * np.sum(np.square(action))
        
        reward = progress_reward + height_reward + (0.5 * upright_reward) + lateral_penalty + effort_penalty
        
        if terminated:
            reward -= 20.0 # Penalty for falling

        return obs, float(reward), terminated, False, {"y_pos": pos_after}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Start upright and slightly above the ground
        self.data.qpos[2] = 0.25 
        # Add small noise to joints for exploration
        self.data.qpos[7:] += np.random.uniform(-0.05, 0.05, size=self.model.nq - 7)
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            if self.renderer is None:
                from mujoco import viewer
                self.renderer = viewer.launch_passive(self.model, self.data)
            self.renderer.sync()

def train():
    num_envs = 8
    # Create evaluation environment
    eval_env = make_vec_env(lambda: TimeLimit(WalkRobotEnv(), 2000), n_envs=1)
    
    # Setup callback to save the best model
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=5000,
                                 deterministic=True, render=False)

    env = make_vec_env(lambda: TimeLimit(WalkRobotEnv(), 2000), n_envs=num_envs)
    
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_walk_tensorboard/")
    
    # Include the callback in the learn call
    model.learn(total_timesteps=1000000, callback=eval_callback)
    model.save("robot_walk_final_model")

def visualize(only_final=False):
    """
    Loads and visualizes the trained agent. 
    Prioritizes the 'best_model' from the evaluation callback.
    """
    # 1. Set a global seed for NumPy to make the reset noise identical
    FIXED_SEED = 42
    np.random.seed(FIXED_SEED)

    # Initialize the environment with human rendering enabled
    base_env = WalkRobotEnv(render_mode="human")
    # Wrap with TimeLimit to match training settings
    env = TimeLimit(base_env, max_episode_steps=2000)
    
    # Path to the best model saved by EvalCallback
    best_model_path = "./logs/best_model"
    final_model_path = "robot_walk_final_model" # Note: Updated to match your model.save name

    # Attempt to load the best model
    try:
        model = PPO.load(best_model_path)
        print(f"Successfully loaded BEST model from {best_model_path}")
    except Exception:
        print(f"Best model not found at {best_model_path}, loading final model instead.")
        model = PPO.load(final_model_path)

    # 2. Reset the environment with the fixed seed
    obs, _ = env.reset(seed=FIXED_SEED)
    
    try:
        while True:
            # deterministic=True is vital for seeing the "best" behavior without noise
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the simulation
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update the viewer
            env.render()
            
            # Handle end of episode
            if terminated or truncated:
                print(f"Episode finished. Final Y-Position: {info.get('y_pos', 'N/A')}")
                
                if only_final:
                    print("Showing final state. Press Ctrl+C to exit.")
                    while True:
                        env.render()
                        time.sleep(0.1)
                
                # Brief pause before resetting
                time.sleep(0.5)
                
                # 3. Re-apply seed on reset to keep every loop identical
                np.random.seed(FIXED_SEED)
                obs, _ = env.reset(seed=FIXED_SEED)
            
            # Control simulation speed
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nVisualizer closed by user.")
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v", action="store_true")
    parser.add_argument("--f", action="store_true")
    args = parser.parse_args()
    if args.v or args.f:
        visualize(only_final=args.f)
    else:
        train()