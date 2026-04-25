import os
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import time
from stable_baselines3 import PPO
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
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        obs_shape = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

        # Pre-identify geom IDs for performance
        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        
        # Lower leg meshes are the only allowed contacts
        self.allowed_geoms = []
        for name in ["leg_lower_l", "leg_lower_r"]:
            g_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if g_id != -1: self.allowed_geoms.append(g_id)

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def step(self, action):
        self.data.ctrl[:] = action 
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        body_height = float(self.data.qpos[2]) 
        
        # --- Strict Contact Check ---
        illegal_contact = False
        # Get IDs for the allowed bodies (the legs)
        allowed_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "leg_lower_l"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "leg_lower_r")
        ]

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # If one geom is the floor...
            if contact.geom1 == self.floor_id or contact.geom2 == self.floor_id:
                # Find which body that geom belongs to
                g_id = contact.geom1 if contact.geom2 == self.floor_id else contact.geom2
                body_id = self.model.geom_bodyid[g_id]
                
                # If the body is NOT one of the lower legs, it's an illegal touch
                if body_id not in allowed_body_ids:
                    illegal_contact = True
                    break

        # --- Reward and Termination ---
        target_height = 0.25
        height_reward = np.exp(-40 * (body_height - target_height)**2)
        upright_reward = float(self.data.qpos[3]) # w-quat
        
        # Fail if illegal contact OR if body is impossibly low
        terminated = bool(illegal_contact or body_height < 0.05)
        
        fall_penalty = -20.0 if terminated else 0.0
        reward = float(height_reward + (0.5 * upright_reward) + fall_penalty)
        
        return obs, reward, terminated, False, {"height": body_height}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Move the root body (qpos[0,1,2] is X,Y,Z) to be above the ground
        self.data.qpos[0] = 0.0  # X
        self.data.qpos[1] = 0.0  # Y
        self.data.qpos[2] = 0.28  # Z (20cm above ground)
        
        # Reset orientation (quat) to identity [1 0 0 0]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        
        # Add slight noise to joint positions (starting from index 7)
        self.data.qpos[7:] += np.random.uniform(-0.05, 0.05, size=self.model.nq - 7)
        
        # If using the strict collision logic, we must step once to 
        # ensure contacts are updated before the first observation
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            if self.renderer is None:
                from mujoco import viewer
                self.renderer = viewer.launch_passive(self.model, self.data)
            self.renderer.sync()

def train():
    base_env = StandUpRobotEnv()
    env = TimeLimit(base_env, max_episode_steps=1000) 
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_robot_tensorboard/")
    print("Training with Strict Collision Rules...")
    model.learn(total_timesteps=500000)
    model.save("robot_stand_up_model")

def visualize(only_final=False):
    # Use the same TimeLimit wrapper so 'truncated' works correctly
    base_env = StandUpRobotEnv(render_mode="human")
    env = TimeLimit(base_env, max_episode_steps=1000)
    model = PPO.load("robot_stand_up_model")
    
    obs, _ = env.reset()
    print("Visualizing... Press Ctrl+C to stop.")
    
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if terminated or truncated:
                if only_final:
                    print(f"Final Position reached. Height: {info['height']:.4f}m. Paused.")
                    while True:
                        env.render()
                        time.sleep(0.1)
                
                print("Fail state or Timeout reached. Resetting episode...")
                time.sleep(0.5) # Brief pause so you can see the fail
                obs, _ = env.reset()
            
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Closing visualizer.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v", action="store_true", help="Run looping visualizer")
    parser.add_argument("--final", action="store_true", help="View final standing pose and pause")
    args = parser.parse_args()

    if args.v or args.final:
        visualize(only_final=args.final)
    else:
        train()