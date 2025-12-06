import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder, DummyVecEnv
import os

def record_video(model_path="models/dqn_pong_final.zip", video_length=1500, prefix="dqn-pong"):
    """
    Records a video of the trained agent playing Pong.
    """
    print(f"Loading model from {model_path}...")
    
    # 1. Create the environment exactly as we did during training
    # We use DummyVecEnv because we only need 1 environment for recording
    # and we want to capture the frames easily.
    env_id = "ALE/Pong-v5"
    env = make_atari_env(
        env_id,
        n_envs=1,
        seed=42,
        env_kwargs={
            "repeat_action_probability": 0.0,
            "full_action_space": False,
            "obs_type": "rgb", # Must match training
            "render_mode": "rgb_array" # Crucial for recording
        }
    )
    env = VecFrameStack(env, n_stack=4)

    # 2. Wrap it in a VideoRecorder
    video_folder = "videos/"
    os.makedirs(video_folder, exist_ok=True)
    
    env = VecVideoRecorder(
        env,
        video_folder,
        record_video_trigger=lambda x: x == 0, # Record starting from step 0
        video_length=video_length,
        name_prefix=prefix
    )

    # 3. Load the model
    model = DQN.load(model_path, env=env)

    # 4. Run the agent
    obs = env.reset()
    print("Recording video...")
    for _ in range(video_length + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)

    # 5. Close everything
    env.close()
    print(f"Video saved to {video_folder}")

if __name__ == "__main__":
    record_video()
