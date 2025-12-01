import os
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt
import numpy as np

def train_dqn_pong():
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Environment: ALE/Pong-v5
    env_id = "ALE/Pong-v5"
    
    # Create the environment
    # Note to self: Stable-Baselines3 handles frame stacking and preprocessing internally
    env = make_atari_env(
        env_id, 
        n_envs=1,  # number of parallel environments
        seed=42,
        env_kwargs={
            "frameskip": 4,
            "repeat_action_probability": 0.0,
            "full_action_space": False,
            "obs_type": "rgb",
        }
    )
    
    # Apply frame stacking (4 frames)
    env = VecFrameStack(env, n_stack=4)
    env = VecMonitor(env, filename=os.path.join('log/', "monitor")) # check if better to wrap eval_env

    # Create evaluation environment
    eval_env = make_atari_env(
        env_id,
        n_envs=1,
        seed=42,
        env_kwargs={
            "frameskip": 4,
            "repeat_action_probability": 0.0,
            "full_action_space": False,
            "obs_type": "rgb",
        }
    )
    eval_env = VecFrameStack(eval_env, n_stack=4)
    
    # Configure DQN hyperparameters
    model = DQN(
        "CnnPolicy", # as per documentation, policy class for DQN when images are input (applies to our Pong)
        env,
        learning_rate=1e-4,       # should be from 0 to 1
        buffer_size=10000,        # size of replay buffer
        learning_starts=1000,     # how many steps of model to collect transitions for before any learning starts
        batch_size=32,           # mini-batch size for each gradient update
        tau=1.0,                 # soft update coefficient - 1 for hard update
        gamma=0.99,              # discount factor
        train_freq=4,            # update the model every 4 steps
        gradient_steps=1,        # how many gradient steps after each rollout/update
        replay_buffer_class=None,  # automatically selected
        replay_buffer_kwargs=None, # keyword arguments to pass to replay buffer on creation
        optimize_memory_usage=False,  # enables memory efficient variant of replay buffer at cost of more complexity
        target_update_interval=1000,  # update target network every 1000 environment steps
        exploration_fraction=0.3,     # fraction of entire training period over which exploration rate is reduced
        exploration_initial_eps=1.0,  # initial value of random action probability
        exploration_final_eps=0.05,   # final value of random action probability
        max_grad_norm=10,            # max value for gradient clipping
        tensorboard_log=None,        # location for tensorboard is we choose to log
        policy_kwargs=None,          # any additional arguments to be passed to policy on creation
        verbose=1,                  # print info messages -> training progress
        seed=42,
        device="auto",
    )
    
    print("Model architecture:")
    print(model.policy)
    
    # Train the model
    print("Starting training...")
    total_timesteps = 50000 # increase for better agent performance later on
    
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True # shows progress of training in terminal when running script
    )
    
    # Save the final model
    model.save("models/dqn_pong_final")
    print("Training completed. Model saved.")

    # Visualize the final model
    plot_training_results()
    
    # Evaluate the trained model
    print("Evaluating the trained model...")
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=5, 
        deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Close environments
    env.close()
    eval_env.close()
    
    return model

def plot_training_results(log_folder="log/"):
    """
    Plot training results using stable-baselines3's built-in plotting tools
    https://stable-baselines3.readthedocs.io/en/master/guide/plotting.html
    """

    # sbs3 plotting: only plots reward vs timesteps/episodes/walltime
    plot_results(
        [log_folder], 
        None, 
        results_plotter.X_TIMESTEPS, 
        "DQN Pong Training"
    )
    plt.savefig('training_progress_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def load_and_test_model():
    """
    Load a trained model and test it with RENDERING
        (nice to have for later stages of agent development since this will show us how the agent is playing)
    This function is only for loading a trained model for playing/testing
    """
    env_id = "ALE/Pong-v5"
    
    # Create environment
    env = make_atari_env(
        env_id,
        n_envs=1,
        seed=42,
        env_kwargs={
            "frameskip": 4,
            "repeat_action_probability": 0.0,
            "full_action_space": False,
            "obs_type": "rgb",
        }
    )
    env = VecFrameStack(env, n_stack=4)
    
    # Load the trained model
    model_path = "models/dqn_pong_final.zip"
    if os.path.exists(model_path):
        model = DQN.load(model_path, env=env)
        print("Model loaded successfully!")
        
        # Test the model with rendering
        print("Testing model with rendering...")
        obs = env.reset()
        for i in range(1000):  # Run for 1000 steps
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            env.render()  # This will show the game
            
            if dones[0]:
                obs = env.reset()
                break
    else:
        print("No trained model found. Please train first.")
    
    env.close()

def resume_training(
    model_path="models/dqn_pong_final.zip",
    total_timesteps=50000
):
    """
    Continue training a previously saved DQN model. This is not the same as prev function to test/play
    """
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Train a model first.")
        return None

    env_id = "ALE/Pong-v5"

    # Recreates the training environment exactly the same as in train_dqn_pong()
    env = make_atari_env(
        env_id,
        n_envs=1,
        seed=42,
        env_kwargs={
            "frameskip": 4,
            "repeat_action_probability": 0.0,
            "full_action_space": False,
            "obs_type": "rgb",
        }
    )
    env = VecFrameStack(env, n_stack=4)
    env = VecMonitor(env, filename=os.path.join('log/', "monitor_resume"))

    # Load the existing model with the new env
    print(f"Loading model from '{model_path}'...")
    model = DQN.load(model_path, env=env)
    print("Model loaded, continuing training...")

    # Continue training
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )

    # Save continued model
    new_model_path = "models/dqn_pong_continued"
    model.save(new_model_path)
    print(f"Training continued. New model saved to '{new_model_path}'")

    env.close()
    return model


if __name__ == "__main__":
    # Reminder to self: need to have these packages before running script
    # uv add stable-baselines3 gymnasium[atari] ale-py torch numpy
    
    # Train the model
    model = train_dqn_pong()
    
    # Uncomment to resume training
    '''
    resume_training(
        model_path="models/dqn_pong_final.zip",
        total_timesteps=50000
    )
    '''
    
    # Uncomment to watch the trained agent play in a separate window (relevant for final versions of our agent)
    # load_and_test_model()
