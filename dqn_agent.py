# dqn_pong.py
import os
import math
import random
from collections import deque, namedtuple
from dataclasses import dataclass
import gymnasium as gym
import ale_py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Utilities / preprocessing
# ----------------------------

def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    """
    Convert RGB (210x160x3) to grayscale (84x84), uint8.
    No external deps (uses PyTorch interpolate under the hood).
    """
    # obs: H x W x C (uint8)
    frame = torch.from_numpy(obs).float() / 255.0           # (H, W, C) in [0,1]
    # luminance to grayscale
    r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b               # (H, W)
    # to NCHW for interpolate
    gray = gray.unsqueeze(0).unsqueeze(0)                   # (1,1,H,W)
    gray84 = F.interpolate(gray, size=(84, 84), mode="bilinear", align_corners=False)
    gray84 = (gray84.squeeze().clamp(0, 1) * 255.0).byte().numpy()  # (84,84) uint8
    return gray84


class FrameStack:
    def __init__(self, k: int = 4):
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self, first_frame: np.ndarray) -> np.ndarray:
        self.frames.clear()
        f = preprocess_obs(first_frame)
        for _ in range(self.k):
            self.frames.append(f)
        return self._get()

    def step(self, frame: np.ndarray) -> np.ndarray:
        f = preprocess_obs(frame)
        self.frames.append(f)
        return self._get()

    def _get(self) -> np.ndarray:
        # shape: (k, 84, 84), uint8
        return np.stack(list(self.frames), axis=0)


# ----------------------------
# Q-network (DQN-style head)
# ----------------------------

class QNet(nn.Module):
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        # Classic DQN-ish torso
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # compute linear input size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            x = self._forward_conv(dummy)
            n_flat = x.view(1, -1).size(1)
        self.fc1 = nn.Linear(n_flat, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Q-values


# ----------------------------
# Replay Buffer
# ----------------------------

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        """Save a transition"""
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


# ----------------------------
# Hyperparameters
# ----------------------------

@dataclass
class Config:
    env_id: str = "ALE/Pong-v5"
    seed: int = 0
    total_episodes: int = 100 #500
    max_steps_per_ep: int = 5000
    gamma: float = 0.99
    learning_rate: float = 1e-4
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 300
    grad_clip: float = 5.0
    eval_every: int = 25
    render_eval: bool = False
    save_path: str = "dqn_pong.pt"
    
    # DQN specific parameters
    replay_buffer_size: int = 10000
    batch_size: int = 32
    target_update_freq: int = 100  # steps
    learning_starts: int = 1000    # steps before learning begins


# ----------------------------
# DQN Agent
# ----------------------------

class DQNAgent:
    def __init__(self, n_actions: int, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.n_actions = n_actions
        
        # Policy network and target network
        self.policy_net = QNet(in_channels=4, n_actions=n_actions).to(device)
        self.target_net = QNet(in_channels=4, n_actions=n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network doesn't train
        
        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)
        self.replay_buffer = ReplayBuffer(cfg.replay_buffer_size)
        self.steps_done = 0

    def epsilon(self, episode_idx: int) -> float:
        # linear decay
        t = min(episode_idx, self.cfg.eps_decay_episodes)
        eps = self.cfg.eps_start + (self.cfg.eps_end - self.cfg.eps_start) * (t / self.cfg.eps_decay_episodes)
        return float(eps)

    @torch.no_grad()
    def select_action(self, state_4x84x84: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.n_actions)
        s = torch.from_numpy(state_4x84x84).float().to(self.device) / 255.0
        s = s.unsqueeze(0)  # batch
        q = self.policy_net(s)
        return int(q.argmax(dim=1).item())

    def update(self):
        """Perform one DQN update using a batch from replay buffer"""
        if len(self.replay_buffer) < self.cfg.batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        transitions = self.replay_buffer.sample(self.cfg.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.from_numpy(np.array(batch.state)).float().to(self.device) / 255.0
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_state_batch = torch.from_numpy(np.array(batch.next_state)).float().to(self.device) / 255.0
        done_batch = torch.tensor(batch.done, dtype=torch.bool, device=self.device)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states using target network
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        
        # Compute the expected Q values
        expected_state_action_values = reward_batch + (self.cfg.gamma * next_state_values * ~done_batch)
        
        # Compute loss
        loss = F.mse_loss(state_action_values.squeeze(), expected_state_action_values)
        
        # Optimize
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.grad_clip)
        self.optim.step()
        
        return loss.item()

    def update_target_network(self):
        """Update the target network to match the policy network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])


# ----------------------------
# Training / Evaluation loops
# ----------------------------

def make_env(_env_id: str, seed: int, render_mode=None):
    env = gym.make(
        "ALE/Pong-v5",
        render_mode=render_mode,
        frameskip=4,
        repeat_action_probability=0.0,
        full_action_space=False,
        obs_type="rgb",
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    try:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except Exception:
        pass
    return env

def evaluate(agent: DQNAgent, episodes=3, render=False, seed=42):
    env = make_env(agent.cfg.env_id, seed=seed, render_mode="human" if render else None)
    fs = FrameStack(4)
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset(seed=random.randint(0, 10_000))
        state = fs.reset(obs)
        done = False
        ep_return = 0.0
        steps = 0
        while not done and steps < agent.cfg.max_steps_per_ep:
            action = agent.select_action(state, eps=0.0)  # Greedy policy
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = fs.step(next_obs)
            ep_return += float(reward)
            state = next_state
            steps += 1
        returns.append(ep_return)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))

def train(cfg: Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(cfg.env_id, seed=cfg.seed)
    fs = FrameStack(4)

    n_actions = env.action_space.n
    agent = DQNAgent(n_actions=n_actions, cfg=cfg, device=device)

    best_eval = -1e9
    total_steps = 0

    for ep in range(1, cfg.total_episodes + 1):
        eps = agent.epsilon(ep)
        obs, _ = env.reset(seed=random.randint(0, 10_000))
        state = fs.reset(obs)
        done = False
        ep_return = 0.0
        ep_loss_sum = 0.0
        steps = 0

        while not done and steps < cfg.max_steps_per_ep:
            action = agent.select_action(state, eps)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = fs.step(next_obs)

            # Store transition in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            ep_return += float(reward)
            steps += 1
            total_steps += 1

            # Learn if we have enough samples
            if total_steps >= cfg.learning_starts:
                loss = agent.update()
                ep_loss_sum += loss if loss else 0.0

                # Update target network periodically
                if total_steps % cfg.target_update_freq == 0:
                    agent.update_target_network()

            state = next_state

        avg_loss = ep_loss_sum / max(1, steps) if total_steps >= cfg.learning_starts else 0.0
        print(f"Episode {ep:4d}/{cfg.total_episodes} | return={ep_return:6.2f} | len={steps:4d} | eps={eps:.3f} | loss={avg_loss:.5f} | steps={total_steps}")

        # Periodic evaluation
        if (ep % cfg.eval_every) == 0:
            mean_ret, std_ret = evaluate(agent, episodes=3, render=cfg.render_eval)
            print(f"  Eval over 3 eps: mean_return={mean_ret:.2f} ± {std_ret:.2f}")
            if mean_ret > best_eval:
                best_eval = mean_ret
                agent.save(cfg.save_path)
                print(f"  Saved checkpoint to {cfg.save_path}")

    env.close()
    # Final save
    agent.save(cfg.save_path)
    print(f"Training complete. Model saved to {cfg.save_path}")

    # Final eval
    mean_ret, std_ret = evaluate(agent, episodes=5, render=cfg.render_eval)
    print(f"Final eval over 5 eps: mean_return={mean_ret:.2f} ± {std_ret:.2f}")

if __name__ == "__main__":
    cfg = Config(
        total_episodes=100, #500
        eval_every=25,
        render_eval=False,
        replay_buffer_size=10000,
        batch_size=32,
        target_update_freq=100,
        learning_starts=1000,
    )
    train(cfg)