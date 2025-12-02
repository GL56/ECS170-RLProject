# sarsa_pong.py
import os
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional
import gymnasium as gym
import ale_py # <--- This is where the magic happens, registering the environments

# --- register ALE Atari environments ---
#egister_envs(ale_gym)
# --------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------

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
# Hyperparameters
# ----------------------------

@dataclass
class Config:
    env_id: str = "ALE/Pong-v5"
    seed: int = 0
    total_episodes: int = 500  # Pong is slow; bump higher for better play
    max_steps_per_ep: int = 5000
    gamma: float = 0.99
    learning_rate: float = 1e-4
    lr_decay_gamma: float = 1.0        # < 1.0 enables exponential LR decay
    lr_decay_frequency: int = 1        # step scheduler every N episodes
    frame_stack: int = 4
    frameskip: int = 4
    repeat_action_probability: float = 0.0
    full_action_space: bool = False
    obs_type: str = "rgb"
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 300  # linear decay over episodes
    eps_decay_strategy: str = "linear"  # "linear" or "exp"
    eps_decay_rate: float = 0.995       # used when strategy == "exp"
    grad_clip: float = 5.0
    reward_clip: Optional[float] = 1.0
    eval_every: int = 25
    eval_episodes: int = 3
    render_eval: bool = False
    save_path: str = "sarsa_pong.pt"


# ----------------------------
# SARSA Agent
# ----------------------------

class SARSAgent:
    def __init__(self, n_actions: int, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.n_actions = n_actions
        self.net = QNet(in_channels=cfg.frame_stack, n_actions=n_actions).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        self.scheduler = None
        if 0.0 < cfg.lr_decay_gamma < 1.0:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=cfg.lr_decay_gamma)

    def epsilon(self, episode_idx: int) -> float:
        strategy = self.cfg.eps_decay_strategy.lower()
        if strategy == "exp":
            decay = max(0.0, self.cfg.eps_decay_rate) ** max(0, episode_idx)
            eps = self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * decay
        else:
            decay_episodes = max(1, self.cfg.eps_decay_episodes)
            t = min(episode_idx, decay_episodes)
            eps = self.cfg.eps_start + (self.cfg.eps_end - self.cfg.eps_start) * (t / decay_episodes)
        return float(max(self.cfg.eps_end, min(self.cfg.eps_start, eps)))

    @torch.no_grad()
    def select_action(self, state_4x84x84: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.n_actions)
        s = torch.from_numpy(state_4x84x84).float().to(self.device) / 255.0
        s = s.unsqueeze(0)  # batch
        q = self.net(s)
        return int(q.argmax(dim=1).item())

    def sarsa_update(self, s, a, r, s_next, a_next, done):
        """
        One-step SARSA TD error and gradient step:
        target = r + gamma * Q(s', a') if not done else r
        """
        s_t = torch.from_numpy(s).float().to(self.device) / 255.0  # (4,84,84)
        s_t = s_t.unsqueeze(0)                                     # (1,4,84,84)
        q_values = self.net(s_t)                                   # (1, nA)
        q_sa = q_values[0, a]

        with torch.no_grad():
            if done:
                target = torch.tensor(r, dtype=torch.float32, device=self.device)
            else:
                sp_t = torch.from_numpy(s_next).float().to(self.device) / 255.0
                sp_t = sp_t.unsqueeze(0)
                q_next = self.net(sp_t)[0, a_next]
                target = torch.tensor(r, dtype=torch.float32, device=self.device) + self.cfg.gamma * q_next

        loss = F.mse_loss(q_sa, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
        self.optim.step()
        return float(loss.item())

    def maybe_step_lr_scheduler(self, episode_idx: int):
        if self.scheduler is None:
            return
        freq = max(1, self.cfg.lr_decay_frequency)
        if episode_idx % freq == 0:
            self.scheduler.step()

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))

#training loops

import gymnasium as gym
def make_env(cfg: Config, seed: int, render_mode=None):
    import gymnasium as gym
    # Build Atari Pong environment using Gymnasium
    env = gym.make(
        cfg.env_id,
        render_mode=render_mode,             # None or "human"
        frameskip=cfg.frameskip,
        repeat_action_probability=cfg.repeat_action_probability,
        full_action_space=cfg.full_action_space,
        obs_type=cfg.obs_type,
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    try:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except Exception:
        pass
    return env



def evaluate(agent: SARSAgent, episodes: Optional[int] = None, render=False, seed=42):
    env = make_env(agent.cfg, seed=seed, render_mode="human" if render else None)
    fs = FrameStack(agent.cfg.frame_stack)
    returns = []
    total_episodes = episodes if episodes is not None else agent.cfg.eval_episodes
    for _ in range(total_episodes):
        obs, _ = env.reset(seed=random.randint(0, 10_000))
        state = fs.reset(obs)
        done = False
        ep_return = 0.0
        # Greedy policy (epsilon=0)
        action = agent.select_action(state, eps=0.0)
        steps = 0
        while not done and steps < agent.cfg.max_steps_per_ep:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = fs.step(next_obs)
            next_action = agent.select_action(next_state, eps=0.0)
            ep_return += float(reward)
            state, action = next_state, next_action
            steps += 1
        returns.append(ep_return)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def train(cfg: Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(cfg, seed=cfg.seed)
    fs = FrameStack(cfg.frame_stack)

    n_actions = env.action_space.n
    agent = SARSAgent(n_actions=n_actions, cfg=cfg, device=device)

    best_eval = -1e9

    for ep in range(1, cfg.total_episodes + 1):
        eps = agent.epsilon(ep)
        obs, _ = env.reset(seed=random.randint(0, 10_000))
        state = fs.reset(obs)
        done = False
        ep_return = 0.0
        ep_loss_sum = 0.0
        steps = 0

        # choose a0 with current policy (ε-greedy)
        action = agent.select_action(state, eps)

        while not done and steps < cfg.max_steps_per_ep:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = fs.step(next_obs)

            # choose a' using ε-greedy on next_state
            next_action = agent.select_action(next_state, eps if not done else 0.0)

            # SARSA update
            reward_td = reward
            if cfg.reward_clip is not None:
                reward_td = float(np.clip(reward, -cfg.reward_clip, cfg.reward_clip))
            loss = agent.sarsa_update(state, action, reward_td, next_state, next_action, done)
            ep_loss_sum += loss
            ep_return += float(reward)

            state, action = next_state, next_action
            steps += 1

        avg_loss = ep_loss_sum / max(1, steps)
        print(f"Episode {ep:4d}/{cfg.total_episodes} | return={ep_return:6.2f} | len={steps:4d} | eps={eps:.3f} | loss={avg_loss:.5f}", flush=True)

        # Periodic evaluation
        if (ep % cfg.eval_every) == 0:
            mean_ret, std_ret = evaluate(agent, episodes=cfg.eval_episodes, render=cfg.render_eval)
            print(f"  Eval over {cfg.eval_episodes} eps: mean_return={mean_ret:.2f} ± {std_ret:.2f}")
            if mean_ret > best_eval:
                best_eval = mean_ret
                agent.save(cfg.save_path)
                print(f"  Saved checkpoint to {cfg.save_path}")

        agent.maybe_step_lr_scheduler(ep)

    env.close()
    # Final save
    agent.save(cfg.save_path)
    print(f"Training complete. Model saved to {cfg.save_path}")

    # Final eval
    final_eps = max(5, cfg.eval_episodes)
    mean_ret, std_ret = evaluate(agent, episodes=final_eps, render=cfg.render_eval)
    print(f"Final eval over {final_eps} eps: mean_return={mean_ret:.2f} ± {std_ret:.2f}")


if __name__ == "__main__":
    cfg = Config(
        total_episodes=500,  # increase for better play (e.g., 2,000+)
        eval_every=25,
        eval_episodes=5,
        eps_decay_strategy="exp",
        eps_decay_rate=0.997,
        lr_decay_gamma=0.9995,
        render_eval=False,
    )
    train(cfg)
