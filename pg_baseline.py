import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------------
# Preprocessing (Karpathy-style)
# ----------------------------

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    processed = frame[35:195]
    processed = processed[::2, ::2, 0]
    processed = processed.astype(np.float32)
    processed[processed == 144.0] = 0.0
    processed[processed == 109.0] = 0.0
    processed[processed != 0.0] = 1.0
    return processed


def frame_diff(curr: np.ndarray, prev: Optional[np.ndarray]) -> np.ndarray:
    if prev is None:
        return np.zeros_like(curr, dtype=np.float32).reshape(-1)
    return (curr.astype(np.float32) - prev.astype(np.float32)).reshape(-1)


def discount_rewards(rewards: Sequence[float], gamma: float) -> np.ndarray:
    discounted = np.zeros(len(rewards), dtype=np.float32)
    running_total = 0.0
    for t in reversed(range(len(rewards))):
        if rewards[t] != 0:
            running_total = 0.0
        running_total = running_total * gamma + rewards[t]
        discounted[t] = running_total
    mean = discounted.mean()
    std = discounted.std()
    return (discounted - mean) / (std + 1e-6)


# ----------------------------
# Policy network
# ----------------------------


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 200) -> None:
        super().__init__()
        self.fc1 = nn.Linear(80 * 80, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


# ----------------------------
# Config / Env helpers
# ----------------------------


@dataclass
class PGConfig:
    episodes: int = 200
    batch_size: int = 10
    lr: float = 1e-3
    gamma: float = 0.99
    decay: float = 0.99
    seed: int = 0
    checkpoint: Path = Path("pg_baseline.pth")
    log_path: Path = Path("pg_raw_training_log.txt")


def make_env(render: Optional[str] = None) -> gym.Env:
    # use ALE defaults similar to DQN code
    env = gym.make(
        "ALE/Pong-v5",
        render_mode=render,
        frameskip=4,
        repeat_action_probability=0.0,
        full_action_space=False,
        obs_type="rgb",
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Training (logs compatible with DQN format)
# ----------------------------


def train_pg(cfg: PGConfig) -> List[float]:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = _device()
    env = make_env()
    policy = PolicyNet().to(device)
    optimizer = optim.RMSprop(policy.parameters(), lr=cfg.lr, alpha=cfg.decay)

    start_ep = 0
    reward_history: List[float] = []
    if cfg.checkpoint.exists():
        ckpt = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt.get("model_state_dict", policy.state_dict()))
        optimizer.load_state_dict(ckpt.get("optimizer_state_dict", optimizer.state_dict()))
        start_ep = int(ckpt.get("episode", 0))
        reward_history = list(ckpt.get("reward_history", []))

    log_f = cfg.log_path.open("w")
    print("A.L.E: Arcade Learning Environment (policy gradient baseline)", file=log_f)
    print("[Powered by PyTorch]", file=log_f)
    log_f.flush()

    observation, _ = env.reset(seed=cfg.seed)
    prev_proc: Optional[np.ndarray] = None

    batch_states: List[np.ndarray] = []
    batch_actions: List[int] = []
    batch_rewards: List[float] = []

    total_steps = 0

    for ep in range(start_ep + 1, cfg.episodes + 1):
        done = False
        prev_proc = None
        episode_states: List[np.ndarray] = []
        episode_actions: List[int] = []
        episode_rewards: List[float] = []
        steps = 0

        while not done:
            proc = preprocess_frame(observation)
            diff = frame_diff(proc, prev_proc)
            prev_proc = proc

            state_t = torch.from_numpy(diff.reshape(1, -1)).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                p_up = float(policy(state_t).item())
            action = 2 if np.random.rand() < p_up else 3
            episode_states.append(diff)
            episode_actions.append(0 if action == 2 else 1)

            observation, reward, terminated, truncated, _ = env.step(action)
            episode_rewards.append(float(reward))
            done = terminated or truncated
            steps += 1
            total_steps += 1

        reward_sum = float(sum(episode_rewards))
        reward_history.append(reward_sum)
        batch_states.extend(episode_states)
        batch_actions.extend(episode_actions)
        batch_rewards.extend(episode_rewards)

        # Compute a per-episode proxy loss value for logging (not exact training loss)
        with torch.no_grad():
            if episode_states:
                st = torch.from_numpy(np.stack(episode_states)).to(device=device, dtype=torch.float32)
                probs = torch.clamp(policy(st).squeeze(1), 1e-6, 1 - 1e-6)
                acts = torch.tensor(episode_actions, device=device)
                rets = torch.from_numpy(discount_rewards(episode_rewards, cfg.gamma)).to(device)
                lp = torch.where(acts == 0, torch.log(probs), torch.log1p(-probs))
                ep_loss = float((-(lp * rets)).mean().item())
            else:
                ep_loss = 0.0

        print(
            f"Episode {ep:4d}/{cfg.episodes} | return={reward_sum:6.2f} | len={steps:4d} | loss={ep_loss:.5f} | steps={total_steps}",
            file=log_f,
            flush=True,
        )

        observation, _ = env.reset()

        # Batch update
        if (ep % cfg.batch_size) == 0 or ep == cfg.episodes:
            states_t = torch.from_numpy(np.stack(batch_states)).to(device=device, dtype=torch.float32)
            acts_t = torch.from_numpy(np.array(batch_actions, dtype=np.int64)).to(device)
            rets_t = torch.from_numpy(discount_rewards(batch_rewards, cfg.gamma)).to(device)

            probs = torch.clamp(policy(states_t).squeeze(1), 1e-6, 1 - 1e-6)
            log_probs = torch.where(acts_t == 0, torch.log(probs), torch.log1p(-probs))
            loss = -(log_probs * rets_t).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_states.clear()
            batch_actions.clear()
            batch_rewards.clear()

        # Periodic lightweight checkpoint
        if (ep % max(50, cfg.batch_size)) == 0 or ep == cfg.episodes:
            torch.save(
                {
                    "episode": ep,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "reward_history": reward_history,
                },
                cfg.checkpoint,
            )

    env.close()
    log_f.write("Training complete.\n")
    log_f.close()
    return reward_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy Gradient baseline for Pong")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--checkpoint", type=str, default="pg_baseline.pth")
    parser.add_argument("--log", type=str, default="pg_raw_training_log.txt")
    args = parser.parse_args()

    cfg = PGConfig(
        episodes=args.episodes,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        seed=args.seed,
        checkpoint=Path(args.checkpoint),
        log_path=Path(args.log),
    )
    train_pg(cfg)


if __name__ == "__main__":
    main()

