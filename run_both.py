"""
Run both DQN and PG baselines with a shared step budget.
Usage:
  python run_both.py --steps 5000000 --n-envs 8
"""

import argparse

from dqn_agent import train_dqn_pong
from pg_baseline import PGConfig, train_pg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DQN and PG baselines")
    parser.add_argument("--steps", type=int, default=5_000_000, help="Total steps for each agent")
    parser.add_argument("--n-envs", type=int, default=16, help="Number of parallel envs for DQN")
    args = parser.parse_args()

    print(f"=== Running DQN for {args.steps} steps with {args.n_envs} envs ===")
    train_dqn_pong(total_timesteps=args.steps, n_envs=args.n_envs)

    print(f"=== Running PG for {args.steps} steps (single env) ===")
    pg_cfg = PGConfig(
        max_steps=args.steps,
        episodes=10_000_000,  # large upper bound; max_steps will stop earlier
    )
    train_pg(pg_cfg)


if __name__ == "__main__":
    main()
