"""
Training script — REINFORCE on Grid Shooter.
Mirrors PDF 3 slide 16 (training loop) + slide 18 (evaluation metrics).

What you will see in the terminal during training:
  - Every LOG_EVERY episodes: return, moving average (last 50), win rate
  - A final evaluation table after training

What gets saved:
  - outputs/grid_shooter_policy.pth      trained policy weights
  - outputs/training_history.json        raw data for plotting (run plot.py after)

Usage:
  python train.py                         # default 2000 episodes
  python train.py --episodes 3000         # longer run
  python train.py --episodes 500 --lr 5e-4 --seed 1   # quick test
"""

import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch

from envs.grid_shooter_env import GridShooterEnv
from agent.reinforce_agent import (
    PolicyNet, collect_episode, compute_returns, reinforce_loss
)

# ── terminal helpers ──────────────────────────────────────────────────────────

def _bar(ep, total, width=20):
    n = int(ep / total * width)
    return "[" + "=" * n + ">" + " " * (width - n - 1) + "]"


def _print_header(obs_dim, n_actions, gamma, lr, num_episodes):
    print()
    print("=" * 75)
    print("  REINFORCE — Grid Shooter Training")
    print("  Lab ref: PDF 3  collect(slide13) → returns(14) → loss(15) → update(16)")
    print("=" * 75)
    print(f"  obs_dim={obs_dim}  n_actions={n_actions}  "
          f"gamma={gamma}  lr={lr}  episodes={num_episodes}")
    print()
    print(f"  {'Episode':>8}  {'Return':>8}  {'Avg-50':>8}  "
          f"{'WinRate':>8}  {'Last result':<16}  Progress")
    print("-" * 75)


def _row(ep, total, ep_return, avg50, win_rate, result):
    bar = _bar(ep, total)
    print(f"  {ep:>8d}  {ep_return:>+8.2f}  {avg50:>+8.2f}  "
          f"{win_rate:>7.1f}%  {result:<16}  {bar}")
    sys.stdout.flush()


# ── training ──────────────────────────────────────────────────────────────────

def train(
    num_episodes=2000,
    gamma=0.99,
    lr=1e-3,
    grid_size=6,
    max_steps=50,
    log_every=50,
    seed=0,
    simple_mode=False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = GridShooterEnv(grid_size=grid_size, max_steps=max_steps,
                         simple_mode=simple_mode)
    obs, _ = env.reset(seed=seed)
    obs_dim, n_actions = obs.shape[0], env.action_space.n

    policy    = PolicyNet(obs_dim=obs_dim, n_actions=n_actions)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    _print_header(obs_dim, n_actions, gamma, lr, num_episodes)

    window50 = deque(maxlen=50)
    win_count   = 0
    t0          = time.time()

    history = {
        "episode":      [],
        "return":       [],
        "avg50":        [],
        "win_rate_pct": [],
    }

    for ep in range(1, num_episodes + 1):

        # 1. Collect episode (slide 13)
        rewards, log_probs, info = collect_episode(env, policy)

        # 2. Discounted returns (slide 14)
        returns = compute_returns(rewards, gamma=gamma)

        # 3. REINFORCE loss (slide 15)
        loss = reinforce_loss(log_probs, returns, normalize=True)

        # 4. Gradient update (slide 16)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ep_return = float(sum(rewards))
        window50.append(ep_return)
        result = info.get("result", "timeout")
        if result == "win":
            win_count += 1

        avg50    = float(np.mean(window50))
        win_rate = win_count / ep * 100.0

        if ep % log_every == 0 or ep == 1:
            _row(ep, num_episodes, ep_return, avg50, win_rate, result)
            history["episode"].append(ep)
            history["return"].append(ep_return)
            history["avg50"].append(avg50)
            history["win_rate_pct"].append(win_rate)

    elapsed = time.time() - t0
    print("-" * 75)
    print(f"  Done in {elapsed:.0f}s — total wins: {win_count} "
          f"({win_count/num_episodes*100:.1f}%)")

    os.makedirs("outputs", exist_ok=True)
    torch.save(policy.state_dict(), "outputs/grid_shooter_policy.pth")
    with open("outputs/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("  Saved: outputs/grid_shooter_policy.pth  outputs/training_history.json")
    print("  Now run:  python plot.py   to generate figures")

    return policy, history


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(policy, num_episodes=100, grid_size=6, max_steps=50,
             seed=9999, simple_mode=False):
    """Greedy evaluation — argmax policy, 100 episodes."""
    env    = GridShooterEnv(grid_size=grid_size, max_steps=max_steps,
                            simple_mode=simple_mode)
    counts = {"win": 0, "lose_shot": 0, "lose_collision": 0, "timeout": 0}
    rets   = []

    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed + i)
        done, ep_ret = False, 0.0
        while not done:
            with torch.no_grad():
                s = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = int(torch.argmax(policy(s), dim=1).item())
            obs, r, term, trunc, info = env.step(action)
            ep_ret += r
            done = term or trunc
        rets.append(ep_ret)
        counts[info.get("result", "timeout")] += 1

    print()
    print("=" * 52)
    print(f"  Evaluation — {num_episodes} greedy episodes")
    print("=" * 52)
    for k, v in counts.items():
        bar = "█" * int(v / num_episodes * 30)
        print(f"  {k:<18s} {v:>4d} ({v/num_episodes*100:5.1f}%)  {bar}")
    print(f"\n  Average return : {np.mean(rets):+.2f}")
    print(f"  Std of return  : {np.std(rets):.2f}")
    print("=" * 52)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",  type=int,   default=2000)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--gamma",     type=float, default=0.99)
    p.add_argument("--grid_size", type=int,   default=6)
    p.add_argument("--max_steps", type=int,   default=50)
    p.add_argument("--log_every",   type=int,            default=50)
    p.add_argument("--seed",        type=int,            default=0)
    p.add_argument("--simple_mode", action="store_true",
                   help="Static enemy, no enemy bullets — faster convergence")
    args = p.parse_args()

    policy, history = train(
        num_episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        log_every=args.log_every,
        seed=args.seed,
        simple_mode=args.simple_mode,
    )
    evaluate(policy, grid_size=args.grid_size, max_steps=args.max_steps,
             simple_mode=args.simple_mode)
