"""
compare.py — Random agent vs REINFORCE agent comparison.

PROJECT SHEET slide 2 (mandatory):
  "evaluate whether the learned behavior improves over random play"

This is the single most important evaluation for your report.
Run AFTER train.py:
    python compare.py

Produces:
  outputs/comparison.png   — side-by-side bar chart + return distribution
  outputs/comparison.json  — raw numbers for the report table
"""

import argparse
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from envs.grid_shooter_env import GridShooterEnv
from agent.reinforce_agent import PolicyNet

NUM_EPISODES = 200   # enough for stable statistics
GRID_SIZE    = 6
MAX_STEPS    = 50
SEED         = 7777


def run_random(num_episodes, grid_size, max_steps, seed, simple_mode):
    """Baseline: purely random action selection."""
    env     = GridShooterEnv(grid_size=grid_size, max_steps=max_steps,
                             simple_mode=simple_mode)
    returns = []
    counts  = {"win": 0, "lose_shot": 0, "lose_collision": 0, "timeout": 0}

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ep_ret = False, 0.0
        while not done:
            action = env.action_space.sample()          # purely random
            obs, r, term, trunc, info = env.step(action)
            ep_ret += r
            done = term or trunc
        returns.append(ep_ret)
        counts[info.get("result", "timeout")] += 1

    return returns, counts


def run_reinforce(num_episodes, grid_size, max_steps, seed,
                  simple_mode, model_path="outputs/grid_shooter_policy.pth"):
    """Trained REINFORCE agent — greedy evaluation."""
    env = GridShooterEnv(grid_size=grid_size, max_steps=max_steps,
                         simple_mode=simple_mode)
    obs, _ = env.reset(seed=seed)
    policy  = PolicyNet(obs_dim=obs.shape[0], n_actions=env.action_space.n)

    try:
        policy.load_state_dict(torch.load(model_path, map_location="cpu"))
    except FileNotFoundError:
        print(f"ERROR: {model_path} not found. Run  python train.py  first.")
        raise

    policy.eval()
    returns = []
    counts  = {"win": 0, "lose_shot": 0, "lose_collision": 0, "timeout": 0}

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ep_ret = False, 0.0
        while not done:
            with torch.no_grad():
                s = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = int(torch.argmax(policy(s), dim=1).item())   # greedy
            obs, r, term, trunc, info = env.step(action)
            ep_ret += r
            done = term or trunc
        returns.append(ep_ret)
        counts[info.get("result", "timeout")] += 1

    return returns, counts


def make_plot(rand_returns, rl_returns, rand_counts, rl_counts, n):
    fig = plt.figure(figsize=(13, 5))
    fig.suptitle(
        "Random Agent  vs  REINFORCE Agent\n"
        f"({n} evaluation episodes each — greedy REINFORCE policy)",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(1, 2, wspace=0.35)

    RAND_C = "#e15759"    # red  for random
    RL_C   = "#59a14f"    # green for REINFORCE

    # ── Left: average return comparison ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    means  = [np.mean(rand_returns), np.mean(rl_returns)]
    stds   = [np.std(rand_returns),  np.std(rl_returns)]
    labels = ["Random agent", "REINFORCE agent"]
    colors = [RAND_C, RL_C]

    bars = ax1.bar(labels, means, yerr=stds, color=colors,
                   capsize=8, edgecolor="white", width=0.45)
    for bar, m, s in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 m + s + abs(m)*0.05 + 0.2,
                 f"{m:+.2f}\n±{s:.2f}",
                 ha="center", fontsize=9, fontweight="bold")

    ax1.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Average episode return", fontsize=10)
    ax1.set_title("Average Return\n(higher = better)", fontsize=10,
                  fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Right: outcome breakdown (stacked bar) ────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    outcomes  = ["win", "lose_shot", "lose_collision", "timeout"]
    out_cols  = ["#59a14f", "#e15759", "#f28e2b", "#bab0ac"]
    out_label = ["Win", "Lose (shot)", "Lose (collision)", "Timeout"]

    x      = np.array([0, 1])
    bottom = np.zeros(2)
    for outcome, col, lab in zip(outcomes, out_cols, out_label):
        vals = np.array([
            rand_counts[outcome] / n * 100,
            rl_counts[outcome]   / n * 100,
        ])
        ax2.bar(x, vals, bottom=bottom, color=col, label=lab, width=0.45,
                edgecolor="white")
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 3:
                ax2.text(xi, b + v/2, f"{v:.0f}%",
                         ha="center", va="center", fontsize=8,
                         color="white", fontweight="bold")
        bottom += vals

    ax2.set_xticks(x)
    ax2.set_xticklabels(["Random agent", "REINFORCE agent"])
    ax2.set_ylabel("Episode outcome (%)", fontsize=10)
    ax2.set_title("Outcome Breakdown\n(% of episodes)", fontsize=10,
                  fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: outputs/comparison.png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",    type=int,  default=NUM_EPISODES)
    p.add_argument("--grid_size",   type=int,  default=GRID_SIZE)
    p.add_argument("--max_steps",   type=int,  default=MAX_STEPS)
    p.add_argument("--simple_mode", action="store_true")
    p.add_argument("--model",       type=str,  default="outputs/grid_shooter_policy.pth")
    args = p.parse_args()

    print(f"Running {args.episodes} episodes each "
          f"({'simple' if args.simple_mode else 'full'} mode)...")

    rand_ret, rand_cnt = run_random(
        args.episodes, args.grid_size, args.max_steps, SEED, args.simple_mode)
    rl_ret, rl_cnt = run_reinforce(
        args.episodes, args.grid_size, args.max_steps, SEED,
        args.simple_mode, args.model)

    # Print table (for report)
    print("\n" + "=" * 52)
    print(f"  {'Metric':<25} {'Random':>10} {'REINFORCE':>10}")
    print("-" * 52)
    print(f"  {'Avg return':<25} {np.mean(rand_ret):>+10.2f} {np.mean(rl_ret):>+10.2f}")
    print(f"  {'Std return':<25} {np.std(rand_ret):>10.2f}  {np.std(rl_ret):>10.2f}")
    print(f"  {'Win rate %':<25} {rand_cnt['win']/args.episodes*100:>9.1f}% "
          f"{rl_cnt['win']/args.episodes*100:>9.1f}%")
    print(f"  {'Timeout rate %':<25} {rand_cnt['timeout']/args.episodes*100:>9.1f}% "
          f"{rl_cnt['timeout']/args.episodes*100:>9.1f}%")
    print("=" * 52)

    # Save numbers for report
    results = {
        "random":    {"returns": rand_ret, "counts": rand_cnt,
                      "mean": float(np.mean(rand_ret)),
                      "std":  float(np.std(rand_ret))},
        "reinforce": {"returns": rl_ret,   "counts": rl_cnt,
                      "mean": float(np.mean(rl_ret)),
                      "std":  float(np.std(rl_ret))},
        "n_episodes": args.episodes,
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: outputs/comparison.json")

    make_plot(rand_ret, rl_ret, rand_cnt, rl_cnt, args.episodes)

    # Key sentence for the report conclusion
    improvement = np.mean(rl_ret) - np.mean(rand_ret)
    direction   = "IMPROVES" if improvement > 0 else "does NOT improve"
    print(f"\n→ REINFORCE {direction} over random play "
          f"by {improvement:+.2f} average return.")
    print("  Use this sentence in your report conclusion.")
