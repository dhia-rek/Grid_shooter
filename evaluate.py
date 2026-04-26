"""
evaluate.py — Compare trained policy against random play.

Runs N greedy episodes with the saved policy and N episodes with a
random agent, then prints a summary table and saves a comparison plot.

This directly answers the project objective:
  "evaluate whether the learned behaviour improves over random play."

Usage
-----
  python evaluate.py
  python evaluate.py --episodes 200 --model outputs/zombie_policy.pth
"""

import argparse
import os

import numpy as np
import torch

from envs.grid_shooter_env import GridShooterEnv, STAGE_NAMES
from agent.reinforce_agent import PolicyNet


# ── episode runners ───────────────────────────────────────────────────────────

def run_trained(policy: PolicyNet, env: GridShooterEnv):
    """One greedy episode with the trained policy (argmax, no sampling)."""
    obs, _ = env.reset()
    done = False
    ep_return = 0.0
    steps = 0
    info = {}
    with torch.no_grad():
        while not done:
            s = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = int(torch.argmax(policy(s), dim=1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            steps += 1
    return ep_return, info.get("kills", env.kills), info.get("stage", env.stage), steps


def run_stochastic(policy: PolicyNet, env: GridShooterEnv):
    """One episode sampling from π_θ(a|s) — mirrors what happens during training."""
    obs, _ = env.reset()
    done = False
    ep_return = 0.0
    steps = 0
    info = {}
    with torch.no_grad():
        while not done:
            s    = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            dist = torch.distributions.Categorical(logits=policy(s))
            action = int(dist.sample().item())
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            steps += 1
    return ep_return, info.get("kills", env.kills), info.get("stage", env.stage), steps


def run_random(env: GridShooterEnv):
    """One episode with uniform-random actions."""
    obs, _ = env.reset()
    done = False
    ep_return = 0.0
    steps = 0
    info = {}
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_return += reward
        steps += 1
    return ep_return, info.get("kills", env.kills), info.get("stage", env.stage), steps


# ── evaluation loop ───────────────────────────────────────────────────────────

def evaluate(model_path: str, num_episodes: int):
    env = GridShooterEnv()
    obs, _ = env.reset()

    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found — run training first.")
        return

    import torch.nn as nn
    ckpt     = torch.load(model_path, map_location="cpu", weights_only=True)
    hidden   = ckpt["net.0.bias"].shape[0]
    n_linear = sum(1 for k in ckpt if k.endswith(".weight"))
    obs_dim  = obs.shape[0]
    n_act    = env.action_space.n

    policy = PolicyNet(obs_dim=obs_dim, n_actions=n_act)
    if n_linear == 2:
        policy.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, n_act))
    else:
        policy.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden),  nn.ReLU(),
                                   nn.Linear(hidden, n_act))
    policy.load_state_dict(ckpt)
    print(f"Loaded: {model_path}  (hidden={hidden}, {n_linear} linear layers)")
    policy.eval()

    print(f"Evaluating {num_episodes} episodes each — trained vs random\n")

    trained_returns,    trained_kills,    trained_stages,    trained_steps    = [], [], [], []
    stochastic_returns, stochastic_kills, stochastic_stages, stochastic_steps = [], [], [], []
    random_returns,     random_kills,     random_stages,     random_steps     = [], [], [], []

    for _ in range(num_episodes):
        r, k, s, st = run_trained(policy, env)
        trained_returns.append(r); trained_kills.append(k)
        trained_stages.append(s); trained_steps.append(st)

        r, k, s, st = run_stochastic(policy, env)
        stochastic_returns.append(r); stochastic_kills.append(k)
        stochastic_stages.append(s); stochastic_steps.append(st)

        r, k, s, st = run_random(env)
        random_returns.append(r); random_kills.append(k)
        random_stages.append(s); random_steps.append(st)

    # ── summary table ─────────────────────────────────────────────────────────
    def _stats(arr):
        a = np.array(arr, dtype=float)
        return a.mean(), a.std(), a.min(), a.max()

    tr_m, tr_s, tr_lo, tr_hi = _stats(trained_returns)
    st_m, st_s, st_lo, st_hi = _stats(stochastic_returns)
    rr_m, rr_s, rr_lo, rr_hi = _stats(random_returns)
    tk_m, tk_s, tk_lo, tk_hi = _stats(trained_kills)
    sk_m, sk_s, sk_lo, sk_hi = _stats(stochastic_kills)
    rk_m, rk_s, rk_lo, rk_hi = _stats(random_kills)

    print("=" * 76)
    print(f"{'Metric':<22} {'Greedy':>16} {'Stochastic':>16} {'Random':>16}")
    print("-" * 76)
    print(f"{'Return  mean±std':<22} {tr_m:>+8.2f}±{tr_s:<5.2f} {st_m:>+8.2f}±{st_s:<5.2f} {rr_m:>+8.2f}±{rr_s:<5.2f}")
    print(f"{'Return  min/max':<22} {tr_lo:>+7.1f}/{tr_hi:<+7.1f} {st_lo:>+7.1f}/{st_hi:<+7.1f} {rr_lo:>+7.1f}/{rr_hi:<+7.1f}")
    print(f"{'Kills   mean±std':<22} {tk_m:>8.2f}±{tk_s:<5.2f} {sk_m:>8.2f}±{sk_s:<5.2f} {rk_m:>8.2f}±{rk_s:<5.2f}")
    print(f"{'Kills   max':<22} {tk_hi:>8.0f}{'':>13} {sk_hi:>8.0f}{'':>13} {rk_hi:>8.0f}")
    print(f"{'Avg steps/episode':<22} {np.mean(trained_steps):>8.1f}{'':>13} {np.mean(stochastic_steps):>8.1f}{'':>13} {np.mean(random_steps):>8.1f}")
    print("-" * 76)
    print(f"{'vs Random (return)':<22} {tr_m - rr_m:>+8.2f}{'':>13} {st_m - rr_m:>+8.2f}")
    print(f"{'vs Random (kills)' :<22} {tk_m - rk_m:>+8.2f}{'':>13} {sk_m - rk_m:>+8.2f}")

    stage_counts = {i: trained_stages.count(i) for i in range(4)}
    print("\nGreedy agent — stage reached distribution:")
    for i, name in enumerate(STAGE_NAMES):
        pct = 100 * stage_counts[i] / num_episodes
        bar = "█" * int(pct / 2)
        print(f"  Stage {i+1} {name:<14} {stage_counts[i]:>4}× ({pct:5.1f}%)  {bar}")
    print("=" * 76)

    _save_plot(trained_returns, stochastic_returns, random_returns,
               trained_kills,   stochastic_kills,   random_kills,
               trained_stages,  num_episodes)


# ── comparison plot ───────────────────────────────────────────────────────────

def _save_plot(tr_rets, st_rets, rr_rets,
               tr_kills, st_kills, rr_kills,
               tr_stages, n_ep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    GREEDY_C     = "#4e79a7"
    STOCHASTIC_C = "#f28e2b"
    RANDOM_C     = "#e15759"
    ZERO_C       = "#d3d3d3"

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f"Greedy vs Stochastic vs Random  ({n_ep} episodes each)",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, hspace=0.46, wspace=0.36)

    def _spine(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    all_rets = tr_rets + st_rets + rr_rets
    bins = np.linspace(min(all_rets), max(all_rets), 40)

    # panel 1: return distribution
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(rr_rets, bins=bins, color=RANDOM_C,     alpha=0.55, label="Random",     density=True)
    ax1.hist(st_rets, bins=bins, color=STOCHASTIC_C, alpha=0.55, label="Stochastic", density=True)
    ax1.hist(tr_rets, bins=bins, color=GREEDY_C,     alpha=0.55, label="Greedy",     density=True)
    for vals, col, name in [
        (rr_rets, RANDOM_C,     "Random"),
        (st_rets, STOCHASTIC_C, "Stochastic"),
        (tr_rets, GREEDY_C,     "Greedy"),
    ]:
        ax1.axvline(np.mean(vals), color=col, linestyle="--", linewidth=1.5,
                    label=f"{name} μ={np.mean(vals):+.0f}")
    ax1.set_xlabel("Episode return"); ax1.set_ylabel("Density")
    ax1.set_title("1. Return Distribution", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=7); _spine(ax1)

    # panel 2: mean return bar (three agents)
    ax2 = fig.add_subplot(gs[0, 2])
    labels = ["Random", "Stochastic", "Greedy"]
    means  = [np.mean(rr_rets), np.mean(st_rets), np.mean(tr_rets)]
    stds   = [np.std(rr_rets),  np.std(st_rets),  np.std(tr_rets)]
    colors = [RANDOM_C, STOCHASTIC_C, GREEDY_C]
    bars   = ax2.bar(labels, means, yerr=stds, color=colors, alpha=0.8,
                     edgecolor="white", capsize=5, width=0.5)
    for bar, v in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 v + (abs(v) * 0.04 + 2) * np.sign(v) if v != 0 else 2,
                 f"{v:+.0f}", ha="center", fontsize=9, fontweight="bold")
    ax2.axhline(0, color=ZERO_C, linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Mean return ± std")
    ax2.set_title("2. Mean Return", fontsize=10, fontweight="bold"); _spine(ax2)

    # panel 3: kills distribution
    ax3 = fig.add_subplot(gs[1, :2])
    max_k  = int(max(max(tr_kills), max(st_kills), max(rr_kills))) + 1
    bins_k = np.arange(0, max_k + 2) - 0.5
    ax3.hist(rr_kills, bins=bins_k, color=RANDOM_C,     alpha=0.55, label="Random",     density=True)
    ax3.hist(st_kills, bins=bins_k, color=STOCHASTIC_C, alpha=0.55, label="Stochastic", density=True)
    ax3.hist(tr_kills, bins=bins_k, color=GREEDY_C,     alpha=0.55, label="Greedy",     density=True)
    for vals, col, name in [
        (rr_kills, RANDOM_C,     "Random"),
        (st_kills, STOCHASTIC_C, "Stochastic"),
        (tr_kills, GREEDY_C,     "Greedy"),
    ]:
        ax3.axvline(np.mean(vals), color=col, linestyle="--", linewidth=1.5,
                    label=f"{name} μ={np.mean(vals):.1f}")
    ax3.set_xlabel("Kills per episode"); ax3.set_ylabel("Density")
    ax3.set_title("3. Kills Distribution", fontsize=10, fontweight="bold")
    ax3.legend(fontsize=7); _spine(ax3)

    # panel 4: stage reached (greedy agent)
    ax4 = fig.add_subplot(gs[1, 2])
    stage_labels = ["Stage 1\nRecruit", "Stage 2\nSoldier",
                    "Stage 3\nVeteran", "Stage 4\nInfinite"]
    stage_counts = [tr_stages.count(i) for i in range(4)]
    stage_pcts   = [100 * c / n_ep for c in stage_counts]
    stage_cols   = ["#aec6cf", "#77b5d9", "#4e79a7", "#2c4a7c"]
    bars4 = ax4.bar(stage_labels, stage_pcts, color=stage_cols,
                    edgecolor="white", alpha=0.85)
    for bar, pct in zip(bars4, stage_pcts):
        if pct > 0:
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     pct + 0.8, f"{pct:.0f}%",
                     ha="center", fontsize=8, fontweight="bold")
    ax4.set_ylabel("% of episodes"); ax4.set_ylim(0, 105)
    ax4.set_title("4. Stage Reached (Greedy)", fontsize=10, fontweight="bold")
    _spine(ax4)

    out = "outputs/evaluation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--model",    type=str, default="outputs/zombie_policy.pth")
    args = p.parse_args()
    evaluate(args.model, args.episodes)
