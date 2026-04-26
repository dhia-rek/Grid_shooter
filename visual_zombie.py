"""
visual_zombie.py — REINFORCE training loop with live Pygame visualisation.

All drawing code lives in renderer.py.
This file contains only: training loop, REINFORCE updates, saving outputs.

Usage:
    python visual_zombie.py
    python visual_zombie.py --episodes 5000 --render_every 5

Controls:
    SPACE    toggle rendering on/off
    + / -    change render frequency
    ESC / Q  quit
"""

import argparse
import json
import os
import time
from collections import deque

import numpy as np
import torch
import pygame

import renderer as R
from envs.grid_shooter_env import (
    GridShooterEnv, MAX_STEPS, STAGE_DEFS, STAGE_NAMES,
    ACTION_WAIT,
)
from agent.reinforce_agent import PolicyNet, select_action, compute_returns, reinforce_loss


def run(num_episodes=3000, gamma=0.99, lr=1e-3, max_steps=MAX_STEPS,
        seed=0, render_every=1):

    pygame.init()
    R.init_fonts()
    screen = pygame.display.set_mode((R.WIN_W, R.WIN_H))
    pygame.display.set_caption("Grid Shooter — REINFORCE Training")
    clock  = pygame.time.Clock()

    torch.manual_seed(seed)
    np.random.seed(seed)

    env       = GridShooterEnv(max_steps=max_steps)
    obs, _    = env.reset(seed=seed)
    policy    = PolicyNet(obs_dim=obs.shape[0], n_actions=env.action_space.n)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    WEIGHTS_PATH = "outputs/zombie_policy.pth"
    HISTORY_PATH = "outputs/training_history.json"

    # auto-resume: load previous weights + history if they exist
    if os.path.exists(WEIGHTS_PATH):
        try:
            policy.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))
            print(f"Resumed from {WEIGHTS_PATH}")
        except RuntimeError:
            print(f"Architecture changed — starting fresh (ignoring {WEIGHTS_PATH})")

    h_ep, h_ret, h_avg, h_kills = [], [], [], []
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            prev = json.load(f)
        h_ep    = prev.get("episode", [])
        h_ret   = prev.get("return",  [])
        h_avg   = prev.get("avg50",   [])
        h_kills = prev.get("kills",   [])

    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
                     optimizer, T_max=num_episodes, eta_min=1e-4)

    # running EMA baseline — initialised from history so resume is smooth
    EMA_ALPHA  = 0.05
    ema_return = float(h_avg[-1]) if h_avg else 0.0

    window50   = deque(maxlen=50)
    best_kills = int(max(h_kills)) if h_kills else 0
    ep_offset  = h_ep[-1] if h_ep else 0
    rendering  = True
    re_freq    = render_every
    quit_flag  = False

    # ── event handler ─────────────────────────────────────────────────────────
    def handle_events():
        nonlocal rendering, re_freq, quit_flag
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_flag = True
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    quit_flag = True
                if event.key == pygame.K_SPACE:
                    rendering = not rendering
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    re_freq = max(1, re_freq - 1)
                if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    re_freq = min(100, re_freq + 1)

    # ── frame renderer ────────────────────────────────────────────────────────
    def render_frame(ep, ep_return, kills, action, stage):
        screen.fill(R.BG)
        R.draw_grid_bg(screen, stage)
        R.draw_zombies(screen, env)
        R.draw_bullet(screen, env)
        R.draw_agent(screen, env)
        R.update_particles(screen)
        R.draw_popups(screen)
        R.draw_banner(screen)
        R.draw_flash(screen)
        R.draw_stage_bar(screen, stage, kills)
        R.draw_info_bar(screen, ep, num_episodes, kills, best_kills,
                        ep_return, action, stage, rendering, re_freq)
        R.draw_panel(screen, ep, num_episodes, stage,
                     h_ep, h_ret, h_avg, h_kills, best_kills)
        pygame.display.flip()

    # ── training loop ─────────────────────────────────────────────────────────
    for ep in range(1, num_episodes + 1):
        handle_events()
        if quit_flag:
            break
        do_render = rendering and (ep % re_freq == 0 or ep == 1)

        obs, _ = env.reset(seed=seed + ep)
        R.reset_effects()

        done        = False
        ep_return   = 0.0
        rewards_buf = []
        logprob_buf = []
        entropy_buf = []
        last_action = ACTION_WAIT
        prev_kills  = 0
        prev_stage  = 0
        z_snapshot  = []

        while not done:
            handle_events()
            if quit_flag:
                break

            action, log_prob, entropy = select_action(policy, obs)
            z_snapshot = [(z[0], z[1], z[2]) for z in env.zombies]

            obs, reward, terminated, truncated, info = env.step(action)
            done        = terminated or truncated
            ep_return  += reward
            last_action = action
            rewards_buf.append(reward)
            logprob_buf.append(log_prob)
            entropy_buf.append(entropy)
            R.bump_tick()

            kills = info.get("kills", env.kills)
            stage = info.get("stage", env.stage)

            if do_render:
                # kill particles
                if kills > prev_kills:
                    for zb in z_snapshot:
                        if zb[2]:
                            killed = not any(
                                z[2] and z[0] == zb[0] and z[1] == zb[1]
                                for z in env.zombies)
                            if killed:
                                cx, cy = R.cell_center(zb[0], zb[1])
                                scol   = R.STAGE_COLS[stage]
                                R.spawn_particles(cx, cy, scol, n=26)
                                R.spawn_particles(cx, cy, (255, 255, 180), n=8)
                                R.spawn_popup(cx, cy, f"+{10 + stage*5}", R.GREEN_C)
                                R.trigger_flash(scol, 60)
                                break

                # stage advance banner
                if stage > prev_stage:
                    if stage < len(STAGE_DEFS) - 1:
                        R.trigger_banner(f"STAGE {stage+1}  {STAGE_NAMES[stage]}!",
                                         R.STAGE_COLS[stage])
                    else:
                        R.trigger_banner("∞  INFINITE MODE!", R.STAGE_COLS[stage])
                    R.trigger_flash(R.STAGE_COLS[stage], 120)

                if terminated:
                    R.trigger_flash(R.RED_C, 160)

            prev_kills = kills
            prev_stage = stage

            if do_render:
                render_frame(ep, ep_return, kills, last_action, stage)
                clock.tick(30)

        if quit_flag:
            break

        # ── REINFORCE update ──────────────────────────────────────────────────
        ema_return = EMA_ALPHA * ep_return + (1 - EMA_ALPHA) * ema_return
        returns = compute_returns(rewards_buf, gamma=gamma)
        loss    = reinforce_loss(logprob_buf, returns, normalize=True,
                                 baseline=ema_return,
                                 entropies=entropy_buf, entropy_coeff=0.01)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()

        kills = info.get("kills", env.kills)
        best_kills = max(best_kills, kills)
        window50.append(ep_return)

        if ep % 10 == 0 or ep == 1:
            h_ep.append(ep_offset + ep)
            h_ret.append(ep_return)
            h_avg.append(float(np.mean(window50)))
            h_kills.append(float(kills))

        if do_render:
            render_frame(ep, ep_return, kills, last_action, env.stage)
            time.sleep(0.12)

    # ── save weights, history, plots (always, even if interrupted) ───────────
    os.makedirs("outputs", exist_ok=True)
    torch.save(policy.state_dict(), WEIGHTS_PATH)

    if h_ep:
        history = {"episode": h_ep, "return": h_ret, "avg50": h_avg, "kills": h_kills}
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)
        _save_plots(h_ep, h_ret, h_avg, h_kills)

    completed_ep = ep - 1 if quit_flag else num_episodes

    # ── final screen ──────────────────────────────────────────────────────────
    screen.fill(R.BG)
    R.draw_panel(screen, completed_ep, num_episodes, env.stage,
                 h_ep, h_ret, h_avg, h_kills, best_kills)
    title = "Interrupted — Progress Saved!" if quit_flag else "Training Complete!"
    title_col = R.TEXT_C if quit_flag else R.GOLD_C
    msgs = [
        (R._fnt_xl, title,                                                  title_col),
        (R._fnt_lg, f"Trained {completed_ep}/{num_episodes} eps  |  Best: {best_kills} kills", R.GREEN_C),
        (R._fnt_sm, "outputs/zombie_policy.pth  — policy saved",            R.TEXT_C),
        (R._fnt_sm, "outputs/training_curves.png — plot saved",             R.TEXT_C),
        (R._fnt_sm, "Press any key to exit",                                R.DIM_C),
    ]
    cy = R.PAD + R.GRID_PX // 2 - 60
    for fnt, txt, col in msgs:
        s = fnt.render(txt, True, col)
        screen.blit(s, s.get_rect(center=(R.PAD + R.GRID_PX // 2, cy)))
        cy += s.get_height() + 10
    pygame.display.flip()

    headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"
    if not headless:
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type in (pygame.QUIT, pygame.KEYDOWN):
                    waiting = False
    pygame.quit()

    print(f"Episodes trained: {completed_ep}/{num_episodes}")
    print(f"Best score: {best_kills} kills")
    print("Saved: outputs/zombie_policy.pth")
    if h_ep:
        print("Saved: outputs/training_history.json")
        print("Saved: outputs/training_curves.png")
    print("Run:   python view_policy.py   to watch the trained agent")


def _save_plots(h_ep, h_ret, h_avg, h_kills):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(13, 8))
    fig.suptitle("Grid Shooter — REINFORCE Training Curves",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32)

    RAW_C  = "#4e79a7"
    AVG_C  = "#f28e2b"
    KIL_C  = "#59a14f"
    ZERO_C = "#d3d3d3"

    def _style(ax, xlabel, ylabel, title):
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0, color=ZERO_C, linewidth=0.8, linestyle="--")
        ax.legend(fontsize=8)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(h_ep, h_ret, color=RAW_C, alpha=0.6, linewidth=1.0, label="Episode return")
    _style(ax1, "Episode", "Return", "1. Raw Episode Return")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(h_ep, h_ret, color=RAW_C, alpha=0.2, linewidth=0.8, label="Raw")
    ax2.plot(h_ep, h_avg, color=AVG_C, linewidth=2.0, label="Avg last 50")
    _style(ax2, "Episode", "Return", "2. Moving Average (50 eps)")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(h_ep, h_kills, color=KIL_C, linewidth=1.5, label="Kills / episode")
    ax3.fill_between(h_ep, h_kills, alpha=0.15, color=KIL_C)
    _style(ax3, "Episode", "Kills", "3. Kills per Episode")

    ax4 = fig.add_subplot(gs[1, 1])
    if len(h_ret) >= 4:
        q     = len(h_ret) // 4
        early = float(np.mean(h_ret[:q]))
        late  = float(np.mean(h_ret[-q:]))
        bars  = ax4.bar(["Early (first 25%)", "Late (last 25%)"],
                        [early, late], color=[RAW_C, AVG_C],
                        edgecolor="white", width=0.5)
        for bar, val in zip(bars, [early, late]):
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     val + abs(val) * 0.04,
                     f"{val:+.2f}", ha="center", fontsize=9, fontweight="bold")
        ax4.axhline(0, color=ZERO_C, linewidth=0.8, linestyle="--")
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)
        ax4.set_ylabel("Avg return", fontsize=9)
        ax4.set_title("4. Early vs Late Performance", fontsize=10, fontweight="bold")

    plt.savefig("outputs/training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",     type=int,   default=3000)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--max_steps",    type=int,   default=MAX_STEPS)
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--render_every", type=int,   default=1)
    args = p.parse_args()
    run(
        num_episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        max_steps=args.max_steps,
        seed=args.seed,
        render_every=args.render_every,
    )
