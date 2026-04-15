"""
visual_train.py — Train the REINFORCE agent with a live Pygame visualisation.

Left panel  : the game grid (current training episode)
Right panel : live charts — episode return, 50-ep moving average, win rate
Bottom bar  : episode / step / action / reward / result

Usage:
    python visual_train.py                        # 2000 episodes
    python visual_train.py --episodes 1000        # shorter run
    python visual_train.py --episodes 3000 --simple   # easier mode
    python visual_train.py --speed fast           # skip rendering every frame

Controls:
    SPACE     toggle rendering on/off (speed up training)
    +/-       increase/decrease render frequency
    ESC / Q   quit
"""

import argparse
import math
import sys
import time
from collections import deque

import numpy as np
import torch
import pygame

from envs.grid_shooter_env import (
    GridShooterEnv,
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SHOOT, ACTION_WAIT,
)
from agent.reinforce_agent import PolicyNet, select_action, compute_returns, reinforce_loss

# ── Layout constants ──────────────────────────────────────────────────────────
CELL        = 72          # px per grid cell
PAD         = 18          # general padding
GRID_SIZE   = 6
GRID_PX     = CELL * GRID_SIZE
CHART_W     = 480
CHART_H     = 200
INFO_H      = 110
WIN_W       = PAD + GRID_PX + PAD + CHART_W + PAD
WIN_H       = PAD + GRID_PX + PAD

# ── Palette ───────────────────────────────────────────────────────────────────
BG          = ( 12,  12,  22)
CELL_BG     = ( 22,  22,  38)
GRID_LINE   = ( 38,  38,  58)
AGENT_C     = ( 60, 150, 255)
ENEMY_C     = (255,  60,  60)
ABUL_C      = (255, 225,  40)
EBUL_C      = (255, 120,  30)
TEXT_C      = (210, 215, 230)
DIM_C       = ( 90,  95, 115)
WIN_C       = ( 70, 210, 110)
LOSE_C      = (220,  70,  70)
TIMEOUT_C   = (200, 180,  50)
CHART_BG    = ( 18,  18,  32)
CHART_GRID  = ( 35,  35,  55)
RETURN_C    = ( 80, 140, 255)
AVG_C       = (255, 200,  40)
WINRATE_C   = ( 70, 210, 110)
ACCENT      = ( 80,  80, 120)

ACTION_NAMES = {
    ACTION_UP: "UP", ACTION_DOWN: "DOWN", ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT", ACTION_SHOOT: "SHOOT", ACTION_WAIT: "WAIT",
}


# ── Drawing helpers ───────────────────────────────────────────────────────────

def rr(surf, colour, rect, r=8):
    pygame.draw.rect(surf, colour, rect, border_radius=r)


def cell_px(gx, gy):
    return PAD + gx * CELL, PAD + gy * CELL


def draw_grid(screen, env):
    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            px, py = cell_px(gx, gy)
            rr(screen, CELL_BG, (px + 2, py + 2, CELL - 4, CELL - 4), 6)
    for i in range(GRID_SIZE + 1):
        x = PAD + i * CELL
        y = PAD + i * CELL
        pygame.draw.line(screen, GRID_LINE, (x, PAD), (x, PAD + GRID_PX), 1)
        pygame.draw.line(screen, GRID_LINE, (PAD, y), (PAD + GRID_PX, y), 1)

    # Enemy
    if env.enemy_alive:
        ex, ey = env.enemy_pos
        px, py = cell_px(ex, ey)
        rr(screen, ENEMY_C, (px + 10, py + 10, CELL - 20, CELL - 20), 10)
        lbl = _font_md.render("E", True, (255, 255, 255))
        screen.blit(lbl, lbl.get_rect(center=(px + CELL // 2, py + CELL // 2)))

    # Agent bullet (triangle pointing up)
    if env.agent_bullet is not None:
        bx, by = env.agent_bullet
        if 0 <= by < GRID_SIZE:
            px, py = cell_px(bx, by)
            cx, cy = px + CELL // 2, py + CELL // 2
            pygame.draw.polygon(screen, ABUL_C,
                                [(cx, cy - 16), (cx - 7, cy + 9), (cx + 7, cy + 9)])

    # Enemy bullet (triangle pointing down)
    if env.enemy_bullet is not None:
        bx, by = env.enemy_bullet
        if 0 <= by < GRID_SIZE:
            px, py = cell_px(bx, by)
            cx, cy = px + CELL // 2, py + CELL // 2
            pygame.draw.polygon(screen, EBUL_C,
                                [(cx, cy + 16), (cx - 7, cy - 9), (cx + 7, cy - 9)])

    # Agent
    ax, ay = env.agent_pos
    px, py = cell_px(ax, ay)
    rr(screen, AGENT_C, (px + 10, py + 10, CELL - 20, CELL - 20), 10)
    lbl = _font_md.render("A", True, (255, 255, 255))
    screen.blit(lbl, lbl.get_rect(center=(px + CELL // 2, py + CELL // 2)))


def draw_chart(screen, history_ep, history_ret, history_avg, history_wr,
               cx, cy, w, h, title):
    """Draw a mini chart at (cx, cy) with size (w, h)."""
    rr(screen, CHART_BG, (cx, cy, w, h), 8)

    # horizontal grid lines
    for pct in [0.25, 0.5, 0.75]:
        y = cy + int(h * pct)
        pygame.draw.line(screen, CHART_GRID, (cx, y), (cx + w, y), 1)

    n = len(history_ep)
    if n < 2:
        lbl = _font_sm.render(title, True, DIM_C)
        screen.blit(lbl, (cx + 6, cy + 6))
        return

    # normalise to chart coords
    def scale_y(vals, margin=10):
        lo, hi = min(vals), max(vals)
        span = hi - lo or 1
        return [cy + h - margin - int((v - lo) / span * (h - margin * 2))
                for v in vals]

    xs = [cx + int((i / (n - 1)) * (w - 1)) for i in range(n)]

    # return curve
    ys_ret = scale_y(history_ret)
    for i in range(1, n):
        pygame.draw.line(screen, (*RETURN_C, 160),
                         (xs[i - 1], ys_ret[i - 1]), (xs[i], ys_ret[i]), 1)

    # moving average
    ys_avg = scale_y(history_avg)
    for i in range(1, n):
        pygame.draw.line(screen, AVG_C,
                         (xs[i - 1], ys_avg[i - 1]), (xs[i], ys_avg[i]), 2)

    # win rate (drawn on separate axis: 0-100%)
    ys_wr = [cy + h - 8 - int((v / 100.0) * (h - 16)) for v in history_wr]
    for i in range(1, n):
        pygame.draw.line(screen, WINRATE_C,
                         (xs[i - 1], ys_wr[i - 1]), (xs[i], ys_wr[i]), 2)

    # title + legend
    screen.blit(_font_sm.render(title, True, DIM_C), (cx + 6, cy + 6))
    for i, (label, col) in enumerate([
        ("Return", RETURN_C), ("Avg-50", AVG_C), ("Win%", WINRATE_C)
    ]):
        lx = cx + 6 + i * 110
        pygame.draw.line(screen, col, (lx, cy + h - 14), (lx + 18, cy + h - 14), 2)
        screen.blit(_font_sm.render(label, True, col), (lx + 22, cy + h - 20))

    # latest values at right edge
    if history_ret:
        screen.blit(_font_sm.render(f"{history_ret[-1]:+.1f}", True, RETURN_C),
                    (cx + w - 48, ys_ret[-1] - 8))


def draw_info_bar(screen, ep, total_ep, step, max_steps, action, reward,
                  ep_return, result, win_count, rendering, render_every):
    """Bottom bar under the grid."""
    bx, by = PAD, PAD + GRID_PX + 8
    bw, bh = GRID_PX, INFO_H - 12

    rr(screen, CHART_BG, (bx, by, bw, bh), 8)

    col = WIN_C if result == "win" else LOSE_C if result and "lose" in result else (
          TIMEOUT_C if result == "timeout" else TEXT_C)

    lines = [
        (f"Episode {ep:>5} / {total_ep}   Win rate {win_count/ep*100:.1f}%   Wins {win_count}", TEXT_C),
        (f"Step    {step:>3} / {max_steps}   Action {ACTION_NAMES.get(action, '-'):<6}   "
         f"Reward {reward:+.2f}   Return {ep_return:+.2f}", DIM_C),
        (f"Result: {result or 'in progress':<18}   "
         f"{'[RENDER ON]' if rendering else '[RENDER OFF — SPACE to enable]'}", col),
    ]
    y = by + 8
    for text, c in lines:
        screen.blit(_font_sm.render(text, True, c), (bx + 10, y))
        y += 22


def draw_right_panel(screen, ep, total_ep, history_ep, history_ret,
                     history_avg, history_wr, win_count):
    rx = PAD + GRID_PX + PAD
    ry = PAD

    # Episode progress bar
    bar_h = 18
    bar_w = CHART_W
    pct = ep / total_ep
    rr(screen, ACCENT,   (rx, ry, bar_w, bar_h), 6)
    rr(screen, AGENT_C,  (rx, ry, int(bar_w * pct), bar_h), 6)
    label = _font_sm.render(f"Training progress  {ep}/{total_ep}", True, TEXT_C)
    screen.blit(label, label.get_rect(midleft=(rx + 8, ry + bar_h // 2)))
    ry += bar_h + 10

    # Big chart
    ch = GRID_PX - bar_h - 10
    draw_chart(screen, history_ep, history_ret, history_avg, history_wr,
               rx, ry, CHART_W, ch, "Live Training Curves")

    # Live stats on top-right corner of chart
    if history_ret:
        stats = [
            (f"Latest return  {history_ret[-1]:+.2f}", RETURN_C),
            (f"Avg-50         {history_avg[-1]:+.2f}", AVG_C),
            (f"Win rate       {history_wr[-1]:.1f}%", WINRATE_C),
            (f"Total wins     {win_count}", WIN_C),
        ]
        sx, sy = rx + CHART_W - 195, ry + 28
        for txt, col in stats:
            surf = _font_sm.render(txt, True, col)
            screen.blit(surf, (sx, sy))
            sy += 20


# ── Global fonts (set after pygame.init) ─────────────────────────────────────
_font_lg = _font_md = _font_sm = None


def init_fonts():
    global _font_lg, _font_md, _font_sm
    _font_lg = pygame.font.SysFont("monospace", 22, bold=True)
    _font_md = pygame.font.SysFont("monospace", 18, bold=True)
    _font_sm = pygame.font.SysFont("monospace", 13)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(num_episodes=2000, gamma=0.99, lr=1e-3, max_steps=50,
        seed=0, simple_mode=False, render_every=1):

    pygame.init()
    init_fonts()

    total_h = PAD + GRID_PX + INFO_H
    screen  = pygame.display.set_mode((WIN_W, total_h))
    pygame.display.set_caption("Grid Shooter — Live REINFORCE Training")
    clock   = pygame.time.Clock()

    torch.manual_seed(seed)
    np.random.seed(seed)

    env       = GridShooterEnv(grid_size=GRID_SIZE, max_steps=max_steps,
                               simple_mode=simple_mode)
    obs, _    = env.reset(seed=seed)
    policy    = PolicyNet(obs_dim=obs.shape[0], n_actions=env.action_space.n)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    history_ep  = []
    history_ret = []
    history_avg = []
    history_wr  = []
    window50    = deque(maxlen=50)
    win_count   = 0
    rendering   = True
    re_freq     = render_every   # render every N episodes

    for ep in range(1, num_episodes + 1):

        # ── handle events before each episode ────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_SPACE:
                    rendering = not rendering
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    re_freq = max(1, re_freq - 1)
                if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    re_freq = min(50, re_freq + 1)

        do_render = rendering and (ep % re_freq == 0 or ep == 1)

        # ── collect episode step-by-step (so we can render mid-episode) ──────
        obs, _ = env.reset(seed=seed + ep)
        done        = False
        step        = 0
        ep_return   = 0.0
        rewards_buf = []
        logprob_buf = []
        last_action = ACTION_WAIT
        last_reward = 0.0
        result      = None

        while not done:
            # event pump inside step loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        pygame.quit(); sys.exit()
                    if event.key == pygame.K_SPACE:
                        rendering = not rendering
                        do_render = rendering and (ep % re_freq == 0 or ep == 1)
                    if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        re_freq = max(1, re_freq - 1)
                    if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        re_freq = min(50, re_freq + 1)

            action, log_prob = select_action(policy, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            ep_return   += reward
            last_action  = action
            last_reward  = reward
            rewards_buf.append(reward)
            logprob_buf.append(log_prob)
            if done:
                result = info.get("result", "timeout")

            # ── render this step ─────────────────────────────────────────────
            if do_render:
                screen.fill(BG)
                draw_grid(screen, env)
                draw_right_panel(screen, ep, num_episodes,
                                 history_ep, history_ret, history_avg, history_wr,
                                 win_count)
                draw_info_bar(screen, ep, num_episodes, step, max_steps,
                              last_action, last_reward, ep_return, result,
                              win_count, rendering, re_freq)
                pygame.display.flip()
                clock.tick(30)   # cap at 30 fps while rendering

        # ── REINFORCE update ──────────────────────────────────────────────────
        returns = compute_returns(rewards_buf, gamma=gamma)
        loss    = reinforce_loss(logprob_buf, returns, normalize=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── bookkeeping ───────────────────────────────────────────────────────
        window50.append(ep_return)
        if result == "win":
            win_count += 1
        avg50    = float(np.mean(window50))
        win_rate = win_count / ep * 100.0

        if ep % 10 == 0 or ep == 1:
            history_ep.append(ep)
            history_ret.append(ep_return)
            history_avg.append(avg50)
            history_wr.append(win_rate)

        # ── final frame after episode ends (brief pause so result is visible) ─
        if do_render:
            screen.fill(BG)
            draw_grid(screen, env)
            draw_right_panel(screen, ep, num_episodes,
                             history_ep, history_ret, history_avg, history_wr,
                             win_count)
            draw_info_bar(screen, ep, num_episodes, step, max_steps,
                          last_action, last_reward, ep_return, result,
                          win_count, rendering, re_freq)
            pygame.display.flip()
            time.sleep(0.08)

    # ── training done — save & show final screen ──────────────────────────────
    torch.save(policy.state_dict(), "outputs/grid_shooter_policy.pth")

    screen.fill(BG)
    draw_right_panel(screen, num_episodes, num_episodes,
                     history_ep, history_ret, history_avg, history_wr, win_count)
    draw_info_bar(screen, num_episodes, num_episodes, 0, max_steps,
                  ACTION_WAIT, 0.0, 0.0, "TRAINING COMPLETE",
                  win_count, True, re_freq)

    msg1 = _font_lg.render("Training complete!", True, WIN_C)
    msg2 = _font_md.render(
        f"Win rate: {win_count/num_episodes*100:.1f}%   "
        f"Wins: {win_count}/{num_episodes}   — policy saved", True, TEXT_C)
    msg3 = _font_sm.render("Press any key or close to exit", True, DIM_C)
    cx = PAD + GRID_PX // 2
    screen.blit(msg1, msg1.get_rect(center=(cx, PAD + GRID_PX // 2 - 30)))
    screen.blit(msg2, msg2.get_rect(center=(cx, PAD + GRID_PX // 2 + 10)))
    screen.blit(msg3, msg3.get_rect(center=(cx, PAD + GRID_PX // 2 + 42)))
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN):
                waiting = False

    pygame.quit()
    print(f"\nTraining complete. Policy saved to outputs/grid_shooter_policy.pth")
    print(f"Win rate: {win_count/num_episodes*100:.1f}%  ({win_count}/{num_episodes} wins)")
    print("Run  python plot.py  to generate figures.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",    type=int,   default=2000)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--gamma",       type=float, default=0.99)
    p.add_argument("--max_steps",   type=int,   default=50)
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--render_every",type=int,   default=1,
                   help="Render every N episodes (default 1). Increase to train faster.")
    p.add_argument("--simple",      action="store_true",
                   help="Simple mode: static enemy, no enemy bullets")
    args = p.parse_args()
    run(
        num_episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        max_steps=args.max_steps,
        seed=args.seed,
        simple_mode=args.simple,
        render_every=args.render_every,
    )
