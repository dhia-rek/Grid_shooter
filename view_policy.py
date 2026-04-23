"""
view_policy.py — Watch the trained agent play.

Loads outputs/zombie_policy.pth and runs it greedily (no training).

Usage:
    python view_policy.py
    python view_policy.py --episodes 10 --delay 80
    python view_policy.py --model outputs/zombie_policy.pth
"""

import argparse
import math
import sys
import time

import numpy as np
import torch
import pygame
import pygame.gfxdraw

from envs.grid_shooter_env import (
    GridShooterEnv, ACTION_NAMES, GRID, MAX_STEPS, STAGE_NAMES, STAGE_DEFS,
    DIR_DOWN, DIR_UP, DIR_RIGHT, DIR_LEFT,
    ACTION_WAIT,
)
from agent.reinforce_agent import PolicyNet

# ── Layout ────────────────────────────────────────────────────────────────────
CELL    = 80
PAD     = 24
GRID_PX = CELL * GRID
INFO_H  = 90
WIN_W   = PAD + GRID_PX + PAD
WIN_H   = PAD + GRID_PX + INFO_H

# ── Colours ───────────────────────────────────────────────────────────────────
BG         = (  8,   8,  16)
CELL_DARK  = ( 14,  14,  24)
CELL_LIGHT = ( 20,  20,  34)
GRID_COL   = ( 28,  28,  46)
AGENT_C    = ( 40, 120, 240)
AGENT_SH   = (110, 180, 255)
BULLET_C   = (255, 235,  50)
PANEL_BG   = ( 11,  11,  20)
TEXT_C     = (220, 225, 240)
DIM_C      = ( 75,  80, 105)
GOLD_C     = (255, 200,  30)
GREEN_C    = ( 55, 215, 100)
RED_C      = (220,  55,  55)

STAGE_COLS = [
    ( 60, 200,  80),
    ( 60, 160, 255),
    (220, 120,  40),
    (220,  50,  50),
]
STAGE_BG_TINTS = [(0,8,0), (0,0,12), (10,4,0), (14,0,0)]

_DIR_ARROW = {DIR_DOWN:(0,1), DIR_UP:(0,-1), DIR_RIGHT:(1,0), DIR_LEFT:(-1,0)}

# ── Fonts ─────────────────────────────────────────────────────────────────────
_fnt_lg = _fnt_md = _fnt_sm = None

def init_fonts():
    global _fnt_lg, _fnt_md, _fnt_sm
    _fnt_lg = pygame.font.SysFont("monospace", 22, bold=True)
    _fnt_md = pygame.font.SysFont("monospace", 16, bold=True)
    _fnt_sm = pygame.font.SysFont("monospace", 13)

# ── Drawing helpers ───────────────────────────────────────────────────────────
_tick = 0

def rr(surf, col, rect, r=8):
    pygame.draw.rect(surf, col, rect, border_radius=r)

def cell_center(gx, gy):
    return PAD + gx*CELL + CELL//2, PAD + gy*CELL + CELL//2

def cell_tl(gx, gy):
    return PAD + gx*CELL, PAD + gy*CELL

def draw_glow(surf, col, cx, cy, rad, steps=3):
    r, g, b = col
    for i in range(steps, 0, -1):
        a    = int(55 * i / steps)
        rad2 = rad + (steps - i + 1) * 5
        s    = pygame.Surface((rad2*2, rad2*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(s, rad2, rad2, rad2, (r, g, b, a))
        surf.blit(s, (cx - rad2, cy - rad2))

def draw_grid_bg(screen, stage):
    tint = STAGE_BG_TINTS[stage]
    for gy in range(GRID):
        for gx in range(GRID):
            base = CELL_LIGHT if (gx+gy) % 2 == 0 else CELL_DARK
            col  = tuple(min(255, base[i] + tint[i]) for i in range(3))
            px, py = cell_tl(gx, gy)
            rr(screen, col, (px+1, py+1, CELL-2, CELL-2), 3)
    for i in range(GRID+1):
        x = PAD + i*CELL; y = PAD + i*CELL
        pygame.draw.line(screen, GRID_COL, (x, PAD), (x, PAD+GRID_PX), 1)
        pygame.draw.line(screen, GRID_COL, (PAD, y), (PAD+GRID_PX, y), 1)

def draw_agent(screen, env):
    ax, ay = env.agent_pos
    cx, cy = cell_center(ax, ay)
    px, py = cell_tl(ax, ay)
    m = 10
    draw_glow(screen, AGENT_C, cx, cy, CELL//2 - 8, steps=3)
    rr(screen, (15, 55, 130), (px+m-2, py+m-2, CELL-m*2+4, CELL-m*2+4), 12)
    rr(screen, AGENT_C,       (px+m,   py+m,   CELL-m*2,   CELL-m*2),   11)
    pygame.draw.line(screen, AGENT_SH, (px+m+5, py+m+5), (px+m+5, py+CELL-m-5), 2)
    lbl = _fnt_md.render("A", True, (230, 240, 255))
    screen.blit(lbl, lbl.get_rect(center=(cx, cy)))

def _draw_dir_arrow(screen, cx, cy, d, col):
    dx, dy = _DIR_ARROW[d]
    tip   = (cx + dx*16, cy + dy*16)
    perp  = (-dy, dx)
    base1 = (cx - dx*6 + perp[0]*7, cy - dy*6 + perp[1]*7)
    base2 = (cx - dx*6 - perp[0]*7, cy - dy*6 - perp[1]*7)
    pygame.draw.polygon(screen, col, [tip, base1, base2])

def draw_zombies(screen, env):
    zcol = STAGE_COLS[env.stage]
    for z in env.zombies:
        if not z[2]:
            continue
        gx, gy = z[0], z[1]
        if not (0 <= gx < GRID and 0 <= gy < GRID):
            continue
        cx, cy = cell_center(gx, gy)
        px, py = cell_tl(gx, gy)
        m     = 10
        pulse = 0.80 + 0.20*math.sin(_tick*0.14 + gx*0.9)
        col   = tuple(int(c * pulse) for c in zcol)
        draw_glow(screen, zcol, cx, cy, CELL//2 - 10, steps=2)
        rr(screen, tuple(c//4 for c in zcol), (px+m-2, py+m-2, CELL-m*2+4, CELL-m*2+4), 10)
        rr(screen, col,                        (px+m,   py+m,   CELL-m*2,   CELL-m*2),   9)
        for ex in [cx-7, cx+7]:
            pygame.gfxdraw.filled_circle(screen, ex, cy-5, 3, (255, 60, 60, 220))
            pygame.gfxdraw.filled_circle(screen, ex, cy-5, 1, (255, 200, 200, 255))
        arrow_col = tuple(min(255, c + 120) for c in zcol)
        _draw_dir_arrow(screen, cx, cy + 7, z[3], arrow_col)
        lbl = _fnt_sm.render("Z", True, tuple(min(255, c+80) for c in zcol))
        screen.blit(lbl, lbl.get_rect(center=(cx, cy - 7)))

def draw_bullet(screen, env):
    if env.bullet is None:
        return
    bx, by, bdx, bdy = env.bullet
    if not (0 <= bx < GRID and 0 <= by < GRID):
        return
    cx, cy = cell_center(bx, by)
    for i in range(1, 6):
        alpha = int(160 * (1 - i/6))
        tx = cx - bdx * i * 8
        ty = cy - bdy * i * 8
        pygame.gfxdraw.filled_circle(screen, int(tx), int(ty),
                                     max(1, 4-i), (*BULLET_C, alpha))
    draw_glow(screen, BULLET_C, cx, cy, 8, steps=2)
    pygame.gfxdraw.filled_circle(screen, cx, cy, 6, (*BULLET_C, 255))
    pygame.gfxdraw.filled_circle(screen, cx, cy, 2, (255, 255, 255, 255))

def draw_hud(screen, ep, total, kills, best, stage, action, ep_return, result):
    bx, by = PAD, PAD + GRID_PX + 8
    bw, bh = GRID_PX, INFO_H - 16
    rr(screen, PANEL_BG, (bx, by, bw, bh), 8)
    scol = STAGE_COLS[stage]
    res_col = GREEN_C if result == "survived" else RED_C if result == "dead" else TEXT_C
    rows = [
        (f"Episode {ep}/{total}   Stage {stage+1} — {STAGE_NAMES[stage]}   "
         f"Kills {kills}   Best {best}   Return {ep_return:+.1f}", scol),
        (f"Action: {ACTION_NAMES.get(action, '-'):<6}   "
         f"Result: {result or 'playing'}", res_col),
        (f"ESC/Q — quit", DIM_C),
    ]
    y = by + 8
    for txt, col in rows:
        screen.blit(_fnt_sm.render(txt, True, col), (bx + 10, y))
        y += 22


# ── Main ──────────────────────────────────────────────────────────────────────

def run(num_episodes=5, delay_ms=80, model_path="outputs/zombie_policy.pth", seed=42):
    global _tick

    pygame.init()
    init_fonts()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Grid Shooter — Policy Viewer")
    clock  = pygame.time.Clock()

    env    = GridShooterEnv()
    obs, _ = env.reset(seed=seed)
    policy = PolicyNet(obs_dim=obs.shape[0], n_actions=env.action_space.n)

    try:
        policy.load_state_dict(torch.load(model_path, map_location="cpu",
                                          weights_only=True))
    except FileNotFoundError:
        print(f"ERROR: {model_path} not found — run  python visual_zombie.py  first.")
        pygame.quit()
        return

    policy.eval()
    best_kills = 0

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        done       = False
        ep_return  = 0.0
        last_action = ACTION_WAIT
        result      = None
        kills       = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        pygame.quit(); sys.exit()

            with torch.no_grad():
                s = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                last_action = int(torch.argmax(policy(s), dim=1).item())

            obs, reward, terminated, truncated, info = env.step(last_action)
            done       = terminated or truncated
            ep_return += reward
            kills      = info.get("kills", env.kills)
            _tick     += 1

            if done:
                result = info.get("result", "survived")
                best_kills = max(best_kills, kills)

            screen.fill(BG)
            draw_grid_bg(screen, env.stage)
            draw_zombies(screen, env)
            draw_bullet(screen, env)
            draw_agent(screen, env)
            draw_hud(screen, ep, num_episodes, kills, best_kills,
                     env.stage, last_action, ep_return, result)
            pygame.display.flip()
            clock.tick(1000 // max(1, delay_ms))

        # pause briefly between episodes so the final frame is visible
        pygame.time.wait(800)

    # end screen
    screen.fill(BG)
    msgs = [
        _fnt_lg.render(f"Done — {num_episodes} episodes", True, GOLD_C),
        _fnt_md.render(f"Best score: {best_kills} kills", True, GREEN_C),
        _fnt_sm.render("Press any key to exit", True, DIM_C),
    ]
    cy = WIN_H // 2 - 40
    for s in msgs:
        screen.blit(s, s.get_rect(center=(WIN_W // 2, cy)))
        cy += s.get_height() + 14
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN):
                waiting = False
    pygame.quit()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int,   default=5)
    p.add_argument("--delay",    type=int,   default=80,
                   help="Milliseconds between frames (default 80)")
    p.add_argument("--model",    type=str,   default="outputs/zombie_policy.pth")
    p.add_argument("--seed",     type=int,   default=42)
    args = p.parse_args()
    run(args.episodes, args.delay, args.model, args.seed)
