"""
visual_zombie.py — Staged Zombie Shooter with polished Pygame visuals.
Live REINFORCE training. No timer — die to end an episode.

Stages:  1-Recruit → 2-Soldier → 3-Veteran → 4-INFINITE

Usage:
    python visual_zombie.py
    python visual_zombie.py --episodes 5000 --render_every 3

Controls:
    SPACE    toggle rendering on/off
    + / -    change render frequency
    ESC / Q  quit
"""

import argparse
import math
import os
import random
import sys
import time
from collections import deque

import numpy as np
import torch
import pygame
import pygame.gfxdraw

from envs.grid_shooter_env import (
    GridShooterEnv, ACTION_NAMES, GRID, MAX_STEPS, STAGE_NAMES, STAGE_DEFS,
    DIR_DOWN, DIR_UP, DIR_RIGHT, DIR_LEFT,
    ACTION_SHOOT_UP, ACTION_SHOOT_DOWN, ACTION_SHOOT_LEFT, ACTION_SHOOT_RIGHT,
    ACTION_WAIT,
)
from agent.reinforce_agent import PolicyNet, select_action, compute_returns, reinforce_loss

# ── Layout ────────────────────────────────────────────────────────────────────
CELL    = 68
PAD     = 20
GRID_PX = CELL * GRID
PANEL_W = 450
INFO_H  = 125
WIN_W   = PAD + GRID_PX + PAD + PANEL_W + PAD
WIN_H   = PAD + GRID_PX + INFO_H

# ── Colours per stage ─────────────────────────────────────────────────────────
STAGE_COLS = [
    (60,  200,  80),   # Stage 1 — green
    (60,  160, 255),   # Stage 2 — blue
    (220, 120,  40),   # Stage 3 — orange
    (220,  50,  50),   # Stage 4 — red (infinite)
]
STAGE_BG_TINTS = [
    (0, 8, 0),
    (0, 0, 12),
    (10, 4, 0),
    (14, 0, 0),
]

# ── Palette ───────────────────────────────────────────────────────────────────
BG         = (  8,   8,  16)
CELL_DARK  = ( 14,  14,  24)
CELL_LIGHT = ( 20,  20,  34)
GRID_COL   = ( 28,  28,  46)
AGENT_C    = ( 40, 120, 240)
AGENT_SH   = (110, 180, 255)
BULLET_C   = (255, 235,  50)
PANEL_BG   = ( 11,  11,  20)
CHART_BG   = (  9,   9,  18)
TEXT_C     = (220, 225, 240)
DIM_C      = ( 75,  80, 105)
GOLD_C     = (255, 200,  30)
GREEN_C    = ( 55, 215, 100)
RED_C      = (220,  55,  55)
BLUE_C     = ( 55, 140, 255)
ORANGE_C   = (255, 140,  30)

# ── Fonts ─────────────────────────────────────────────────────────────────────
_fnt_xl = _fnt_lg = _fnt_md = _fnt_sm = None


def init_fonts():
    global _fnt_xl, _fnt_lg, _fnt_md, _fnt_sm
    _fnt_xl = pygame.font.SysFont("monospace", 32, bold=True)
    _fnt_lg = pygame.font.SysFont("monospace", 20, bold=True)
    _fnt_md = pygame.font.SysFont("monospace", 16, bold=True)
    _fnt_sm = pygame.font.SysFont("monospace", 12)


# ── Particle system ───────────────────────────────────────────────────────────
class Particle:
    __slots__ = ("x","y","vx","vy","life","max_life","col","size")
    def __init__(self, x, y, col):
        a = random.uniform(0, 2*math.pi)
        s = random.uniform(1.5, 5.0)
        self.x, self.y = x, y
        self.vx, self.vy = math.cos(a)*s, math.sin(a)*s
        self.max_life = random.randint(14, 32)
        self.life = self.max_life
        self.col  = col
        self.size = random.randint(2, 5)

    def update(self):
        self.x  += self.vx;  self.y  += self.vy
        self.vy += 0.1;      self.vx *= 0.92
        self.life -= 1

    def draw(self, surf):
        a = int(255 * self.life / self.max_life)
        sz = max(1, int(self.size * self.life / self.max_life))
        pygame.gfxdraw.filled_circle(surf, int(self.x), int(self.y),
                                     sz, (*self.col, a))


_particles: list = []


def spawn_particles(x, y, col, n=22):
    for _ in range(n):
        _particles.append(Particle(x, y, col))


def update_particles(surf):
    for p in _particles:
        p.update()
        p.draw(surf)
    _particles[:] = [p for p in _particles if p.life > 0]


# ── Pop-up text ───────────────────────────────────────────────────────────────
_popups: list = []   # [cx, cy, text, col, life]


def spawn_popup(cx, cy, text, col):
    _popups.append([cx, cy - 10, text, col, 45])


def draw_popups(screen):
    for p in _popups:
        alpha = min(255, int(255 * p[4] / 45))
        s = _fnt_md.render(p[2], True, p[3])
        s.set_alpha(alpha)
        screen.blit(s, s.get_rect(center=(int(p[0]), int(p[1]))))
        p[1] -= 1.1
        p[4] -= 1
    _popups[:] = [p for p in _popups if p[4] > 0]


# ── Stage banner ──────────────────────────────────────────────────────────────
_banner_life = 0
_banner_text = ""
_banner_col  = (255, 255, 255)


def trigger_banner(text, col):
    global _banner_life, _banner_text, _banner_col
    _banner_life = 80
    _banner_text = text
    _banner_col  = col


def draw_banner(screen):
    global _banner_life
    if _banner_life <= 0:
        return
    alpha = min(255, int(255 * min(_banner_life, 30) / 30))
    s = _fnt_xl.render(_banner_text, True, _banner_col)
    s.set_alpha(alpha)
    screen.blit(s, s.get_rect(center=(PAD + GRID_PX//2, PAD + GRID_PX//2)))
    _banner_life -= 1


# ── Flash ─────────────────────────────────────────────────────────────────────
_flash_alpha = 0
_flash_col   = (0, 0, 0)


def trigger_flash(col, strength=70):
    global _flash_alpha, _flash_col
    _flash_alpha = strength
    _flash_col   = col


def draw_flash(screen):
    global _flash_alpha
    if _flash_alpha <= 0:
        return
    s = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    s.fill((*_flash_col, _flash_alpha))
    screen.blit(s, (0, 0))
    _flash_alpha = max(0, _flash_alpha - 6)


# ── Helpers ───────────────────────────────────────────────────────────────────
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
        a   = int(55 * i / steps)
        rad2 = rad + (steps - i + 1) * 5
        s   = pygame.Surface((rad2*2, rad2*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(s, rad2, rad2, rad2, (r, g, b, a))
        surf.blit(s, (cx - rad2, cy - rad2))


# ── Grid ──────────────────────────────────────────────────────────────────────

def draw_grid_bg(screen, stage):
    tint = STAGE_BG_TINTS[stage]
    for gy in range(GRID):
        for gx in range(GRID):
            base = CELL_LIGHT if (gx+gy) % 2 == 0 else CELL_DARK
            col  = tuple(min(255, base[i] + tint[i]) for i in range(3))
            px, py = cell_tl(gx, gy)
            rr(screen, col, (px+1, py+1, CELL-2, CELL-2), 3)
    for i in range(GRID+1):
        x = PAD + i*CELL;  y = PAD + i*CELL
        pygame.draw.line(screen, GRID_COL, (x, PAD), (x, PAD+GRID_PX), 1)
        pygame.draw.line(screen, GRID_COL, (PAD, y), (PAD+GRID_PX, y), 1)


def draw_agent(screen, env):
    ax, ay = env.agent_pos
    cx, cy = cell_center(ax, ay)
    px, py = cell_tl(ax, ay)
    m = 9
    draw_glow(screen, AGENT_C, cx, cy, CELL//2 - 8, steps=3)
    rr(screen, (15, 55, 130), (px+m-2, py+m-2, CELL-m*2+4, CELL-m*2+4), 11)
    rr(screen, AGENT_C,       (px+m,   py+m,   CELL-m*2,   CELL-m*2),   10)
    pygame.draw.line(screen, AGENT_SH, (px+m+5, py+m+5), (px+m+5, py+CELL-m-5), 2)
    lbl = _fnt_md.render("A", True, (230, 240, 255))
    screen.blit(lbl, lbl.get_rect(center=(cx, cy)))


_DIR_ARROW = {
    DIR_DOWN:  ( 0,  1),
    DIR_UP:    ( 0, -1),
    DIR_RIGHT: ( 1,  0),
    DIR_LEFT:  (-1,  0),
}

def _draw_dir_arrow(screen, cx, cy, d, col):
    """Draw a small filled triangle inside the zombie cell indicating direction."""
    dx, dy = _DIR_ARROW[d]
    tip    = (cx + dx*14, cy + dy*14)
    perp   = (-dy, dx)
    base1  = (cx - dx*6 + perp[0]*6, cy - dy*6 + perp[1]*6)
    base2  = (cx - dx*6 - perp[0]*6, cy - dy*6 - perp[1]*6)
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
        m = 9
        pulse = 0.80 + 0.20*math.sin(_tick*0.14 + gx*0.9)
        col   = tuple(int(c * pulse) for c in zcol)
        draw_glow(screen, zcol, cx, cy, CELL//2 - 9, steps=2)
        rr(screen, tuple(c//4 for c in zcol), (px+m-2, py+m-2, CELL-m*2+4, CELL-m*2+4), 10)
        rr(screen, col,                        (px+m,   py+m,   CELL-m*2,   CELL-m*2),   9)
        # eyes
        for ex in [cx-6, cx+6]:
            pygame.gfxdraw.filled_circle(screen, ex, cy-4, 3, (255, 60, 60, 220))
            pygame.gfxdraw.filled_circle(screen, ex, cy-4, 1, (255, 200, 200, 255))
        # direction arrow
        arrow_col = tuple(min(255, c + 120) for c in zcol)
        _draw_dir_arrow(screen, cx, cy + 6, z[3], arrow_col)
        lbl = _fnt_sm.render("Z", True, tuple(min(255, c+80) for c in zcol))
        screen.blit(lbl, lbl.get_rect(center=(cx, cy - 6)))


def draw_bullet(screen, env):
    if env.bullet is None:
        return
    bx, by, bdx, bdy = env.bullet
    if not (0 <= bx < GRID and 0 <= by < GRID):
        return
    cx, cy = cell_center(bx, by)
    # trail drawn behind the bullet (opposite of travel direction)
    for i in range(1, 6):
        alpha = int(160 * (1 - i/6))
        tx = cx - bdx * i * 7
        ty = cy - bdy * i * 7
        pygame.gfxdraw.filled_circle(screen, int(tx), int(ty),
                                     max(1, 4-i), (*BULLET_C, alpha))
    draw_glow(screen, BULLET_C, cx, cy, 7, steps=2)
    pygame.gfxdraw.filled_circle(screen, cx, cy, 5, (*BULLET_C, 255))
    pygame.gfxdraw.filled_circle(screen, cx, cy, 2, (255, 255, 255, 255))


# ── Stage indicator bar ───────────────────────────────────────────────────────

def draw_stage_bar(screen, stage, kills):
    bx, by = PAD, PAD + GRID_PX + 8
    bw, bh = GRID_PX, 22
    col = STAGE_COLS[stage]

    rr(screen, (20, 20, 36), (bx, by, bw, bh), 6)

    if stage < len(STAGE_DEFS) - 1:
        threshold  = STAGE_DEFS[stage][0]
        prev_kills = STAGE_DEFS[stage-1][0] if stage > 0 else 0
        pct = min(1.0, (kills - prev_kills) / max(1, threshold - prev_kills))
        rr(screen, col, (bx, by, max(6, int(bw*pct)), bh), 6)
        # next stage threshold label
        lbl = _fnt_sm.render(
            f"  STAGE {stage+1}  {STAGE_NAMES[stage]}    "
            f"kills {kills} / {threshold} → next stage", True, TEXT_C)
    else:
        # infinite — pulse bar
        pulse_w = int(bw * (0.5 + 0.5*math.sin(_tick * 0.08)))
        rr(screen, col, (bx, by, pulse_w, bh), 6)
        lbl = _fnt_sm.render(
            f"  ∞  INFINITE STAGE    kills {kills}   SCORE IT ALL!", True, TEXT_C)

    screen.blit(lbl, lbl.get_rect(midleft=(bx+6, by+bh//2)))


# ── Info bar ──────────────────────────────────────────────────────────────────

def draw_info_bar(screen, ep, total_ep, kills, best_kills,
                  ep_return, action, stage, rendering, re_freq):
    bx, by = PAD, PAD + GRID_PX + 34
    bw, bh = GRID_PX, INFO_H - 44
    rr(screen, PANEL_BG, (bx, by, bw, bh), 8)
    y = by + 8
    rows = [
        (f"Episode {ep:>5}/{total_ep}     "
         f"Score(kills) {kills:>3}     Best {best_kills:>3}     Return {ep_return:+.1f}", TEXT_C),
        (f"Action  {ACTION_NAMES.get(action,'-'):<6}     "
         f"Stage  {stage+1} — {STAGE_NAMES[stage]}", STAGE_COLS[stage]),
        (f"{'[RENDER ON  — SPACE]' if rendering else '[RENDER OFF — SPACE]'}"
         f"   speed +/-  (every {re_freq} ep)", DIM_C),
    ]
    for txt, col in rows:
        screen.blit(_fnt_sm.render(txt, True, col), (bx+10, y))
        y += 22


# ── Right panel ───────────────────────────────────────────────────────────────

def draw_panel(screen, ep, total_ep, stage, h_ep, h_ret, h_avg, h_kills, best_kills):
    rx, ry = PAD + GRID_PX + PAD, PAD

    # Header
    t = _fnt_lg.render("REINFORCE — Zombie Shooter", True, GOLD_C)
    screen.blit(t, (rx, ry));  ry += t.get_height() + 4

    # Stage badge
    scol = STAGE_COLS[stage]
    badge = _fnt_lg.render(f"Stage {stage+1}  {STAGE_NAMES[stage]}", True, scol)
    screen.blit(badge, (rx, ry));  ry += badge.get_height() + 6

    # Training progress bar
    bw, bh = PANEL_W, 14
    pct = ep / total_ep
    rr(screen, (22, 22, 40), (rx, ry, bw, bh), 5)
    rr(screen, BLUE_C,       (rx, ry, int(bw*pct), bh), 5)
    screen.blit(_fnt_sm.render(f"Training  {ep}/{total_ep}  ({pct*100:.0f}%)", True, DIM_C),
                (rx+6, ry+1))
    ry += bh + 10

    # Stat boxes
    if h_ret:
        stats = [
            ("Return",   f"{h_ret[-1]:+.1f}",  BLUE_C),
            ("Avg-50",   f"{h_avg[-1]:+.1f}",  GOLD_C),
            ("Kills/ep", f"{h_kills[-1]:.0f}", GREEN_C),
            ("Best",     f"{best_kills}",        ORANGE_C),
        ]
        bxw = (PANEL_W - 3*6) // 4
        for i, (label, val, col) in enumerate(stats):
            bx2 = rx + i*(bxw+6)
            rr(screen, (16, 16, 30), (bx2, ry, bxw, 54), 8)
            pygame.draw.rect(screen, col, (bx2, ry, bxw, 54), 1, border_radius=8)
            v = _fnt_md.render(val,   True, col)
            l = _fnt_sm.render(label, True, DIM_C)
            screen.blit(v, v.get_rect(center=(bx2+bxw//2, ry+20)))
            screen.blit(l, l.get_rect(center=(bx2+bxw//2, ry+40)))
        ry += 60

    # Chart
    ch = WIN_H - ry - 10
    if ch > 60:
        draw_chart(screen, h_ep, h_ret, h_avg, h_kills, rx, ry, PANEL_W, ch)


def draw_chart(screen, h_ep, h_ret, h_avg, h_kills, cx, cy, w, h):
    rr(screen, CHART_BG, (cx, cy, w, h), 8)
    pygame.draw.rect(screen, GRID_COL, (cx, cy, w, h), 1, border_radius=8)
    for pct in [0.25, 0.5, 0.75]:
        yy = cy + int(h*pct)
        pygame.draw.line(screen, (22, 22, 36), (cx+1, yy), (cx+w-1, yy), 1)

    n = len(h_ep)
    if n < 2:
        screen.blit(_fnt_sm.render("Waiting for data...", True, DIM_C),
                    (cx+10, cy+10))
        return

    xs = [cx + int(i/(n-1)*(w-2)) for i in range(n)]

    def ys(vals, mg=12):
        lo, hi = min(vals), max(vals)
        span = hi - lo or 1.0
        return [cy + h - mg - int((v-lo)/span*(h-mg*2)) for v in vals]

    def draw_line(vals, col, thick=2):
        pts = list(zip(xs, ys(vals)))
        for i in range(1, len(pts)):
            pygame.draw.line(screen, col, pts[i-1], pts[i], thick)

    draw_line(h_ret,   (*BLUE_C, 160), 1)
    draw_line(h_avg,   GOLD_C,  2)

    mk = max(h_kills) if max(h_kills) > 0 else 1
    ys_k = [cy + h - 10 - int(v/mk*(h-20)) for v in h_kills]
    for i in range(1, len(xs)):
        pygame.draw.line(screen, GREEN_C, (xs[i-1], ys_k[i-1]), (xs[i], ys_k[i]), 2)

    items = [("Return", BLUE_C), ("Avg-50", GOLD_C), ("Kills", GREEN_C)]
    lx = cx + 8
    for label, col in items:
        pygame.draw.line(screen, col, (lx, cy+h-12), (lx+16, cy+h-12), 2)
        screen.blit(_fnt_sm.render(label, True, col), (lx+20, cy+h-18))
        lx += 110


# ── Main ──────────────────────────────────────────────────────────────────────

def run(num_episodes=3000, gamma=0.99, lr=1e-3, max_steps=MAX_STEPS,
        seed=0, render_every=1):
    global _tick

    pygame.init()
    init_fonts()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Zombie Shooter — Staged REINFORCE Training")
    clock  = pygame.time.Clock()

    torch.manual_seed(seed)
    np.random.seed(seed)

    env    = GridShooterEnv(max_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    policy    = PolicyNet(obs_dim=obs.shape[0], n_actions=env.action_space.n)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    h_ep, h_ret, h_avg, h_kills = [], [], [], []
    window50   = deque(maxlen=50)
    best_kills = 0
    rendering  = True
    re_freq    = render_every

    def handle_events():
        nonlocal rendering, re_freq
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
                    re_freq = min(100, re_freq + 1)

    def render_frame(ep, ep_return, kills, action, stage):
        screen.fill(BG)
        draw_grid_bg(screen, stage)
        draw_zombies(screen, env)
        draw_bullet(screen, env)
        draw_agent(screen, env)
        update_particles(screen)
        draw_popups(screen)
        draw_banner(screen)
        draw_flash(screen)
        draw_stage_bar(screen, stage, kills)
        draw_info_bar(screen, ep, num_episodes, kills, best_kills,
                      ep_return, action, stage, rendering, re_freq)
        draw_panel(screen, ep, num_episodes, stage,
                   h_ep, h_ret, h_avg, h_kills, best_kills)
        pygame.display.flip()

    for ep in range(1, num_episodes + 1):
        handle_events()
        do_render = rendering and (ep % re_freq == 0 or ep == 1)

        obs, _ = env.reset(seed=seed + ep)
        _particles.clear()
        _popups.clear()

        done        = False
        ep_return   = 0.0
        rewards_buf = []
        logprob_buf = []
        last_action = ACTION_WAIT
        prev_kills  = 0
        prev_stage  = 0
        z_snapshot  = []

        while not done:
            handle_events()

            action, log_prob = select_action(policy, obs)
            z_snapshot = [(z[0], z[1], z[2]) for z in env.zombies]

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return   += reward
            last_action  = action
            rewards_buf.append(reward)
            logprob_buf.append(log_prob)
            _tick += 1

            kills = info.get("kills", env.kills)
            stage = info.get("stage", env.stage)

            if do_render:
                # kill particles
                if kills > prev_kills:
                    for zb in z_snapshot:
                        if zb[2]:
                            killed = not any(
                                z[2] and z[0]==zb[0] and z[1]==zb[1]
                                for z in env.zombies)
                            if killed:
                                cx, cy = cell_center(zb[0], zb[1])
                                scol   = STAGE_COLS[stage]
                                spawn_particles(cx, cy, scol, n=26)
                                spawn_particles(cx, cy, (255, 255, 180), n=8)
                                spawn_popup(cx, cy,
                                            f"+{10 + stage*5}", GREEN_C)
                                trigger_flash(scol, 60)
                                break

                # stage advance banner
                if stage > prev_stage:
                    if stage < len(STAGE_DEFS) - 1:
                        trigger_banner(f"STAGE {stage+1}  {STAGE_NAMES[stage]}!", STAGE_COLS[stage])
                    else:
                        trigger_banner("∞  INFINITE MODE!", STAGE_COLS[stage])
                    trigger_flash(STAGE_COLS[stage], 120)

                # death flash
                if terminated:
                    trigger_flash(RED_C, 160)

            prev_kills = kills
            prev_stage = stage

            if do_render:
                render_frame(ep, ep_return, kills, last_action, stage)
                clock.tick(30)

        # REINFORCE update
        returns = compute_returns(rewards_buf, gamma=gamma)
        loss    = reinforce_loss(logprob_buf, returns, normalize=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        kills = info.get("kills", env.kills)
        best_kills = max(best_kills, kills)
        window50.append(ep_return)

        if ep % 10 == 0 or ep == 1:
            h_ep.append(ep)
            h_ret.append(ep_return)
            h_avg.append(float(np.mean(window50)))
            h_kills.append(float(kills))

        if do_render:
            render_frame(ep, ep_return, kills, last_action, env.stage)
            time.sleep(0.12)

    # ── done ─────────────────────────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    torch.save(policy.state_dict(), "outputs/zombie_policy.pth")
    screen.fill(BG)
    draw_panel(screen, num_episodes, num_episodes, env.stage,
               h_ep, h_ret, h_avg, h_kills, best_kills)
    msgs = [
        _fnt_xl.render("Training Complete!",            True, GOLD_C),
        _fnt_lg.render(f"Best score: {best_kills} kills  — Policy saved", True, GREEN_C),
        _fnt_sm.render("Press any key to exit",         True, DIM_C),
    ]
    cy = PAD + GRID_PX//2 - 40
    for s in msgs:
        screen.blit(s, s.get_rect(center=(PAD + GRID_PX//2, cy)))
        cy += s.get_height() + 12
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN):
                waiting = False
    pygame.quit()
    print(f"Best score: {best_kills} kills | Saved: outputs/zombie_policy.pth")


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
