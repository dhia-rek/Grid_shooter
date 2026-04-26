"""
renderer.py — All Pygame drawing code for the Grid Shooter visualiser.

Imported by visual_zombie.py (training) and view_policy.py (playback).
Contains no game logic, no training loop, no RL code.
"""

import math
import random

import pygame
import pygame.gfxdraw

from envs.grid_shooter_env import (
    ACTION_NAMES, GRID, STAGE_NAMES, STAGE_DEFS,
    DIR_DOWN, DIR_UP, DIR_RIGHT, DIR_LEFT,
)

# ── Layout ────────────────────────────────────────────────────────────────────
CELL    = 68
PAD     = 20
GRID_PX = CELL * GRID
PANEL_W = 450
INFO_H  = 125
WIN_W   = PAD + GRID_PX + PAD + PANEL_W + PAD
WIN_H   = PAD + GRID_PX + INFO_H

# ── Stage colours ─────────────────────────────────────────────────────────────
STAGE_COLS = [
    ( 60, 200,  80),   # Stage 1 — green
    ( 60, 160, 255),   # Stage 2 — blue
    (220, 120,  40),   # Stage 3 — orange
    (220,  50,  50),   # Stage 4 — red
]
STAGE_BG_TINTS = [(0,8,0), (0,0,12), (10,4,0), (14,0,0)]

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

# ── Fonts (set after pygame.init via init_fonts()) ────────────────────────────
_fnt_xl = _fnt_lg = _fnt_md = _fnt_sm = None


def init_fonts():
    global _fnt_xl, _fnt_lg, _fnt_md, _fnt_sm
    _fnt_xl = pygame.font.SysFont("monospace", 32, bold=True)
    _fnt_lg = pygame.font.SysFont("monospace", 20, bold=True)
    _fnt_md = pygame.font.SysFont("monospace", 16, bold=True)
    _fnt_sm = pygame.font.SysFont("monospace", 12)


# ── Animation tick (incremented each game step) ───────────────────────────────
_tick = 0


def bump_tick():
    global _tick
    _tick += 1


# ── Particle system ───────────────────────────────────────────────────────────
class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "col", "size")

    def __init__(self, x, y, col):
        a = random.uniform(0, 2 * math.pi)
        s = random.uniform(1.5, 5.0)
        self.x, self.y       = x, y
        self.vx, self.vy     = math.cos(a) * s, math.sin(a) * s
        self.max_life        = random.randint(14, 32)
        self.life            = self.max_life
        self.col             = col
        self.size            = random.randint(2, 5)

    def update(self):
        self.x  += self.vx
        self.y  += self.vy
        self.vy += 0.1
        self.vx *= 0.92
        self.life -= 1

    def draw(self, surf):
        a  = int(255 * self.life / self.max_life)
        sz = max(1, int(self.size * self.life / self.max_life))
        pygame.gfxdraw.filled_circle(surf, int(self.x), int(self.y), sz,
                                     (*self.col, a))


_particles: list = []
_popups:    list = []  # [cx, cy, text, col, life]
_banner_life = 0
_banner_text = ""
_banner_col  = (255, 255, 255)
_flash_alpha = 0
_flash_col   = (0, 0, 0)


def reset_effects():
    """Clear all active particles and popups (call at episode start)."""
    _particles.clear()
    _popups.clear()
    _zombie_sprites.clear()


def spawn_particles(x, y, col, n=22):
    for _ in range(n):
        _particles.append(Particle(x, y, col))


def update_particles(surf):
    for p in _particles:
        p.update()
        p.draw(surf)
    _particles[:] = [p for p in _particles if p.life > 0]


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
    screen.blit(s, s.get_rect(center=(PAD + GRID_PX // 2, PAD + GRID_PX // 2)))
    _banner_life -= 1


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


# ── Low-level helpers ─────────────────────────────────────────────────────────

def rr(surf, col, rect, r=8):
    pygame.draw.rect(surf, col, rect, border_radius=r)


def cell_center(gx, gy):
    return PAD + gx * CELL + CELL // 2, PAD + gy * CELL + CELL // 2


def cell_tl(gx, gy):
    return PAD + gx * CELL, PAD + gy * CELL


def draw_glow(surf, col, cx, cy, rad, steps=3):
    r, g, b = col
    for i in range(steps, 0, -1):
        a    = int(55 * i / steps)
        rad2 = rad + (steps - i + 1) * 5
        s    = pygame.Surface((rad2 * 2, rad2 * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(s, rad2, rad2, rad2, (r, g, b, a))
        surf.blit(s, (cx - rad2, cy - rad2))


# ── Grid ──────────────────────────────────────────────────────────────────────

def draw_grid_bg(screen, stage):
    tint = STAGE_BG_TINTS[stage]
    for gy in range(GRID):
        for gx in range(GRID):
            base = CELL_LIGHT if (gx + gy) % 2 == 0 else CELL_DARK
            col  = tuple(min(255, base[i] + tint[i]) for i in range(3))
            px, py = cell_tl(gx, gy)
            rr(screen, col, (px + 1, py + 1, CELL - 2, CELL - 2), 3)
    for i in range(GRID + 1):
        x = PAD + i * CELL
        y = PAD + i * CELL
        pygame.draw.line(screen, GRID_COL, (x, PAD),       (x, PAD + GRID_PX), 1)
        pygame.draw.line(screen, GRID_COL, (PAD, y), (PAD + GRID_PX, y),       1)


# ── Agent sprite (generated once, cached) ────────────────────────────────────

_agent_sprite = None


def _make_agent_sprite(size):
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    c = size // 2

    # --- shadow layer ---
    shadow = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.ellipse(shadow, (0, 0, 0, 60), (4, size // 2, size - 8, size // 2 - 2))
    s.blit(shadow, (0, 0))

    # --- left wing ---
    lw = [(c - 3, c - 4), (c - 20, c + 14), (c - 10, c + 16), (c - 4, c + 4)]
    pygame.draw.polygon(s, (20, 80, 180), lw)
    pygame.draw.polygon(s, (60, 140, 255), lw, 1)

    # --- right wing ---
    rw = [(c + 3, c - 4), (c + 20, c + 14), (c + 10, c + 16), (c + 4, c + 4)]
    pygame.draw.polygon(s, (20, 80, 180), rw)
    pygame.draw.polygon(s, (60, 140, 255), rw, 1)

    # --- engine exhaust glow ---
    for r, a in [(9, 40), (6, 80), (4, 140)]:
        pygame.gfxdraw.filled_circle(s, c, c + 18, r, (80, 160, 255, a))
    pygame.gfxdraw.filled_circle(s, c, c + 18, 3, (200, 230, 255, 255))

    # --- main hull ---
    hull = [
        (c,      4),           # nose
        (c + 10, c - 6),       # right shoulder
        (c + 8,  c + 14),      # right tail corner
        (c,      c + 18),      # tail centre
        (c - 8,  c + 14),      # left tail corner
        (c - 10, c - 6),       # left shoulder
    ]
    pygame.draw.polygon(s, (30, 110, 220), hull)
    pygame.draw.polygon(s, (100, 180, 255), hull, 2)

    # --- cockpit glass ---
    pygame.gfxdraw.filled_circle(s, c, c - 4, 9, (160, 220, 255, 210))
    pygame.gfxdraw.filled_circle(s, c, c - 4, 9, (100, 180, 255, 255))  # rim
    pygame.gfxdraw.filled_circle(s, c, c - 4, 7, (180, 230, 255, 180))  # inner fill
    # shine dot
    pygame.gfxdraw.filled_circle(s, c - 2, c - 7, 2, (255, 255, 255, 200))

    # --- hull centre stripe ---
    pygame.draw.line(s, (120, 200, 255), (c, c - 4 + 9), (c, c + 14), 2)

    return s


def draw_agent(screen, env):
    global _agent_sprite
    ax, ay = env.agent_pos
    cx, cy = cell_center(ax, ay)

    size = CELL - 14
    if _agent_sprite is None or _agent_sprite.get_width() != size:
        _agent_sprite = _make_agent_sprite(size)

    draw_glow(screen, AGENT_C, cx, cy, CELL // 2 - 6, steps=3)
    screen.blit(_agent_sprite, _agent_sprite.get_rect(center=(cx, cy)))


# ── Zombie sprite (one per stage, cached; rotated per direction) ──────────────

_zombie_sprites: dict = {}  # keyed by stage index

# Sprite is drawn facing UP by default; rotate to match movement direction.
_DIR_ROTATE = {
    DIR_UP:    0,
    DIR_DOWN:  180,
    DIR_LEFT:  -90,
    DIR_RIGHT:  90,
}


def _make_zombie_sprite(size, col):
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    c = size // 2
    r = size // 2 - 1

    # Outer glow halo
    for ri, a in [(r + 4, 18), (r + 2, 35)]:
        pygame.gfxdraw.filled_circle(s, c, c, ri, (*col, a))

    # Dark body
    body_col = tuple(max(0, x // 4) for x in col)
    pygame.gfxdraw.filled_circle(s, c, c, r, (*body_col, 255))

    # Mid-tone texture ring
    mid_col = tuple(x // 2 for x in col)
    for ri in range(r - 4, r - 1):
        pygame.gfxdraw.aacircle(s, c, c, ri, (*mid_col, 120))

    # Bright outline
    pygame.gfxdraw.aacircle(s, c, c, r, (*col, 255))
    pygame.gfxdraw.aacircle(s, c, c, r - 1, (*col, 180))

    # Forehead scar (top, since sprite faces UP)
    scar_col = tuple(min(255, x + 70) for x in col)
    pygame.draw.lines(s, scar_col, False,
                      [(c + 1, c - r + 5), (c - 1, c - r + 10), (c + 2, c - r + 16)], 1)

    # Eyes — glowing red, near top of sprite (the "face" direction)
    eye_y = c - r // 3
    for ex in [c - r // 3, c + r // 3]:
        pygame.gfxdraw.filled_circle(s, ex, eye_y, 6, (140, 10, 10, 255))
        pygame.gfxdraw.filled_circle(s, ex, eye_y, 5, (255, 50, 50, 255))
        pygame.gfxdraw.filled_circle(s, ex, eye_y, 3, (255, 140, 60, 255))
        pygame.gfxdraw.filled_circle(s, ex, eye_y, 1, (255, 240, 200, 255))

    # Mouth — jagged teeth, below centre
    my = c + r // 4
    mw = r - 5
    pygame.draw.line(s, (60, 5, 5), (c - mw, my), (c + mw, my), 2)
    tw = (mw * 2) // 4
    for i in range(4):
        tx = c - mw + i * tw
        pts = [(tx + 1, my), (tx + tw // 2, my + 7), (tx + tw - 1, my)]
        pygame.draw.polygon(s, (210, 205, 190), pts)
        pygame.draw.polygon(s, (140, 135, 120), pts, 1)

    return s


def draw_zombies(screen, env):
    zcol = STAGE_COLS[env.stage]
    size = CELL - 14

    if env.stage not in _zombie_sprites:
        _zombie_sprites[env.stage] = _make_zombie_sprite(size, zcol)
    base_sprite = _zombie_sprites[env.stage]

    for z in env.zombies:
        if not z[2]:
            continue
        gx, gy = z[0], z[1]
        if not (0 <= gx < GRID and 0 <= gy < GRID):
            continue
        cx, cy = cell_center(gx, gy)

        pulse = 0.75 + 0.25 * math.sin(_tick * 0.14 + gx * 0.9)
        glow_col = tuple(int(x * pulse) for x in zcol)
        draw_glow(screen, glow_col, cx, cy, CELL // 2 - 8, steps=2)

        angle = _DIR_ROTATE[z[3]]
        sprite = pygame.transform.rotate(base_sprite, angle) if angle != 0 else base_sprite
        screen.blit(sprite, sprite.get_rect(center=(cx, cy)))


# ── Bullet ────────────────────────────────────────────────────────────────────

def draw_bullet(screen, env):
    if env.bullet is None:
        return
    bx, by, bdx, bdy = env.bullet
    if not (0 <= bx < GRID and 0 <= by < GRID):
        return
    cx, cy = cell_center(bx, by)
    for i in range(1, 6):
        alpha = int(160 * (1 - i / 6))
        tx = cx - bdx * i * 7
        ty = cy - bdy * i * 7
        pygame.gfxdraw.filled_circle(screen, int(tx), int(ty),
                                     max(1, 4 - i), (*BULLET_C, alpha))
    draw_glow(screen, BULLET_C, cx, cy, 7, steps=2)
    pygame.gfxdraw.filled_circle(screen, cx, cy, 5, (*BULLET_C, 255))
    pygame.gfxdraw.filled_circle(screen, cx, cy, 2, (255, 255, 255, 255))


# ── Stage progress bar ────────────────────────────────────────────────────────

def draw_stage_bar(screen, stage, kills):
    bx, by = PAD, PAD + GRID_PX + 8
    bw, bh = GRID_PX, 22
    col = STAGE_COLS[stage]
    rr(screen, (20, 20, 36), (bx, by, bw, bh), 6)
    if stage < len(STAGE_DEFS) - 1:
        threshold  = STAGE_DEFS[stage][0]
        prev_kills = STAGE_DEFS[stage - 1][0] if stage > 0 else 0
        pct = min(1.0, (kills - prev_kills) / max(1, threshold - prev_kills))
        rr(screen, col, (bx, by, max(6, int(bw * pct)), bh), 6)
        lbl = _fnt_sm.render(
            f"  STAGE {stage+1}  {STAGE_NAMES[stage]}    "
            f"kills {kills} / {threshold} → next stage", True, TEXT_C)
    else:
        pulse_w = int(bw * (0.5 + 0.5 * math.sin(_tick * 0.08)))
        rr(screen, col, (bx, by, pulse_w, bh), 6)
        lbl = _fnt_sm.render(
            f"  ∞  INFINITE STAGE    kills {kills}   SCORE IT ALL!", True, TEXT_C)
    screen.blit(lbl, lbl.get_rect(midleft=(bx + 6, by + bh // 2)))


# ── Info bar (below grid) ─────────────────────────────────────────────────────

def draw_info_bar(screen, ep, total_ep, kills, best_kills,
                  ep_return, action, stage, rendering, re_freq):
    bx, by = PAD, PAD + GRID_PX + 34
    bw, bh = GRID_PX, INFO_H - 44
    rr(screen, PANEL_BG, (bx, by, bw, bh), 8)
    y = by + 8
    rows = [
        (f"Episode {ep:>5}/{total_ep}     "
         f"Score(kills) {kills:>3}     Best {best_kills:>3}     Return {ep_return:+.1f}",
         TEXT_C),
        (f"Action  {ACTION_NAMES.get(action, '-'):<10}     "
         f"Stage  {stage+1} — {STAGE_NAMES[stage]}",
         STAGE_COLS[stage]),
        (f"{'[RENDER ON  — SPACE]' if rendering else '[RENDER OFF — SPACE]'}"
         f"   speed +/-  (every {re_freq} ep)",
         DIM_C),
    ]
    for txt, col in rows:
        screen.blit(_fnt_sm.render(txt, True, col), (bx + 10, y))
        y += 22


# ── Right stats panel ─────────────────────────────────────────────────────────

def draw_panel(screen, ep, total_ep, stage, h_ep, h_ret, h_avg, h_kills, best_kills):
    rx, ry = PAD + GRID_PX + PAD, PAD

    t = _fnt_lg.render("REINFORCE — Grid Shooter", True, GOLD_C)
    screen.blit(t, (rx, ry))
    ry += t.get_height() + 4

    scol  = STAGE_COLS[stage]
    badge = _fnt_lg.render(f"Stage {stage+1}  {STAGE_NAMES[stage]}", True, scol)
    screen.blit(badge, (rx, ry))
    ry += badge.get_height() + 6

    bw, bh = PANEL_W, 14
    pct = ep / total_ep
    rr(screen, (22, 22, 40), (rx, ry, bw, bh), 5)
    rr(screen, BLUE_C,       (rx, ry, int(bw * pct), bh), 5)
    screen.blit(_fnt_sm.render(f"Training  {ep}/{total_ep}  ({pct*100:.0f}%)", True, DIM_C),
                (rx + 6, ry + 1))
    ry += bh + 10

    if h_ret:
        stats = [
            ("Return",   f"{h_ret[-1]:+.1f}",   BLUE_C),
            ("Avg-50",   f"{h_avg[-1]:+.1f}",   GOLD_C),
            ("Kills/ep", f"{h_kills[-1]:.0f}",  GREEN_C),
            ("Best",     f"{best_kills}",         ORANGE_C),
        ]
        bxw = (PANEL_W - 3 * 6) // 4
        for i, (label, val, col) in enumerate(stats):
            bx2 = rx + i * (bxw + 6)
            rr(screen, (16, 16, 30), (bx2, ry, bxw, 54), 8)
            pygame.draw.rect(screen, col, (bx2, ry, bxw, 54), 1, border_radius=8)
            v = _fnt_md.render(val,   True, col)
            l = _fnt_sm.render(label, True, DIM_C)
            screen.blit(v, v.get_rect(center=(bx2 + bxw // 2, ry + 20)))
            screen.blit(l, l.get_rect(center=(bx2 + bxw // 2, ry + 40)))
        ry += 60

    ch = WIN_H - ry - 10
    if ch > 60:
        draw_chart(screen, h_ep, h_ret, h_avg, h_kills, rx, ry, PANEL_W, ch)


def draw_chart(screen, h_ep, h_ret, h_avg, h_kills, cx, cy, w, h):
    rr(screen, CHART_BG, (cx, cy, w, h), 8)
    pygame.draw.rect(screen, GRID_COL, (cx, cy, w, h), 1, border_radius=8)
    for pct in [0.25, 0.5, 0.75]:
        yy = cy + int(h * pct)
        pygame.draw.line(screen, (22, 22, 36), (cx + 1, yy), (cx + w - 1, yy), 1)

    n = len(h_ep)
    if n < 2:
        screen.blit(_fnt_sm.render("Waiting for data...", True, DIM_C), (cx + 10, cy + 10))
        return

    xs = [cx + int(i / (n - 1) * (w - 2)) for i in range(n)]

    def ys(vals, mg=12):
        lo, hi = min(vals), max(vals)
        span = hi - lo or 1.0
        return [cy + h - mg - int((v - lo) / span * (h - mg * 2)) for v in vals]

    def draw_line(vals, col, thick=2):
        pts = list(zip(xs, ys(vals)))
        for i in range(1, len(pts)):
            pygame.draw.line(screen, col, pts[i - 1], pts[i], thick)

    draw_line(h_ret, (*BLUE_C, 160), 1)
    draw_line(h_avg, GOLD_C, 2)

    mk   = max(h_kills) if max(h_kills) > 0 else 1
    ys_k = [cy + h - 10 - int(v / mk * (h - 20)) for v in h_kills]
    for i in range(1, len(xs)):
        pygame.draw.line(screen, GREEN_C, (xs[i-1], ys_k[i-1]), (xs[i], ys_k[i]), 2)

    lx = cx + 8
    for label, col in [("Return", BLUE_C), ("Avg-50", GOLD_C), ("Kills", GREEN_C)]:
        pygame.draw.line(screen, col, (lx, cy + h - 12), (lx + 16, cy + h - 12), 2)
        screen.blit(_fnt_sm.render(label, True, col), (lx + 20, cy + h - 18))
        lx += 110
