"""
grid_shooter_env.py — Staged Grid Shooter (no timer, death-based episodes).

Stages advance by kill count. Last stage is infinite max-difficulty.
Episode ends only when a zombie reaches the agent.

Stage 1 — Recruit   : kill  5 → advance  (zombies from top only)
Stage 2 — Soldier   : kill 15 → advance  (top + sides)
Stage 3 — Veteran   : kill 30 → advance  (all 4 directions)
Stage 4 — INFINITE  : endless, max speed, all directions

Actions (9):
  0-3  Move:  UP / DOWN / LEFT / RIGHT
  4-7  Shoot: SHOOT_UP / SHOOT_DOWN / SHOOT_LEFT / SHOOT_RIGHT
  8    WAIT
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

ACTION_UP          = 0
ACTION_DOWN        = 1
ACTION_LEFT        = 2
ACTION_RIGHT       = 3
ACTION_SHOOT_UP    = 4
ACTION_SHOOT_DOWN  = 5
ACTION_SHOOT_LEFT  = 6
ACTION_SHOOT_RIGHT = 7
ACTION_WAIT        = 8

ACTION_NAMES = {
    ACTION_UP:          "UP",
    ACTION_DOWN:        "DOWN",
    ACTION_LEFT:        "LEFT",
    ACTION_RIGHT:       "RIGHT",
    ACTION_SHOOT_UP:    "SHOOT ↑",
    ACTION_SHOOT_DOWN:  "SHOOT ↓",
    ACTION_SHOOT_LEFT:  "SHOOT ←",
    ACTION_SHOOT_RIGHT: "SHOOT →",
    ACTION_WAIT:        "WAIT",
}

# Bullet direction vectors per shoot action
_SHOOT_VEC = {
    ACTION_SHOOT_UP:    ( 0, -1),
    ACTION_SHOOT_DOWN:  ( 0,  1),
    ACTION_SHOOT_LEFT:  (-1,  0),
    ACTION_SHOOT_RIGHT: ( 1,  0),
}

# Move direction vectors (module-level so step() doesn't rebuild each call)
_MOVE_VEC = {
    ACTION_UP:    ( 0, -1),
    ACTION_DOWN:  ( 0,  1),
    ACTION_LEFT:  (-1,  0),
    ACTION_RIGHT: ( 1,  0),
}

# Bullet direction index for obs encoding (UP=0 DOWN=1 LEFT=2 RIGHT=3)
_VEC_TO_BDIR = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}

# Zombie spawn directions
DIR_DOWN  = 0   # spawns top,   moves down
DIR_UP    = 1   # spawns bottom, moves up
DIR_RIGHT = 2   # spawns left,  moves right
DIR_LEFT  = 3   # spawns right, moves left

# Per-stage allowed spawn directions
STAGE_DIRS = [
    [DIR_DOWN],
    [DIR_DOWN, DIR_RIGHT, DIR_LEFT],
    [DIR_DOWN, DIR_UP, DIR_RIGHT, DIR_LEFT],
    [DIR_DOWN, DIR_UP, DIR_RIGHT, DIR_LEFT],
]

GRID      = 8
MAX_Z     = 10
MAX_STEPS = 4000

# (kill_threshold, spawn_every, zombie_move_every, max_active)
STAGE_DEFS = [
    (  5,  9, 10,  3),
    ( 15,  6,  7,  5),
    ( 30,  4,  5,  7),
    (9999, 2,  3, 10),
]
STAGE_NAMES = ["Recruit", "Soldier", "Veteran", "∞  INFINITE"]


class GridShooterEnv(gym.Env):

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, grid_size=GRID, max_steps=MAX_STEPS):
        super().__init__()
        self.G         = grid_size
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(9)

        # obs: agent(2) + bullet(4: x,y,active,dir/3) + zombies(MAX_Z*4) + stage(1) + kills(1)
        obs_dim = 2 + 4 + MAX_Z * 4 + 1 + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self._reset_state()

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        self._spawn_zombie()
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        reward = 0.0

        # 1. Agent move
        if action in _MOVE_VEC:
            dx, dy = _MOVE_VEC[action]
            self.agent_pos[0] = int(np.clip(self.agent_pos[0] + dx, 0, self.G - 1))
            self.agent_pos[1] = int(np.clip(self.agent_pos[1] + dy, 0, self.G - 1))

        # 2. Agent shoot
        elif action in _SHOOT_VEC and self.bullet is None:
            dx, dy = _SHOOT_VEC[action]
            bx = self.agent_pos[0] + dx
            by = self.agent_pos[1] + dy
            if 0 <= bx < self.G and 0 <= by < self.G:
                self.bullet = np.array([bx, by, dx, dy], dtype=np.int32)
                if self._zombie_in_los(bx, by, dx, dy):
                    reward += 0.5   # alignment bonus

        # 3. Advance bullet
        if self.bullet is not None:
            bx, by, bdx, bdy = self.bullet
            for z in self.zombies:
                if z[2] and z[0] == bx and z[1] == by:
                    z[2] = 0
                    self.kills += 1
                    reward += 10.0 + self.stage * 5.0
                    self.bullet = None
                    break
            if self.bullet is not None:
                nx, ny = bx + bdx, by + bdy
                if 0 <= nx < self.G and 0 <= ny < self.G:
                    self.bullet[:2] = [nx, ny]
                else:
                    self.bullet = None

        # 4. Move zombies
        _, _, zombie_step, _ = STAGE_DEFS[self.stage]
        if self.steps % zombie_step == 0:
            for z in self.zombies:
                if not z[2]:
                    continue
                if   z[3] == DIR_DOWN:  z[1] += 1
                elif z[3] == DIR_UP:    z[1] -= 1
                elif z[3] == DIR_RIGHT: z[0] += 1
                else:                   z[0] -= 1  # DIR_LEFT

        # 5. Collision check
        ax, ay = self.agent_pos
        for z in self.zombies:
            if z[2] and z[0] == ax and z[1] == ay:
                reward -= 20.0
                return self._obs(), reward, True, False, {
                    "result": "dead", "kills": self.kills, "stage": self.stage}

        # 6. Purge off-screen / dead zombies
        self.zombies = [z for z in self.zombies if z[2] and self._on_screen(z)]

        # 7. Spawn
        _, spawn_every, _, max_active = STAGE_DEFS[self.stage]
        self.spawn_timer += 1
        if self.spawn_timer >= spawn_every and len(self.zombies) < max_active:
            self.spawn_timer = 0
            self._spawn_zombie()

        # 8. Stage advancement
        self.just_advanced = False
        if self.stage < len(STAGE_DEFS) - 1 and self.kills >= STAGE_DEFS[self.stage][0]:
            self.stage += 1
            self.just_advanced = True
            self.spawn_timer = 0

        truncated = self.steps >= self.max_steps
        return self._obs(), reward, False, truncated, {
            "result": "survived" if truncated else "alive",
            "kills": self.kills, "stage": self.stage,
            "just_advanced": self.just_advanced}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _reset_state(self):
        mid = self.G // 2
        self.agent_pos    = np.array([mid, self.G - 1], dtype=np.int32)
        self.bullet       = None
        self.zombies      = []
        self.steps        = 0
        self.kills        = 0
        self.stage        = 0
        self.spawn_timer  = 0
        self.just_advanced = False

    def _on_screen(self, z):
        d = z[3]
        if d == DIR_DOWN:  return z[1] < self.G
        if d == DIR_UP:    return z[1] >= 0
        if d == DIR_RIGHT: return z[0] < self.G
        return z[0] >= 0

    def _zombie_in_los(self, bx, by, dx, dy):
        """True if any zombie lies along the bullet's line of sight."""
        for z in self.zombies:
            if not z[2]:
                continue
            if dx == 0 and z[0] == bx and (dy * (z[1] - by) > 0):
                return True
            if dy == 0 and z[1] == by and (dx * (z[0] - bx) > 0):
                return True
        return False

    def _spawn_zombie(self):
        G  = self.G
        ax, ay = self.agent_pos
        dirs = STAGE_DIRS[self.stage]
        d    = dirs[int(self.np_random.integers(0, len(dirs)))]

        if d == DIR_DOWN:
            x, y = int(self.np_random.integers(0, G)), 0
            if x == ax and y == ay: x = (x + 1) % G
        elif d == DIR_UP:
            x, y = int(self.np_random.integers(0, G)), G - 1
            if x == ax and y == ay: x = (x + 1) % G
        elif d == DIR_RIGHT:
            x, y = 0, int(self.np_random.integers(0, G))
            if x == ax and y == ay: y = (y + 1) % G
        else:  # DIR_LEFT
            x, y = G - 1, int(self.np_random.integers(0, G))
            if x == ax and y == ay: y = (y + 1) % G

        self.zombies.append([x, y, 1, d])

    def _obs(self):
        g = float(self.G)
        ax, ay = self.agent_pos / g

        if self.bullet is not None:
            bx   = self.bullet[0] / g
            by   = self.bullet[1] / g
            ba   = 1.0
            bdir = _VEC_TO_BDIR[(int(self.bullet[2]), int(self.bullet[3]))] / 3.0
        else:
            bx = by = ba = bdir = 0.0

        alive = sorted(
            [z for z in self.zombies if z[2]],
            key=lambda z: abs(z[0] - self.agent_pos[0]) + abs(z[1] - self.agent_pos[1])
        )
        slots = (alive + [[0, 0, 0, 0]] * MAX_Z)[:MAX_Z]
        zobs  = [v for z in slots
                 for v in (z[0]/g, z[1]/g, float(z[2]), z[3]/3.0)]

        stage_n = self.stage / (len(STAGE_DEFS) - 1)
        kills_n = min(self.kills / 60.0, 1.0)
        return np.array([ax, ay, bx, by, ba, bdir] + zobs + [stage_n, kills_n],
                        dtype=np.float32)
