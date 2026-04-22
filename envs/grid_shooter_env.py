"""
grid_shooter_env.py — Staged Grid Shooter (no timer, death-based episodes).

Stages advance by kill count. Last stage is infinite max-difficulty.
Episode ends only when a zombie reaches the agent.

Stage 1 — Recruit   : kill  5 → advance  (top only)
Stage 2 — Soldier   : kill 15 → advance  (top + sides)
Stage 3 — Veteran   : kill 30 → advance  (all 4 directions)
Stage 4 — INFINITE  : endless, max speed, all directions
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

ACTION_UP    = 0
ACTION_DOWN  = 1
ACTION_LEFT  = 2
ACTION_RIGHT = 3
ACTION_SHOOT = 4
ACTION_WAIT  = 5

ACTION_NAMES = {
    ACTION_UP: "UP", ACTION_DOWN: "DOWN", ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT", ACTION_SHOOT: "SHOOT", ACTION_WAIT: "WAIT",
}

# Zombie movement directions
DIR_DOWN  = 0   # spawns at top row,    moves down
DIR_UP    = 1   # spawns at bottom row, moves up
DIR_RIGHT = 2   # spawns at left col,   moves right
DIR_LEFT  = 3   # spawns at right col,  moves left

DIR_NAMES = {DIR_DOWN: "↓", DIR_UP: "↑", DIR_RIGHT: "→", DIR_LEFT: "←"}

# Per-stage allowed spawn directions
STAGE_DIRS = [
    [DIR_DOWN],                           # Stage 1 — top only
    [DIR_DOWN, DIR_RIGHT, DIR_LEFT],      # Stage 2 — top + sides
    [DIR_DOWN, DIR_UP, DIR_RIGHT, DIR_LEFT],  # Stage 3 — all 4
    [DIR_DOWN, DIR_UP, DIR_RIGHT, DIR_LEFT],  # Stage 4 — all 4
]

GRID      = 8
MAX_Z     = 10      # max zombie slots in state vector
MAX_STEPS = 4000    # safety cap (very generous — effectively no timer)

# Stage definitions: (kill_threshold_to_advance, spawn_every, zombie_move_every, max_active)
STAGE_DEFS = [
    (  5,  9, 10,  3),   # Stage 1 — Recruit
    ( 15,  6,  7,  5),   # Stage 2 — Soldier
    ( 30,  4,  5,  7),   # Stage 3 — Veteran
    (9999, 2,  3, 10),   # Stage 4 — INFINITE (never advances)
]
STAGE_NAMES = ["Recruit", "Soldier", "Veteran", "∞  INFINITE"]


class GridShooterEnv(gym.Env):

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, grid_size=GRID, max_steps=MAX_STEPS):
        super().__init__()
        self.G         = grid_size
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(6)

        # obs: agent(2) + bullet(3) + zombies(MAX_Z*4) + stage(1) + kills_norm(1)
        # each zombie slot: (x/g, y/g, alive, dir/3)
        obs_dim = 2 + 3 + MAX_Z * 4 + 1 + 1
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
        reward = -0.05   # tiny step penalty

        # 1. Agent action
        _moves = {ACTION_UP:(0,-1), ACTION_DOWN:(0,1), ACTION_LEFT:(-1,0), ACTION_RIGHT:(1,0)}
        if action in _moves:
            dx, dy = _moves[action]
            nx = np.clip(self.agent_pos[0] + dx, 0, self.G - 1)
            ny = np.clip(self.agent_pos[1] + dy, 0, self.G - 1)
            self.agent_pos[:] = [nx, ny]
        elif action == ACTION_SHOOT and self.bullet is None:
            self.bullet = np.array(
                [self.agent_pos[0], self.agent_pos[1] - 1], dtype=np.int32)
            if any(z[2] and z[0] == self.bullet[0] for z in self.zombies):
                reward += 0.5   # alignment bonus

        # 2. Move bullet up
        if self.bullet is not None:
            self.bullet[1] -= 1
            for z in self.zombies:
                if z[2] and z[0] == self.bullet[0] and z[1] == self.bullet[1]:
                    z[2] = 0
                    self.kills += 1
                    kill_bonus = 10.0 + self.stage * 5.0
                    reward += kill_bonus
                    self.bullet = None
                    break
            if self.bullet is not None and self.bullet[1] < 0:
                self.bullet = None

        # 3. Move zombies (speed depends on stage)
        _, _, zombie_step, _ = STAGE_DEFS[self.stage]
        if self.steps % zombie_step == 0:
            for z in self.zombies:
                if not z[2]:
                    continue
                if z[3] == DIR_DOWN:
                    z[1] += 1
                elif z[3] == DIR_UP:
                    z[1] -= 1
                elif z[3] == DIR_RIGHT:
                    z[0] += 1
                else:  # DIR_LEFT
                    z[0] -= 1

        # 4. Check collisions
        ax, ay = self.agent_pos
        for z in self.zombies:
            if z[2] and z[0] == ax and z[1] == ay:
                reward -= 20.0
                return self._obs(), reward, True, False, {
                    "result": "dead", "kills": self.kills, "stage": self.stage}

        # 5. Purge dead / off-screen zombies
        self.zombies = [z for z in self.zombies if z[2] and self._on_screen(z)]

        # 6. Spawn
        _, spawn_every, _, max_active = STAGE_DEFS[self.stage]
        self.spawn_timer += 1
        if self.spawn_timer >= spawn_every:
            self.spawn_timer = 0
            if len(self.zombies) < max_active:
                self._spawn_zombie()

        # 7. Stage advancement
        self.just_advanced = False
        if self.stage < len(STAGE_DEFS) - 1:
            threshold = STAGE_DEFS[self.stage][0]
            if self.kills >= threshold:
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
        return z[0] >= 0   # DIR_LEFT

    def _spawn_zombie(self):
        G   = self.G
        ax, ay = self.agent_pos
        dirs = STAGE_DIRS[self.stage]
        d    = dirs[int(self.np_random.integers(0, len(dirs)))]

        if d == DIR_DOWN:
            x = int(self.np_random.integers(0, G))
            y = 0
            if y == ay and x == ax:
                x = (x + 1) % G
        elif d == DIR_UP:
            x = int(self.np_random.integers(0, G))
            y = G - 1
            if y == ay and x == ax:
                x = (x + 1) % G
        elif d == DIR_RIGHT:
            x = 0
            y = int(self.np_random.integers(0, G))
            if x == ax and y == ay:
                y = (y + 1) % G
        else:  # DIR_LEFT
            x = G - 1
            y = int(self.np_random.integers(0, G))
            if x == ax and y == ay:
                y = (y + 1) % G

        self.zombies.append([x, y, 1, d])

    def _obs(self):
        g = float(self.G)
        ax, ay = self.agent_pos / g
        if self.bullet is not None:
            bx, by, ba = self.bullet[0]/g, self.bullet[1]/g, 1.0
        else:
            bx, by, ba = 0.0, 0.0, 0.0

        alive = sorted(
            [z for z in self.zombies if z[2]],
            key=lambda z: abs(z[0] - self.agent_pos[0]) + abs(z[1] - self.agent_pos[1])
        )
        slots = (alive + [[0, 0, 0, 0]] * MAX_Z)[:MAX_Z]
        # 4 values per slot: x, y, alive, direction (normalised)
        zobs = [v for z in slots
                for v in (z[0]/g, z[1]/g, float(z[2]), z[3]/3.0)]

        stage_n = self.stage / (len(STAGE_DEFS) - 1)
        kills_n = min(self.kills / 60.0, 1.0)
        return np.array([ax, ay, bx, by, ba] + zobs + [stage_n, kills_n],
                        dtype=np.float32)
