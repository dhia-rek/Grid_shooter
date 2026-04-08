"""
Grid Shooter Environment — Gymnasium-compatible.

6×6 grid, 1 agent vs 1 enemy.
State  (11 floats): agent pos, enemy pos+alive, agent bullet, enemy bullet
Actions (6):        UP / DOWN / LEFT / RIGHT / SHOOT / WAIT
Rewards:            +10 kill, −10 die, −0.1/step, +0.5 aligned shot
Modes:              full (moving enemy + bullets) | simple (static, no bullets)
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


class GridShooterEnv(gym.Env):
    """1v1 grid shooter. Episode ends on kill, death, or step timeout."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, grid_size: int = 6, max_steps: int = 50,
                 enemy_fire_rate: float = 0.3, simple_mode: bool = False):
        super().__init__()
        self.grid_size       = grid_size
        self.max_steps       = max_steps
        self.enemy_fire_rate = enemy_fire_rate
        self.simple_mode     = simple_mode

        self.action_space      = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(11,), dtype=np.float32)

        self.agent_pos    = None
        self.enemy_pos    = None
        self.enemy_alive  = False
        self.agent_bullet = None
        self.enemy_bullet = None
        self.steps        = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mid = self.grid_size // 2
        self.agent_pos    = np.array([mid, self.grid_size - 1], dtype=np.int32)
        self.enemy_pos    = np.array(
            [self.np_random.integers(0, self.grid_size), 0], dtype=np.int32)
        self.enemy_alive  = True
        self.agent_bullet = None
        self.enemy_bullet = None
        self.steps        = 0
        return self._obs(), {}

    def step(self, action: int):
        self.steps += 1
        reward = -0.1

        # 1. Agent move / shoot
        ax, ay = self.agent_pos
        if   action == ACTION_UP    and ay > 0:               self.agent_pos[1] -= 1
        elif action == ACTION_DOWN  and ay < self.grid_size-1: self.agent_pos[1] += 1
        elif action == ACTION_LEFT  and ax > 0:               self.agent_pos[0] -= 1
        elif action == ACTION_RIGHT and ax < self.grid_size-1: self.agent_pos[0] += 1
        elif action == ACTION_SHOOT and self.agent_bullet is None:
            self.agent_bullet = np.array(
                [self.agent_pos[0], self.agent_pos[1] - 1], dtype=np.int32)
            if self.enemy_alive and self.agent_bullet[0] == self.enemy_pos[0]:
                reward += 0.5

        # 2. Agent bullet moves up
        if self.agent_bullet is not None:
            self.agent_bullet[1] -= 1
            if self.enemy_alive and np.array_equal(self.agent_bullet, self.enemy_pos):
                reward += 10.0
                self.enemy_alive  = False
                self.agent_bullet = None
                return self._obs(), reward, True, False, {"result": "win"}
            if self.agent_bullet[1] < 0:
                self.agent_bullet = None

        # 3. Enemy behaviour (full mode only)
        if self.enemy_alive and not self.simple_mode:
            if self.steps % 2 == 0 and self.enemy_pos[1] < self.grid_size - 1:
                self.enemy_pos[1] += 1
            if (self.enemy_bullet is None
                    and self.np_random.random() < self.enemy_fire_rate):
                self.enemy_bullet = np.array(
                    [self.enemy_pos[0], self.enemy_pos[1] + 1], dtype=np.int32)
            if np.array_equal(self.enemy_pos, self.agent_pos):
                reward -= 10.0
                return self._obs(), reward, True, False, {"result": "lose_collision"}

        # 4. Enemy bullet moves down (full mode only)
        if self.enemy_bullet is not None and not self.simple_mode:
            self.enemy_bullet[1] += 1
            if np.array_equal(self.enemy_bullet, self.agent_pos):
                reward -= 10.0
                return self._obs(), reward, True, False, {"result": "lose_shot"}
            if self.enemy_bullet[1] >= self.grid_size:
                self.enemy_bullet = None

        truncated = self.steps >= self.max_steps
        return self._obs(), reward, False, truncated, \
               {"result": "timeout"} if truncated else {}

    def _obs(self) -> np.ndarray:
        g = float(self.grid_size)
        ax, ay = self.agent_pos / g
        ex, ey = self.enemy_pos / g
        ea     = 1.0 if self.enemy_alive else 0.0
        if self.agent_bullet is not None:
            abx, aby, ab = self.agent_bullet[0]/g, self.agent_bullet[1]/g, 1.0
        else:
            abx, aby, ab = 0.0, 0.0, 0.0
        if self.enemy_bullet is not None:
            ebx, eby, eb = self.enemy_bullet[0]/g, self.enemy_bullet[1]/g, 1.0
        else:
            ebx, eby, eb = 0.0, 0.0, 0.0
        return np.array([ax, ay, ex, ey, ea, abx, aby, ab, ebx, eby, eb],
                        dtype=np.float32)

    def render(self):
        g = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        if self.enemy_alive:
            g[self.enemy_pos[1]][self.enemy_pos[0]] = "E"
        if self.enemy_bullet is not None:
            bx, by = self.enemy_bullet
            if 0 <= by < self.grid_size:
                g[by][bx] = "v"
        if self.agent_bullet is not None:
            bx, by = self.agent_bullet
            if 0 <= by < self.grid_size:
                g[by][bx] = "^"
        g[self.agent_pos[1]][self.agent_pos[0]] = "A"
        print("\n".join(" ".join(row) for row in g))
        print(f"step={self.steps}  mode={'simple' if self.simple_mode else 'full'}")
