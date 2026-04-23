# Grid Shooter — REINFORCE Agent

A reinforcement learning project implementing the REINFORCE policy gradient
algorithm on a custom Gymnasium environment: a staged zombie shooter with
escalating difficulty, multi-directional enemies, and directional shooting.

---

## Project Structure

```
Grid_shooter/
├── envs/
│   └── grid_shooter_env.py   # The game environment (rules, physics, rewards)
├── agent/
│   └── reinforce_agent.py    # The RL algorithm (PolicyNet, training functions)
├── renderer.py               # All Pygame drawing code (visuals only, no game logic)
├── visual_zombie.py          # Training entry point  (imports renderer + env + agent)
├── view_policy.py            # Policy viewer         (imports renderer + env + agent)
└── outputs/                  # Auto-generated: weights, history, plots
```

### How the files relate

```
visual_zombie.py          view_policy.py
  (training loop)           (greedy playback)
       │                          │
       ├──────────────────────────┤
       │                          │
       ▼                          ▼
  renderer.py          ──►   renderer.py
  (draws everything)         (draws everything)
       │
       ▼
  envs/grid_shooter_env.py   agent/reinforce_agent.py
  (game rules & state)       (policy network & REINFORCE)
```

**Why separated this way:**

- `envs/grid_shooter_env.py` — pure game logic. No Pygame, no PyTorch. Knows nothing about how it is displayed or trained.
- `agent/reinforce_agent.py` — pure RL code. `PolicyNet`, `select_action`, `compute_returns`, `reinforce_loss`. No Pygame, no game rules.
- `renderer.py` — pure Pygame drawing. Takes the env state and draws it. Knows nothing about training or rewards.
- `visual_zombie.py` — the glue for **training**: runs the REINFORCE loop, calls `renderer.py` to display each step, saves outputs when done.
- `view_policy.py` — the glue for **watching**: loads saved weights, runs the agent greedily, calls `renderer.py` to display.

---

## Quick Start

```bash
pip install -r requirements.txt

# Train
python visual_zombie.py
python visual_zombie.py --episodes 5000 --render_every 5   # faster

# Watch trained agent
python view_policy.py
python view_policy.py --episodes 10 --delay 120
```

---

## Environment — `GridShooterEnv`

**8×8 grid.** The agent starts at the bottom-centre. Zombies spawn from
the grid edges and march inward. The episode ends only when a zombie
reaches the agent — there is no time limit.

### Actions (9)

| Code | Action       |
|------|--------------|
| 0–3  | Move UP / DOWN / LEFT / RIGHT |
| 4–7  | SHOOT ↑ / ↓ / ← / → |
| 8    | WAIT |

The agent must explicitly choose a shoot direction. Firing in the wrong
direction misses entirely — the policy has to learn to aim.

### Reward shaping

| Event                   | Reward              |
|-------------------------|---------------------|
| Kill a zombie           | +10 + 5 × stage     |
| Shoot aligned with zombie | +0.5 bonus        |
| Hit by zombie           | −20 (terminal)      |
| Per step                | −0.05               |

Kill reward scales with stage so the agent keeps incentive to fight even
as difficulty rises. The alignment bonus nudges early exploration toward
aimed shots rather than random firing.

### Stages and zombie directions

Directions unlock progressively so the agent is not overwhelmed from the start.

| Stage | Name     | Advance at | Directions        | Spawn | Speed | Max |
|-------|----------|-----------|-------------------|-------|-------|-----|
| 1     | Recruit  | 5 kills   | ↓ top only        | /9    | /10   | 3   |
| 2     | Soldier  | 15 kills  | ↓ ← → top+sides   | /6    | /7    | 5   |
| 3     | Veteran  | 30 kills  | ↓ ↑ ← → all 4    | /4    | /5    | 7   |
| 4     | INFINITE | —         | ↓ ↑ ← → all 4    | /2    | /3    | 10  |

### Observation vector (48 floats)

| Slice    | Contents                                          |
|----------|---------------------------------------------------|
| `[0:2]`  | Agent position (x/G, y/G)                         |
| `[2:6]`  | Bullet: x/G, y/G, active, direction/3             |
| `[6:46]` | 10 zombie slots × 4: (x/G, y/G, alive, dir/3)    |
| `[46]`   | Stage normalised (0–1)                            |
| `[47]`   | Kill count normalised (capped at 60)              |

---

## Algorithm — REINFORCE

```
for each episode:
    1. collect_episode()   — roll out π_θ(a|s), record (s, a, r, log π)
    2. compute_returns()   — G_t = Σ γ^k · r_{t+k}
    3. reinforce_loss()    — L = −Σ log π_θ(aₜ|sₜ) · Gₜ  (normalised)
    4. optimizer.step()    — gradient ascent on expected return
```

Hyperparameters: `gamma=0.99` · `lr=1e-3` · `hidden=128`

---

## What We Built and Why It Changed

### Started with two separate environments

The project originally had two environments: a simple 1v1 grid shooter
(agent vs one enemy, 6×6) and a separate zombie shooter. They shared no
code and made the project harder to reason about. We consolidated into one
environment — `GridShooterEnv` — based on the zombie shooter, which is
richer and more interesting for learning.

### Bullet was upward-only — side zombies were unkillable

The original zombie env fired bullets straight up regardless of action.
When we added zombies from the left, right, and bottom, those zombies
became pure obstacles: the agent could only dodge them, never shoot them.
The policy had no reason to develop any targeting behaviour for most of
the threats it faced.

We replaced the single `SHOOT` action with four directional shoot actions.
Now the bullet carries a direction vector `[x, y, dx, dy]` and travels
in whichever direction the agent chose. This means every zombie on the
grid is a potential target.

### AI behaviour with directional shooting

With 9 actions the policy now has to learn:

- **Aim** — choose the shoot direction that intercepts the nearest threat
- **Reposition** — move to place a zombie in its line of fire before shooting
- **Dodge** — evade zombies it cannot currently shoot without taking a hit
- **Prioritise** — decide whether to shoot or move when multiple threats converge

This is meaningfully harder than the upward-only version, but the
behaviour that emerges is closer to real game strategy.

### Challenges

**Action space growth.** Going from 6 to 9 actions slows early exploration.
The alignment bonus (+0.5 when a zombie is in the bullet's path) was added
specifically to guide the policy toward aimed shots during the random phase
before it has learned anything useful.

**Direction in observation.** Each zombie slot needed its direction encoded
so the policy can distinguish a zombie moving toward it from one moving
away. Similarly the active bullet's direction is in the observation so the
agent knows where its shot is going without re-inferring it from position
changes.

**Progressive stage difficulty.** Releasing all four directions from stage 1
caused the agent to be surrounded immediately and never learn to shoot at
all. Gating directions per stage (top-only → top+sides → all four) gives
the policy time to develop basic shoot-and-dodge skills before the
full chaos of stage 3 and 4.

**Off-screen purge.** Side and bottom zombies leave the grid in different
directions than top zombies. Each direction needs its own boundary check
or zombies accumulate off-screen and inflate the active count incorrectly.

---

## Visual Controls (Pygame)

| Key       | Action                    |
|-----------|---------------------------|
| `SPACE`   | Toggle rendering on/off   |
| `+` / `-` | Render every N episodes   |
| `ESC`/`Q` | Quit                      |
