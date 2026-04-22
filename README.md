# Grid Shooter — REINFORCE Agent

A reinforcement learning project implementing the REINFORCE policy gradient
algorithm on a custom Gymnasium environment: a staged zombie shooter with
escalating difficulty and multi-directional enemies.

---

## Project Structure

```
Grid_shooter/
├── envs/
│   └── grid_shooter_env.py    # Staged zombie shooter (8×8, 4 stages, 4 directions)
│
├── agent/
│   └── reinforce_agent.py     # REINFORCE: PolicyNet, loss, returns
│
├── outputs/                   # Saved models (auto-generated)
│
├── visual_zombie.py           # Main entry point — live Pygame training visualiser
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

python visual_zombie.py                              # 3000 episodes, full render
python visual_zombie.py --episodes 5000 --render_every 5   # faster
```

---

## Environment — `GridShooterEnv`

- **8×8 grid**, agent starts at bottom-centre
- **Actions**: UP / DOWN / LEFT / RIGHT / SHOOT / WAIT
- **Bullet**: fires upward; hits any zombie in its column path
- **Episode**: ends when a zombie reaches the agent (no time limit)

### Zombie directions (unlock per stage)

| Stage | Name     | Advance at | Directions active        | Spawn | Speed | Max |
|-------|----------|-----------|--------------------------|-------|-------|-----|
| 1     | Recruit  | 5 kills   | ↓ top only               | /9    | /10   | 3   |
| 2     | Soldier  | 15 kills  | ↓ top + ← → sides       | /6    | /7    | 5   |
| 3     | Veteran  | 30 kills  | ↓ ↑ ← → all 4 dirs      | /4    | /5    | 7   |
| 4     | INFINITE | —         | ↓ ↑ ← → all 4 dirs      | /2    | /3    | 10  |

Each zombie displays a direction arrow (↑ ↓ ← →) in the Pygame window.

### Observation vector (47 floats)

| Slice      | Contents                                       |
|------------|------------------------------------------------|
| `[0:2]`    | Agent position (x/G, y/G)                      |
| `[2:5]`    | Bullet (x/G, y/G, active)                      |
| `[5:45]`   | 10 zombie slots × 4: (x/G, y/G, alive, dir/3) |
| `[45]`     | Stage (normalised 0–1)                         |
| `[46]`     | Kill count (normalised, capped at 60)          |

### Rewards

| Event              | Reward                          |
|--------------------|---------------------------------|
| Kill a zombie      | +10 + 5×stage (scales up)       |
| Aligned shot fired | +0.5                            |
| Hit by zombie      | −20                             |
| Per step           | −0.05                           |

---

## Algorithm — REINFORCE

```
for each episode:
    1. collect_episode()   — roll out π_θ(a|s), record (s, a, r, log π)
    2. compute_returns()   — G_t = Σ γ^k · r_{t+k}  (discounted reward-to-go)
    3. reinforce_loss()    — L = −Σ log π_θ(aₜ|sₜ) · Gₜ  (normalised)
    4. optimizer.step()    — gradient ascent on expected return
```

Hyperparameters: `gamma=0.99`, `lr=1e-3`, `hidden=128`

---

## Visual Controls (Pygame)

| Key       | Action                              |
|-----------|-------------------------------------|
| `SPACE`   | Toggle rendering on/off             |
| `+` / `-` | Render every N episodes             |
| `ESC`/`Q` | Quit                                |

---

## What We've Implemented

1. **Single unified environment** (`GridShooterEnv`) — consolidated from two separate envs into one staged zombie shooter.
2. **Multi-directional zombies** — zombies spawn from all 4 edges; directions unlock progressively per stage (top-only → top+sides → all 4).
3. **Direction in observation** — each zombie slot includes a normalised direction value so the policy can distinguish approach angle.
4. **Direction arrows in visualiser** — Pygame renderer draws a filled triangle on each zombie indicating its movement direction.

## What We're Doing Next

- Directional shooting (agent fires toward last move direction)
- Curriculum learning hooks (auto-tune stage thresholds based on performance)
- Policy evaluation script adapted for the new env (kills/stage metrics)
