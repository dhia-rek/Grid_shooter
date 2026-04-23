"""
view_policy.py — Watch the trained agent play.

Loads outputs/zombie_policy.pth and runs it greedily (no training).
All drawing code is in renderer.py.

Usage:
    python view_policy.py
    python view_policy.py --episodes 10 --delay 80
    python view_policy.py --model outputs/zombie_policy.pth
"""

import argparse
import sys

import numpy as np
import torch
import pygame

import renderer as R
from envs.grid_shooter_env import GridShooterEnv, STAGE_NAMES, ACTION_WAIT
from agent.reinforce_agent import PolicyNet


def run(num_episodes=5, delay_ms=80, model_path="outputs/zombie_policy.pth", seed=42):

    pygame.init()
    R.init_fonts()
    screen = pygame.display.set_mode((R.WIN_W, R.WIN_H))
    pygame.display.set_caption("Grid Shooter — Policy Viewer")
    clock  = pygame.time.Clock()

    env    = GridShooterEnv()
    obs, _ = env.reset(seed=seed)
    policy = PolicyNet(obs_dim=obs.shape[0], n_actions=env.action_space.n)

    try:
        policy.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True))
    except FileNotFoundError:
        print(f"ERROR: {model_path} not found — run  python visual_zombie.py  first.")
        pygame.quit()
        return

    policy.eval()
    best_kills = 0

    def handle_events():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit(); sys.exit()

    def render_frame(ep, ep_return, kills, action, stage, result):
        screen.fill(R.BG)
        R.draw_grid_bg(screen, stage)
        R.draw_zombies(screen, env)
        R.draw_bullet(screen, env)
        R.draw_agent(screen, env)
        R.update_particles(screen)
        R.draw_popups(screen)
        R.draw_flash(screen)
        R.draw_stage_bar(screen, stage, kills)
        _draw_viewer_hud(screen, ep, num_episodes, kills, best_kills,
                         ep_return, action, stage, result)
        pygame.display.flip()

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        R.reset_effects()

        done        = False
        ep_return   = 0.0
        last_action = ACTION_WAIT
        result      = None
        kills       = 0

        while not done:
            handle_events()

            with torch.no_grad():
                s = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                last_action = int(torch.argmax(policy(s), dim=1).item())

            obs, reward, terminated, truncated, info = env.step(last_action)
            done       = terminated or truncated
            ep_return += reward
            kills      = info.get("kills", env.kills)
            R.bump_tick()

            if done:
                result     = info.get("result", "survived")
                best_kills = max(best_kills, kills)
                if result == "dead":
                    R.trigger_flash(R.RED_C, 160)

            render_frame(ep, ep_return, kills, last_action, env.stage, result)
            clock.tick(1000 // max(1, delay_ms))

        pygame.time.wait(800)

    # end screen
    screen.fill(R.BG)
    msgs = [
        (R._fnt_lg, f"Done — {num_episodes} episodes",    R.GOLD_C),
        (R._fnt_md, f"Best score: {best_kills} kills",     R.GREEN_C),
        (R._fnt_sm, "Press any key to exit",               R.DIM_C),
    ]
    cy = R.WIN_H // 2 - 40
    for fnt, txt, col in msgs:
        s = fnt.render(txt, True, col)
        screen.blit(s, s.get_rect(center=(R.WIN_W // 2, cy)))
        cy += s.get_height() + 14
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN):
                waiting = False
    pygame.quit()


def _draw_viewer_hud(screen, ep, total, kills, best, stage, action, ep_return, result):
    from envs.grid_shooter_env import ACTION_NAMES
    bx, by = R.PAD, R.PAD + R.GRID_PX + 8
    bw, bh = R.GRID_PX, R.INFO_H - 16
    R.rr(screen, R.PANEL_BG, (bx, by, bw, bh), 8)
    res_col = R.GREEN_C if result == "survived" else R.RED_C if result == "dead" else R.TEXT_C
    rows = [
        (f"Episode {ep}/{total}   Stage {stage+1} — {STAGE_NAMES[stage]}"
         f"   Kills {kills}   Best {best}   Return {ep_return:+.1f}",
         R.STAGE_COLS[stage]),
        (f"Action: {ACTION_NAMES.get(action, '-'):<12}   "
         f"Result: {result or 'playing'}",
         res_col),
        ("ESC / Q — quit", R.DIM_C),
    ]
    y = by + 8
    for txt, col in rows:
        screen.blit(R._fnt_sm.render(txt, True, col), (bx + 10, y))
        y += 22


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int,   default=5)
    p.add_argument("--delay",    type=int,   default=80,
                   help="Milliseconds between frames (default 80)")
    p.add_argument("--model",    type=str,   default="outputs/zombie_policy.pth")
    p.add_argument("--seed",     type=int,   default=42)
    args = p.parse_args()
    run(args.episodes, args.delay, args.model, args.seed)
