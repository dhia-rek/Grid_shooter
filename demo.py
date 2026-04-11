"""
demo.py — Watch the trained agent play in the terminal.

Run AFTER train.py:
    python demo.py
    python demo.py --episodes 3 --delay 0.5
"""

import argparse
import time
import torch

from envs.grid_shooter_env import (
    GridShooterEnv,
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SHOOT, ACTION_WAIT,
)
from agent.reinforce_agent import PolicyNet

ACTION_NAMES = {
    ACTION_UP:    "UP",
    ACTION_DOWN:  "DOWN",
    ACTION_LEFT:  "LEFT",
    ACTION_RIGHT: "RIGHT",
    ACTION_SHOOT: "SHOOT",
    ACTION_WAIT:  "WAIT",
}


def run_demo(num_episodes=5, delay=0.3, grid_size=6, max_steps=50, seed=500):
    env = GridShooterEnv(grid_size=grid_size, max_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    policy = PolicyNet(obs_dim=obs.shape[0], n_actions=env.action_space.n)
    try:
        policy.load_state_dict(
            torch.load("outputs/grid_shooter_policy.pth", map_location="cpu"))
        policy.eval()
        print("Loaded: outputs/grid_shooter_policy.pth")
    except FileNotFoundError:
        print("ERROR: outputs/grid_shooter_policy.pth not found. Run python train.py first.")
        return

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        done, step, ep_return = False, 0, 0.0
        info = {}
        print(f"\n{'='*30}\n  Episode {ep} / {num_episodes}\n{'='*30}")
        time.sleep(delay)

        while not done:
            with torch.no_grad():
                s = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = int(torch.argmax(policy(s), dim=1).item())

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated
            step += 1

            print(f"\n  Step {step}  |  Action: {ACTION_NAMES[action]:<6}"
                  f"|  Reward: {reward:+.2f}  |  Return: {ep_return:+.2f}")
            env.render()
            time.sleep(delay)

        result = info.get("result", "timeout")
        emoji  = "🏆" if result == "win" else "💀" if "lose" in result else "⏱"
        print(f"\n  {emoji}  {result.upper()}   return {ep_return:+.2f}   steps {step}")
        time.sleep(delay * 3)

    print("\n  Demo finished.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",  type=int,   default=5)
    p.add_argument("--delay",     type=float, default=0.3)
    p.add_argument("--grid_size", type=int,   default=6)
    p.add_argument("--max_steps", type=int,   default=50)
    p.add_argument("--seed",      type=int,   default=500)
    args = p.parse_args()
    run_demo(args.episodes, args.delay, args.grid_size, args.max_steps, args.seed)
