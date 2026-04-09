"""
REINFORCE policy gradient agent.

PolicyNet       → policy network π_θ(a|s)
select_action() → sample action + log prob from policy
collect_episode()→ roll out one full episode
compute_returns()→ discounted reward-to-go G_t
reinforce_loss() → REINFORCE loss: −Σ log π(a|s) · G_t
"""

import torch
import torch.nn as nn
import torch.distributions as D


class PolicyNet(nn.Module):
    """Stochastic policy network: obs → action logits."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def select_action(policy: PolicyNet, obs):
    """Sample one action from π_θ(·|obs). Returns (action, log_prob)."""
    state_t  = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    dist     = D.Categorical(logits=policy(state_t))
    action   = dist.sample()
    return action.item(), dist.log_prob(action)


def collect_episode(env, policy: PolicyNet):
    """Roll out one episode. Returns (rewards, log_probs, info)."""
    rewards, log_probs = [], []
    obs, _ = env.reset()
    done   = False
    info   = {}
    while not done:
        action, log_prob = select_action(policy, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards.append(reward)
        log_probs.append(log_prob)
    return rewards, log_probs, info


def compute_returns(rewards: list, gamma: float = 0.99) -> torch.Tensor:
    """Discounted reward-to-go: G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …"""
    returns, G = [], 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def reinforce_loss(log_probs: list, returns: torch.Tensor,
                   normalize: bool = True) -> torch.Tensor:
    """REINFORCE loss = −Σ log π(a|s) · G_t  (normalised returns by default)."""
    if normalize and len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    log_probs_t = torch.stack(log_probs).squeeze(-1)
    return -(log_probs_t * returns).sum()
