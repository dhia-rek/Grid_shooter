"""
REINFORCE policy gradient agent.

PolicyNet       → policy network π_θ(a|s)
select_action() → sample action + log prob + entropy from policy
compute_returns()→ discounted reward-to-go G_t
reinforce_loss() → REINFORCE loss with optional entropy regularisation
"""

import torch
import torch.nn as nn
import torch.distributions as D


class PolicyNet(nn.Module):
    """Stochastic policy network: obs → action logits (two hidden layers, 256 units)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def select_action(policy: PolicyNet, obs):
    """Sample one action from π_θ(·|obs). Returns (action, log_prob, entropy)."""
    state_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    dist    = D.Categorical(logits=policy(state_t))
    action  = dist.sample()
    return action.item(), dist.log_prob(action), dist.entropy()


def compute_returns(rewards: list, gamma: float = 0.99) -> torch.Tensor:
    """Discounted reward-to-go: G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …"""
    returns, G = [], 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def reinforce_loss(log_probs: list, returns: torch.Tensor,
                   normalize: bool = True,
                   baseline: float = 0.0,
                   entropies: list = None,
                   entropy_coeff: float = 0.01) -> torch.Tensor:
    """REINFORCE loss = −Σ log π(a|s)·(G_t − b)  minus an entropy bonus.

    baseline: running EMA of past returns, subtracted before normalisation
              so the gradient reflects whether this episode was above average.
    """
    returns = returns - baseline
    if normalize and len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    log_probs_t = torch.stack(log_probs).squeeze(-1)
    policy_loss = -(log_probs_t * returns).sum()
    if entropies is not None and entropy_coeff > 0:
        entropy_bonus = torch.stack(entropies).mean()
        return policy_loss - entropy_coeff * entropy_bonus
    return policy_loss
