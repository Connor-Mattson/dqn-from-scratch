"""Q-network architecture (provided scaffold)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class QNetwork(nn.Module):
    """MLP approximating ``Q_theta(s, a)`` for every action at once.

    Input:  float32 observations, shape ``[batch, obs_dim]``.
    Output: Q-values, shape ``[batch, num_actions]`` (one column per action).

    The same class is used for both the online network and the target network.
    """

    def __init__(self, obs_dim: int, num_actions: int, hidden_sizes: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden_sizes = tuple(hidden_sizes)

        layers: list[nn.Module] = []
        in_features = obs_dim
        for width in self.hidden_sizes:
            layers += [nn.Linear(in_features, width), nn.ReLU()]
            in_features = width
        layers.append(nn.Linear(in_features, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
