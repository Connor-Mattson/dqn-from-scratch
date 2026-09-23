"""Test doubles shared by the learner-task tests."""

from __future__ import annotations

import torch
from torch import nn


class TableQNetwork(nn.Module):
    """Returns a fixed, trainable table of Q-values instead of computing them from observations.

    ``forward(obs)`` requires batched observations ``[B, obs_dim]`` and returns the first ``B``
    rows of the table, so each test controls exactly which Q-values a network reports. Every
    call records ``(input shape, grad enabled)`` in ``self.calls``.
    """

    def __init__(self, q_values: list[list[float]], obs_dim: int = 4) -> None:
        super().__init__()
        self.q_values = nn.Parameter(torch.tensor(q_values, dtype=torch.float32))
        self.obs_dim = obs_dim
        self.num_actions = self.q_values.shape[1]
        self.calls: list[tuple[tuple[int, ...], bool]] = []

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        assert obs.ndim == 2 and obs.shape[1] == self.obs_dim, (
            f"expected observations shaped [batch, {self.obs_dim}], got {tuple(obs.shape)}"
        )
        assert obs.shape[0] <= self.q_values.shape[0], "batch is larger than the Q-table in this test"
        self.calls.append((tuple(obs.shape), torch.is_grad_enabled()))
        return self.q_values[: obs.shape[0]]
