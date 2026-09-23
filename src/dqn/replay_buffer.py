"""Experience replay.

Provided: the ``Batch`` container, which fixes the shape/dtype contract.
Learner-owned: ``ReplayBuffer`` storage, eviction, and sampling.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch


class Batch(NamedTuple):
    """A minibatch of transitions. ``B`` = batch size, ``D`` = observation dimension.

    states:      float32 tensor, shape [B, D]
    actions:     int64   tensor, shape [B]
    rewards:     float32 tensor, shape [B]
    next_states: float32 tensor, shape [B, D]
    dones:       float32 tensor, shape [B]; 1.0 for terminal transitions, 0.0 otherwise
    """

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor

    def to(self, device: torch.device | str) -> Batch:
        return Batch(*(tensor.to(device) for tensor in self))


class ReplayBuffer:
    """Fixed-capacity store of ``(state, action, reward, next_state, done)`` transitions.

    Contract (checked by ``tests/test_replay_buffer.py``):

    - ``len(buffer)`` is the number of stored transitions and never exceeds ``capacity``;
    - once full, pushing a new transition replaces the oldest stored transition;
    - ``sample(batch_size)`` returns a :class:`Batch` of ``batch_size`` distinct stored
      transitions chosen uniformly at random with ``self.rng``, as CPU tensors that follow
      the ``Batch`` shape/dtype contract;
    - ``sample`` raises ``ValueError`` when ``batch_size`` exceeds ``len(buffer)``.
    """

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.buffer = []

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition, replacing the oldest one when the buffer is full."""
        if len(self) + 1 > self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Batch:
        """Return ``batch_size`` distinct random transitions as a ``Batch``."""
        if batch_size > len(self):
            raise ValueError(f"Batch Size must not be larger than the buffer size, got {batch_size} > {len(self)}")

        choices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in choices:
            c = self.buffer[i]
            states.append(c[0])
            actions.append(c[1])
            rewards.append(c[2])
            next_states.append(c[3])
            dones.append(c[4])

        batch = Batch(
            states=torch.Tensor(np.array(states)),
            actions=torch.tensor(actions, dtype=torch.int64),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            next_states=torch.Tensor(np.array(next_states)),
            dones=torch.Tensor(dones)
        )
        return batch
        

    def __len__(self) -> int:
        return len(self.buffer)
