"""Acting and target-network management (learner-owned functions)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def select_action(
    q_network: nn.Module,
    state: np.ndarray,
    epsilon: float,
    num_actions: int,
    rng: np.random.Generator,
    device: torch.device | str = "cpu",
) -> int:
    """Epsilon-greedy action selection.

    With probability ``epsilon``, return a uniformly random action from
    ``range(num_actions)`` drawn with ``rng``. Otherwise return the action with the
    highest Q-value under ``q_network``. Evaluation calls this with ``epsilon=0.0``.

    Args:
        q_network: maps float32 observations ``[batch, obs_dim]`` to Q-values
            ``[batch, num_actions]``.
        state: one observation from the environment, shape ``[obs_dim]``.
        epsilon: exploration probability in ``[0, 1]``.
        num_actions: number of valid discrete actions.
        rng: random generator for all exploration randomness.
        device: device the network lives on.

    Returns:
        A plain Python ``int`` that ``env.step`` accepts.

    Constraints (checked by ``tests/test_select_action.py``):
        - the network is called with a batched input of shape ``[1, obs_dim]``;
        - gradient tracking is disabled while the network is queried.
    """
    action = None
    if rng.random() < epsilon:
        action = int(rng.choice(range(num_actions)))
    else:
        with torch.no_grad():
            logits = q_network(torch.tensor(state).unsqueeze(0).to(device))
            action = int(torch.argmax(logits))
    if action is None:
        raise ValueError(f"Agent Selected a 'None' action, must be a valid int in range range({num_actions})")
    return action

def should_sync_target(gradient_step: int, interval: int) -> bool:
    """Decide whether the training loop should hard-sync the target network now.

    ``gradient_step`` is the number of gradient steps completed so far (the training
    loop calls this right after each step, so it starts at 1). Return ``True`` exactly
    when ``gradient_step`` is a positive multiple of ``interval``. Raise ``ValueError``
    if ``interval`` is not positive.
    """
    if interval <= 0:
        raise ValueError(f"should_sync_target expects a positive interval, got {interval}")
    return (gradient_step % interval) == 0


def sync_target_network(online_network: nn.Module, target_network: nn.Module) -> None:
    """Hard update: make ``target_network``'s weights an exact copy of ``online_network``'s.

    Modify ``target_network`` in place. Afterwards the two networks must hold equal
    values but must not share storage — later online updates must not leak into the
    target network.
    """
    target_network.load_state_dict(online_network.state_dict())


def soft_update_target_network(online_network: nn.Module, target_network: nn.Module, tau: float) -> None:
    """Soft (Polyak) update: move the target network a fraction ``tau`` toward the online one.

    Where :func:`sync_target_network` replaces the target's weights wholesale every N
    gradient steps, a soft update runs after *every* gradient step and moves each target
    weight only part of the way::

        theta_target  <-  tau * theta_online  +  (1 - tau) * theta_target

    ``tau=1.0`` therefore degenerates into exactly the hard copy above, while a small
    ``tau`` (0.005 is the config default) makes the target a slow, continuously trailing
    average of the online weights rather than a periodically refreshed snapshot.

    Args:
        online_network: the network being trained. Read-only here — it must come out of
            this call bit-for-bit unchanged.
        target_network: the lagged network, updated **in place**. The training loop and
            the rest of the program hold references to this module, so rebinding or
            returning a new module has no effect; the existing tensors must be mutated.
        tau: interpolation coefficient in ``(0, 1]``.

    Returns:
        Nothing. The update is a side effect on ``target_network``.

    Contract (checked by ``tests/test_soft_target.py``):
        - every parameter of ``target_network`` ends at the blend above, elementwise;
        - ``online_network`` is unchanged;
        - the two networks must not end up sharing storage — a later in-place change to
          an online parameter must not show up in the target;
        - target parameters stay *leaf* tensors outside the autograd graph: still
          ``requires_grad``, but with no ``grad_fn`` and no ``.grad`` populated by this
          call. (Target weights are inputs to the Bellman target, never things the
          optimizer trains — ``compute_dqn_targets`` already relies on this.)

    Note: this repo's :class:`~dqn.network.QNetwork` is a plain MLP with no registered
    buffers, so its parameters are its entire state. A network with buffers (BatchNorm's
    running statistics, say) would need them handled separately; that is out of scope.
    """
    if not 0.0 < tau <= 1.0:
        raise ValueError(f"soft_update_target_network expects tau in (0, 1], got {tau}")

    with torch.no_grad():
        for online_param, target_param in zip(online_network.parameters(), target_network.parameters()):
            target_param.data.mul_(1 - tau)
            target_param.data.add_(tau * online_param)

    return None


def update_target_network(
    online_network: nn.Module,
    target_network: nn.Module,
    gradient_step: int,
    mode: str,
    interval: int,
    tau: float,
) -> str | None:
    """Apply whichever target-network update ``mode`` calls for after ``gradient_step``.

    This is the single seam the training loop calls after each gradient step, so the two
    update strategies stay comparable: only the target-network schedule differs between a
    hard run and a soft one.

    Returns:
        ``"soft"`` or ``"hard"`` naming the update performed, or ``None`` when this
        gradient step falls between hard syncs and nothing was done.
    """
    if mode == "soft":
        soft_update_target_network(online_network, target_network, tau)
        return "soft"
    if mode == "hard":
        if should_sync_target(gradient_step, interval):
            sync_target_network(online_network, target_network)
            return "hard"
        return None
    raise ValueError(f"update_target_network got an unknown mode {mode!r}; expected 'hard' or 'soft'")
