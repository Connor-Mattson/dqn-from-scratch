"""Bellman targets and the DQN optimisation step (learner-owned functions).

Loss choice for this scaffold: Smooth L1 / Huber loss with ``beta=1.0`` and mean
reduction, i.e. the defaults of ``torch.nn.functional.smooth_l1_loss``. The tests
pin this choice.

Two bootstrap targets live here and are selected by the ``double`` flag that
:func:`dqn_loss` and :func:`training_step` take:

- :func:`compute_dqn_targets` — vanilla DQN. One network both selects and values the
  next action.
- :func:`compute_double_dqn_targets` — Double DQN. The two roles are split, which is
  what removes the systematic upward bias in the vanilla ``max``.
"""

from __future__ import annotations

import torch
from torch import nn

from dqn.replay_buffer import Batch


def compute_dqn_targets(
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    target_network: nn.Module,
) -> torch.Tensor:
    """Compute DQN bootstrap targets for a minibatch.

    Args:
        rewards: float32, shape ``[B]``.
        next_states: float32, shape ``[B, obs_dim]``.
        dones: float32, shape ``[B]``; 1.0 marks a terminal transition.
        gamma: discount factor.
        target_network: the lagged network ``Q_target``.

    Returns:
        float32 tensor of shape ``[B]`` that does not require grad, where

        - terminal transition:     ``target = reward``
        - non-terminal transition: ``target = reward + gamma * max_a' Q_target(next_state, a')``
    """
    with torch.no_grad():
        s_prime_predictions = target_network(next_states)
        best_q_primes = torch.max(s_prime_predictions, dim=1).values
        return torch.where(
            dones.bool(),
            rewards,
            rewards + (gamma * best_q_primes)
        )


def compute_double_dqn_targets(
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    online_network: nn.Module,
    target_network: nn.Module,
) -> torch.Tensor:
    """Compute Double DQN bootstrap targets for a minibatch.

    Vanilla DQN asks one network both *which* action is best in ``s'`` and *how much* it
    is worth. Whichever action's value that network happens to overestimate is the one its
    ``max`` selects, so the error does not average out: it accumulates upward and inflates
    every target that bootstraps through it. Double DQN gives the two questions to
    different networks — the online network names the action, the target network prices
    it — so an action has to be overvalued by *both* to survive into the target.

    Args:
        rewards: float32, shape ``[B]``.
        next_states: float32, shape ``[B, obs_dim]``.
        dones: float32, shape ``[B]``; 1.0 marks a terminal transition.
        gamma: discount factor.
        online_network: the network being trained, ``Q_online``. **Selects** the next
            action here; it is never trained through this call.
        target_network: the lagged network, ``Q_target``. **Values** the selected action.

    Returns:
        float32 tensor of shape ``[B]`` that does not require grad, where

        - terminal transition:     ``target = reward``
        - non-terminal transition: ``target = reward + gamma * Q_target(s', a*)``
          with ``a* = argmax_a' Q_online(s', a')``, chosen per row.

    Notes and edge cases:
        - The action index comes from one network and the value from the other. Both are
          ``[B, num_actions]`` outputs, and the result is ``[B]``: one value per row, at
          that row's own ``a*``. A global max or a ``[B, num_actions]`` return is a bug —
          see the shape discussion in :func:`dqn_loss`.
        - When both networks happen to rank ``s'``'s actions the same way, this returns
          exactly what :func:`compute_dqn_targets` would. The functions diverge only where
          the two networks disagree about the best action, which is precisely where the
          vanilla target was trusting its own noise.
        - Nothing here may leak gradients: this call sits inside the loss, so an attached
          graph would train the online network to raise its own targets, and would put a
          ``grad_fn`` on a value the optimizer is supposed to treat as a constant.
        - Terminal rows bootstrap nothing, so neither network's opinion of ``s'`` may
          reach the result for them.
    """
    with torch.no_grad():
        s_prime_online_prediction = online_network(next_states)
        a_prime_online_greedy = torch.argmax(s_prime_online_prediction, dim=1, keepdim=True)
        s_prime_predictions = target_network(next_states)
        q_primes = torch.gather(s_prime_predictions, 1, a_prime_online_greedy).squeeze(1)
        return torch.where(
            dones.bool(),
            rewards,
            rewards + (gamma * q_primes)
        )


def dqn_loss(
    online_network: nn.Module,
    target_network: nn.Module,
    batch: Batch,
    gamma: float,
    double: bool = False,
) -> torch.Tensor:
    """Huber loss between ``Q_online(state, action)`` and the DQN targets.

    Only the Q-values of the actions actually taken in ``batch`` contribute. Targets come
    from :func:`compute_dqn_targets`, or from :func:`compute_double_dqn_targets` when
    ``double`` is set — the *only* difference between a vanilla and a Double DQN run.

    Shapes (the whole difficulty of this function):
        ``online_network(batch.states)`` is ``[B, num_actions]`` — every action's value.
        ``batch.actions`` is int64 ``[B]`` — which action was actually taken per row.
        The prediction handed to the loss must be ``[B]``: one value per row, chosen by
        that row's action. The targets are already ``[B]``.

    A ``[B, num_actions]`` prediction against a ``[B]`` target is always a bug: when
    ``num_actions == B`` PyTorch broadcasts to ``[B, B]`` and silently returns a wrong
    scalar; otherwise it raises. Neither is the contract.

    Returns:
        A scalar (0-dim) tensor whose gradients flow into ``online_network`` only,
        and only into the entries for the actions in ``batch.actions``.
    """
    q_learner_eval = online_network(batch.states)
    q_learner = torch.gather(
        q_learner_eval, 
        dim=1, 
        index=batch.actions.unsqueeze(1)
    ).squeeze(1)
    if double:
        q_target = compute_double_dqn_targets(
            batch.rewards,
            batch.next_states,
            batch.dones,
            gamma,
            online_network,
            target_network
        )
    else:
        q_target = compute_dqn_targets(
            batch.rewards,
            batch.next_states,
            batch.dones,
            gamma,
            target_network
        )
    return torch.nn.HuberLoss()(q_learner, q_target)


def training_step(
    online_network: nn.Module,
    target_network: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    gamma: float,
    max_grad_norm: float | None = None,
    double: bool = False,
) -> float:
    """Run one gradient update of the online network on ``batch``.

    ``optimizer`` was built over ``online_network.parameters()`` only. ``double`` selects
    the Double DQN bootstrap target and is passed straight through to :func:`dqn_loss`;
    it changes what the online network is pulled *toward*, never which parameters move.

    After this call (checked by ``tests/test_dqn_learning.py``):
        - online-network parameters have been updated by exactly one optimizer step
          using gradients from this batch only;
        - target-network parameters are unchanged and have no gradients (``.grad is None``);
        - if ``max_grad_norm`` is not None, the online gradients' total norm was clipped
          to ``max_grad_norm`` before the optimizer step.

    Returns:
        The loss value as a Python float, for logging. ``train.py`` formats it with
        ``f"{last_loss:.4f}"``, and the test asserts ``type(loss) is float`` — a tensor
        or ``None`` does not satisfy this.
    """
    optimizer.zero_grad()
    
    loss = dqn_loss(online_network, target_network, batch, gamma, double)
    loss.backward()

    # Gradient clipping
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(online_network.parameters(), max_norm=max_grad_norm)

    optimizer.step()
    return float(loss.detach())