"""Exploration schedule (provided scaffold)."""

from __future__ import annotations


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    """Linearly anneal epsilon from ``start`` to ``end`` over ``decay_steps``, then hold at ``end``."""
    if decay_steps <= 0:
        return end
    fraction = min(max(step, 0) / decay_steps, 1.0)
    return start + fraction * (end - start)
