"""Scoring utility functions for rules."""

from __future__ import annotations


def score_triangular(
    val: float, min_val: float, ideal_val: float, max_val: float
) -> float:
    """Calculate a triangular score peaking at ideal_val.

    Returns:
        0.0 if val < min_val or val > max_val
        Linear ramp from 0.0 to 1.0 between min_val and ideal_val
        Linear ramp from 1.0 to 0.0 between ideal_val and max_val
    """
    if val < min_val or val > max_val:
        return 0.0

    if val < ideal_val:
        if ideal_val == min_val:  # Avoid division by zero
            return 1.0
        return (val - min_val) / (ideal_val - min_val)
    else:
        if max_val == ideal_val:  # Avoid division by zero
            return 1.0
        return 1.0 - (val - ideal_val) / (max_val - ideal_val)


def score_linear(
    val: float,
    min_val: float,
    max_val: float,
    min_score: float = 0.0,
    max_score: float = 1.0,
) -> float:
    """Calculate a linear score between min_val and max_val.

    Returns:
        min_score if val <= min_val
        max_score if val >= max_val
        Linear interpolation between min_score and max_score otherwise
    """
    if val <= min_val:
        return min_score
    if val >= max_val:
        return max_score

    fraction = (val - min_val) / (max_val - min_val)
    return min_score + fraction * (max_score - min_score)


def score_exponential_decay(val: float, scale: float) -> float:
    """Calculate exponential decay score: exp(-val / scale)."""
    import math

    return math.exp(-val / scale)
