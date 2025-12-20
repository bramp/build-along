"""Scale functions for scoring rules.

This module provides abstractions for different scoring scales used by rules.
Instead of hardcoding scoring logic in each rule, rules can use these
reusable scale functions.

Key Scale Types:
- LinearScale: Multi-point piecewise linear interpolation. Supports 2+ points,
  with linear interpolation between each pair. Values beyond the range are
  clamped to endpoint scores. This is the most flexible scale type.
  For triangular patterns, use 3 points: {min: 0.0, ideal: 1.0, max: 0.0}.
- BooleanScale: Simple threshold-based binary scoring.
- DiscreteScale: Maps value ranges to specific scores.
- ExponentialDecayScale: Exponential decay function.
- ThresholdScale: Multi-threshold step function.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class ScaleFunction(ABC):
    """Abstract base for scale functions that map values to scores."""

    @abstractmethod
    def __call__(self, value: float) -> float:
        """Map a value to a score [0.0, 1.0]."""
        pass


class BooleanScale(ScaleFunction):
    """Boolean scale: returns 1.0 if value >= threshold, else 0.0."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, value: float) -> float:
        return 1.0 if value >= self.threshold else 0.0


class LinearScale(ScaleFunction):
    """Linear scale: linearly interpolates between multiple points.

    Supports dict initialization for readability: LinearScale({value: score, ...})
    Maps each value to its corresponding score, with linear interpolation
    between points. Values beyond the range are clamped to the nearest
    endpoint score.

    Supports both ascending (score increases with value) and descending (score decreases
    with value) scales, as well as arbitrary multi-point piecewise linear functions.

    Example:
        LinearScale({0.0: 1.0, 0.4: 0.0})  # 1.0 at 0.0, 0.0 at 0.4 (2 points)
        LinearScale({0.0: 0.0, 0.5: 1.0, 1.0: 0.0})  # Triangular (3 points)
        LinearScale({0.0: 0.0, 1.0: 1.0})  # Ascending (2 points)
        LinearScale(
            min_val=0.0, max_val=1.0, min_score=0.0, max_score=1.0
        )  # 2-point form
    """

    def __init__(
        self,
        min_val: float | dict[float, float] | None = None,
        max_val: float | None = None,
        min_score: float = 0.0,
        max_score: float = 1.0,
    ):
        # Support dict-based initialization: {value: score, ...}
        if isinstance(min_val, dict):
            if len(min_val) < 2:
                raise ValueError("Dict must have at least 2 entries: {value: score}")
            # Store as sorted list of (value, score) tuples
            self.points = sorted(min_val.items(), key=lambda x: x[0])
        else:
            # Legacy 2-point initialization
            if min_val is None or max_val is None:
                raise ValueError("Must provide min_val and max_val, or a dict")
            self.points = [(min_val, min_score), (max_val, max_score)]

    def __call__(self, value: float) -> float:
        # Clamp to first point if below range
        if value <= self.points[0][0]:
            return self.points[0][1]

        # Clamp to last point if above range
        if value >= self.points[-1][0]:
            return self.points[-1][1]

        # Find the two points to interpolate between
        for i in range(len(self.points) - 1):
            val1, score1 = self.points[i]
            val2, score2 = self.points[i + 1]

            if val1 <= value <= val2:
                # Linear interpolation between these two points
                if val2 == val1:  # Avoid division by zero
                    return score1
                fraction = (value - val1) / (val2 - val1)
                return score1 + fraction * (score2 - score1)

        # Should never reach here due to clamping checks above
        return self.points[-1][1]


class ExponentialDecayScale(ScaleFunction):
    """Exponential decay scale: exp(-value / scale)."""

    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, value: float) -> float:
        return math.exp(-value / self.scale)


class DiscreteScale(ScaleFunction):
    """Discrete scale: maps value ranges to specific scores.

    Example:
        DiscreteScale({
            (0, 10): 0.5,
            (10, 20): 0.8,
            (20, float('inf')): 1.0
        })
    """

    def __init__(self, ranges: dict[tuple[float, float], float], default: float = 0.0):
        """Initialize discrete scale.

        Args:
            ranges: Dict mapping (min, max) tuples to scores
            default: Default score if value doesn't match any range
        """
        self.ranges = sorted(ranges.items(), key=lambda x: x[0][0])
        self.default = default

    def __call__(self, value: float) -> float:
        for (min_val, max_val), score in self.ranges:
            if min_val <= value < max_val:
                return score
        return self.default


class ThresholdScale(ScaleFunction):
    """Multi-threshold scale: returns different scores based on thresholds.

    Example:
        ThresholdScale([
            (0.9, 1.0),   # value >= 0.9 -> 1.0
            (0.7, 0.7),   # value >= 0.7 -> 0.7
            (0.5, 0.3),   # value >= 0.5 -> 0.3
        ], default=0.0)  # value < 0.5 -> 0.0
    """

    def __init__(self, thresholds: list[tuple[float, float]], default: float = 0.0):
        """Initialize threshold scale.

        Args:
            thresholds: List of (threshold, score) tuples,
                sorted descending by threshold
            default: Default score if below all thresholds
        """
        self.thresholds = sorted(thresholds, key=lambda x: x[0], reverse=True)
        self.default = default

    def __call__(self, value: float) -> float:
        for threshold, score in self.thresholds:
            if value >= threshold:
                return score
        return self.default


class InvertedScale(ScaleFunction):
    """Inverts another scale function: returns 1.0 - scale(value)."""

    def __init__(self, scale: ScaleFunction):
        self.scale = scale

    def __call__(self, value: float) -> float:
        return 1.0 - self.scale(value)


class ClampedScale(ScaleFunction):
    """Clamps the output of another scale to [min_score, max_score]."""

    def __init__(
        self, scale: ScaleFunction, min_score: float = 0.0, max_score: float = 1.0
    ):
        self.scale = scale
        self.min_score = min_score
        self.max_score = max_score

    def __call__(self, value: float) -> float:
        score = self.scale(value)
        return max(self.min_score, min(self.max_score, score))


class RangeCheckScale(ScaleFunction):
    """Checks if value is within a range, returns 1.0 if in range, 0.0 otherwise.

    Useful for boolean range filters.
    """

    def __init__(self, min_val: float | None = None, max_val: float | None = None):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, value: float) -> float:
        if self.min_val is not None and value < self.min_val:
            return 0.0
        if self.max_val is not None and value > self.max_val:
            return 0.0
        return 1.0
