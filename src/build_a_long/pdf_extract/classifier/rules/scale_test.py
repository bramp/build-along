"""Tests for scale functions."""

from __future__ import annotations

import math

import pytest

from build_a_long.pdf_extract.classifier.rules.scale import (
    BooleanScale,
    ClampedScale,
    DiscreteScale,
    ExponentialDecayScale,
    InvertedScale,
    LinearScale,
    RangeCheckScale,
    ThresholdScale,
)


class TestBooleanScale:
    def test_below_threshold(self):
        scale = BooleanScale(threshold=0.5)
        assert scale(0.0) == 0.0
        assert scale(0.3) == 0.0
        assert scale(0.49) == 0.0

    def test_at_and_above_threshold(self):
        scale = BooleanScale(threshold=0.5)
        assert scale(0.5) == 1.0
        assert scale(0.7) == 1.0
        assert scale(1.0) == 1.0


class TestLinearScale:
    def test_below_range(self):
        scale = LinearScale(min_val=10, max_val=20)
        assert scale(5) == 0.0
        assert scale(10) == 0.0

    def test_above_range(self):
        scale = LinearScale(min_val=10, max_val=20)
        assert scale(20) == 1.0
        assert scale(25) == 1.0

    def test_within_range(self):
        scale = LinearScale(min_val=10, max_val=20)
        assert scale(15) == 0.5
        assert scale(12) == 0.2
        assert scale(18) == 0.8

    def test_custom_score_range(self):
        scale = LinearScale(min_val=0, max_val=100, min_score=0.5, max_score=1.0)
        assert scale(-10) == 0.5
        assert scale(0) == 0.5
        assert scale(50) == 0.75
        assert scale(100) == 1.0
        assert scale(150) == 1.0

    def test_dict_two_points(self):
        scale = LinearScale({0.0: 0.0, 1.0: 1.0})
        assert scale(-0.5) == 0.0  # Clamp to first point
        assert scale(0.0) == 0.0
        assert scale(0.5) == 0.5
        assert scale(1.0) == 1.0
        assert scale(1.5) == 1.0  # Clamp to last point

    def test_dict_three_points_triangular(self):
        # Triangular scale can be expressed as 3-point linear
        scale = LinearScale({10.0: 0.0, 15.0: 1.0, 20.0: 0.0})
        assert scale(5.0) == 0.0  # Clamp below
        assert scale(10.0) == 0.0  # First point
        assert scale(12.5) == 0.5  # Ascending slope
        assert scale(15.0) == 1.0  # Peak
        assert scale(17.5) == 0.5  # Descending slope
        assert scale(20.0) == 0.0  # Last point
        assert scale(25.0) == 0.0  # Clamp above

    def test_dict_four_points_piecewise(self):
        # Piecewise linear with 4 segments
        scale = LinearScale({0.0: 0.0, 10.0: 1.0, 20.0: 0.5, 30.0: 1.0})
        assert scale(-5.0) == 0.0  # Clamp below
        assert scale(0.0) == 0.0  # First point
        assert scale(5.0) == 0.5  # First segment
        assert scale(10.0) == 1.0  # Second point
        assert scale(15.0) == 0.75  # Second segment (1.0 to 0.5)
        assert scale(20.0) == 0.5  # Third point
        assert scale(25.0) == 0.75  # Third segment (0.5 to 1.0)
        assert scale(30.0) == 1.0  # Fourth point
        assert scale(35.0) == 1.0  # Clamp above

    def test_descending_scale(self):
        scale = LinearScale({0.0: 1.0, 10.0: 0.0})
        assert scale(-5.0) == 1.0
        assert scale(0.0) == 1.0
        assert scale(5.0) == 0.5
        assert scale(10.0) == 0.0
        assert scale(15.0) == 0.0

    def test_dict_unsorted_gets_sorted(self):
        # Dict with unsorted values should still work
        scale = LinearScale({20.0: 0.0, 10.0: 1.0, 15.0: 0.5})
        assert scale(10.0) == 1.0
        assert scale(15.0) == 0.5
        assert scale(20.0) == 0.0


class TestExponentialDecayScale:
    def test_zero_value(self):
        scale = ExponentialDecayScale(scale=10)
        assert scale(0) == 1.0

    def test_decay(self):
        scale = ExponentialDecayScale(scale=10)
        assert scale(10) == pytest.approx(math.exp(-1))
        assert scale(20) == pytest.approx(math.exp(-2))

    def test_different_scales(self):
        scale1 = ExponentialDecayScale(scale=5)
        scale2 = ExponentialDecayScale(scale=20)
        # scale2 decays slower
        assert scale1(10) < scale2(10)


class TestDiscreteScale:
    def test_basic_ranges(self):
        scale = DiscreteScale(
            {
                (0, 10): 0.3,
                (10, 20): 0.7,
                (20, float("inf")): 1.0,
            }
        )
        assert scale(5) == 0.3
        assert scale(15) == 0.7
        assert scale(25) == 1.0

    def test_boundary_values(self):
        scale = DiscreteScale(
            {
                (0, 10): 0.5,
                (10, 20): 0.8,
            }
        )
        assert scale(0) == 0.5
        assert scale(10) == 0.8
        assert scale(20) == 0.0  # default

    def test_custom_default(self):
        scale = DiscreteScale(
            {
                (0, 10): 0.5,
            },
            default=0.2,
        )
        assert scale(5) == 0.5
        assert scale(15) == 0.2


class TestThresholdScale:
    def test_multiple_thresholds(self):
        scale = ThresholdScale(
            [
                (0.9, 1.0),
                (0.7, 0.7),
                (0.5, 0.3),
            ],
            default=0.0,
        )
        assert scale(0.95) == 1.0
        assert scale(0.9) == 1.0
        assert scale(0.8) == 0.7
        assert scale(0.7) == 0.7
        assert scale(0.6) == 0.3
        assert scale(0.5) == 0.3
        assert scale(0.3) == 0.0

    def test_unsorted_thresholds(self):
        # Should work even if thresholds are not sorted
        scale = ThresholdScale(
            [
                (0.5, 0.3),
                (0.9, 1.0),
                (0.7, 0.7),
            ],
            default=0.0,
        )
        assert scale(0.95) == 1.0
        assert scale(0.8) == 0.7
        assert scale(0.6) == 0.3


class TestInvertedScale:
    def test_invert_boolean(self):
        scale = InvertedScale(BooleanScale(threshold=0.5))
        assert scale(0.3) == 1.0
        assert scale(0.7) == 0.0

    def test_invert_linear(self):
        scale = InvertedScale(LinearScale(min_val=0, max_val=10))
        assert scale(0) == 1.0
        assert scale(5) == 0.5
        assert scale(10) == 0.0


class TestClampedScale:
    def test_clamp_to_range(self):
        base_scale = LinearScale(min_val=0, max_val=10, min_score=-0.5, max_score=1.5)
        clamped = ClampedScale(base_scale, min_score=0.0, max_score=1.0)

        assert clamped(-5) == 0.0  # was -0.5
        assert clamped(5) == 0.5  # unchanged
        assert clamped(15) == 1.0  # was 1.5

    def test_custom_clamp_range(self):
        base_scale = LinearScale(min_val=0, max_val=10)
        clamped = ClampedScale(base_scale, min_score=0.2, max_score=0.8)

        assert clamped(-5) == 0.2
        assert clamped(5) == 0.5
        assert clamped(15) == 0.8


class TestRangeCheckScale:
    def test_within_range(self):
        scale = RangeCheckScale(min_val=10, max_val=20)
        assert scale(10) == 1.0
        assert scale(15) == 1.0
        assert scale(20) == 1.0

    def test_outside_range(self):
        scale = RangeCheckScale(min_val=10, max_val=20)
        assert scale(5) == 0.0
        assert scale(9.9) == 0.0
        assert scale(20.1) == 0.0
        assert scale(25) == 0.0

    def test_min_only(self):
        scale = RangeCheckScale(min_val=10)
        assert scale(5) == 0.0
        assert scale(10) == 1.0
        assert scale(100) == 1.0

    def test_max_only(self):
        scale = RangeCheckScale(max_val=20)
        assert scale(5) == 1.0
        assert scale(20) == 1.0
        assert scale(25) == 0.0

    def test_no_limits(self):
        scale = RangeCheckScale()
        assert scale(-100) == 1.0
        assert scale(0) == 1.0
        assert scale(100) == 1.0
