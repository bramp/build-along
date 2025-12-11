"""Tests for scoring utilities."""

import pytest

from build_a_long.pdf_extract.classifier.rules.scoring import (
    score_exponential_decay,
    score_linear,
    score_triangular,
)


def test_score_triangular():
    # Peak at 10, range 0-20
    assert score_triangular(10, 0, 10, 20) == 1.0
    assert score_triangular(0, 0, 10, 20) == 0.0
    assert score_triangular(20, 0, 10, 20) == 0.0
    assert score_triangular(5, 0, 10, 20) == 0.5
    assert score_triangular(15, 0, 10, 20) == 0.5

    # Out of bounds
    assert score_triangular(-1, 0, 10, 20) == 0.0
    assert score_triangular(21, 0, 10, 20) == 0.0

    # Edge case: min == ideal
    assert score_triangular(10, 10, 10, 20) == 1.0

    # Edge case: max == ideal
    assert score_triangular(10, 0, 10, 10) == 1.0


def test_score_linear():
    # Range 0-10, score 0.0-1.0
    assert score_linear(0, 0, 10) == 0.0
    assert score_linear(10, 0, 10) == 1.0
    assert score_linear(5, 0, 10) == 0.5

    # Clamping
    assert score_linear(-5, 0, 10) == 0.0
    assert score_linear(15, 0, 10) == 1.0

    # Custom score range
    # Range 0-10, score 0.5-1.0
    assert score_linear(0, 0, 10, min_score=0.5) == 0.5
    assert score_linear(10, 0, 10, min_score=0.5) == 1.0
    assert score_linear(5, 0, 10, min_score=0.5) == 0.75


def test_score_exponential_decay():
    # exp(-0/10) = 1.0
    assert score_exponential_decay(0, 10) == 1.0
    # exp(-10/10) = 0.367
    assert score_exponential_decay(10, 10) == pytest.approx(0.3678, abs=0.001)
