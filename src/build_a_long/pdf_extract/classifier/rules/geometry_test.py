"""Tests for geometry rules."""

from __future__ import annotations

import pytest

from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.classifier.rules.base import RuleContext
from build_a_long.pdf_extract.classifier.rules.geometry import (
    AspectRatioRule,
    CornerDistanceScore,
    CoverageRule,
    EdgeProximityRule,
    InBottomBandFilter,
    IsHorizontalDividerRule,
    IsVerticalDividerRule,
    SizeRangeRule,
    SizeRatioRule,
    TextContainerFitRule,
    TopLeftPositionScore,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Text


@pytest.fixture
def context() -> RuleContext:
    """Create a dummy RuleContext with 1000x1000 page."""
    return RuleContext(
        page_data=PageData(
            page_number=1,
            bbox=BBox(0, 0, 1000, 1000),
            blocks=[],
        ),
        config=ClassifierConfig(),
    )


class TestInBottomBandFilter:
    def test_in_band(self, context: RuleContext):
        rule = InBottomBandFilter(threshold_ratio=0.1)  # Bottom 100px
        # Center at y=950 (inside 900-1000 band)
        block = Text(bbox=BBox(0, 940, 10, 960), text="foo", id=1)
        assert rule.calculate(block, context) == 1.0

    def test_out_of_band(self, context: RuleContext):
        rule = InBottomBandFilter(threshold_ratio=0.1)
        # Center at y=800 (outside 900-1000 band)
        block = Text(bbox=BBox(0, 790, 10, 810), text="foo", id=1)
        assert rule.calculate(block, context) == 0.0

    def test_inverted(self, context: RuleContext):
        rule = InBottomBandFilter(threshold_ratio=0.1, invert=True)
        # Center at y=950 (inside band, so should return 0.0)
        block = Text(bbox=BBox(0, 940, 10, 960), text="foo", id=1)
        assert rule.calculate(block, context) == 0.0


class TestCornerDistanceScore:
    def test_close_to_corner(self, context: RuleContext):
        rule = CornerDistanceScore(scale=100.0)
        # Bottom-left corner is (0, 1000)
        # Block at (0, 1000)
        block = Text(bbox=BBox(0, 1000, 0, 1000), text="foo", id=1)
        # Dist = 0, exp(0) = 1.0
        assert rule.calculate(block, context) == pytest.approx(1.0)

    def test_far_from_corner(self, context: RuleContext):
        rule = CornerDistanceScore(scale=100.0)
        # Center of page (500, 500)
        # Dist to (0, 1000) is sqrt(500^2 + 500^2) ≈ 707
        # exp(-7.07) ≈ 0.0008
        block = Text(bbox=BBox(500, 500, 500, 500), text="foo", id=1)
        assert rule.calculate(block, context) == pytest.approx(0.0008, abs=0.0001)


class TestTopLeftPositionScore:
    def test_top_left(self, context: RuleContext):
        rule = TopLeftPositionScore()
        # Top-left corner (0, 0)
        block = Text(bbox=BBox(0, 0, 10, 10), text="foo", id=1)
        # Vertical ratio = 5/1000 = 0.005 -> score = 1 - 0.005/0.4 = 0.9875
        # Horizontal ratio = 5/1000 = 0.005 -> score 1.0 (<= 0.5)
        # Combined = 0.7*0.9875 + 0.3*1.0 = 0.69125 + 0.3 = 0.99125
        assert rule.calculate(block, context) == pytest.approx(0.99125)

    def test_bottom_right(self, context: RuleContext):
        rule = TopLeftPositionScore()
        # Bottom-right corner (990, 990)
        block = Text(bbox=BBox(990, 990, 1000, 1000), text="foo", id=1)
        # Vertical ratio ~1.0 > 0.4 -> score 0.0
        assert rule.calculate(block, context) == 0.0


class TestSizeRangeRule:
    def test_in_range(self, context: RuleContext):
        rule = SizeRangeRule(min_width=10, max_width=20, min_height=10, max_height=20)
        block = Text(bbox=BBox(0, 0, 15, 15), text="foo", id=1)
        assert rule.calculate(block, context) == 1.0

    def test_too_small(self, context: RuleContext):
        rule = SizeRangeRule(min_width=10)
        block = Text(bbox=BBox(0, 0, 5, 15), text="foo", id=1)
        assert rule.calculate(block, context) == 0.0

    def test_too_large(self, context: RuleContext):
        rule = SizeRangeRule(max_width=20)
        block = Text(bbox=BBox(0, 0, 25, 15), text="foo", id=1)
        assert rule.calculate(block, context) == 0.0


class TestAspectRatioRule:
    def test_in_range(self, context: RuleContext):
        rule = AspectRatioRule(min_ratio=0.5, max_ratio=2.0)
        # Ratio 1.0
        block = Text(bbox=BBox(0, 0, 10, 10), text="foo", id=1)
        assert rule.calculate(block, context) == 1.0

    def test_out_of_range(self, context: RuleContext):
        rule = AspectRatioRule(min_ratio=0.5, max_ratio=2.0)
        # Ratio 0.1
        block = Text(bbox=BBox(0, 0, 1, 10), text="foo", id=1)
        assert rule.calculate(block, context) == 0.0


class TestCoverageRule:
    def test_high_coverage(self, context: RuleContext):
        rule = CoverageRule(min_ratio=0.8)
        # 90% coverage (900x1000 on 1000x1000 page)
        block = Drawing(bbox=BBox(0, 0, 900, 1000), id=1)
        # Norm = (0.9 - 0.8) / 0.2 = 0.5
        # Score = 0.5 + 0.5 * 0.5 = 0.75
        assert rule.calculate(block, context) == 0.75

    def test_low_coverage(self, context: RuleContext):
        rule = CoverageRule(min_ratio=0.8)
        # 10% coverage
        block = Drawing(bbox=BBox(0, 0, 100, 1000), id=1)
        assert rule.calculate(block, context) == 0.0


class TestEdgeProximityRule:
    def test_at_edges(self, context: RuleContext):
        rule = EdgeProximityRule(threshold=10.0)
        # Block exactly covering page
        block = Drawing(bbox=BBox(0, 0, 1000, 1000), id=1)
        assert rule.calculate(block, context) == 1.0

    def test_far_from_edges(self, context: RuleContext):
        rule = EdgeProximityRule(threshold=10.0)
        # Block in center (100, 100, 900, 900)
        # Distances: L=100, R=100, T=100, B=100. Avg=100.
        # Score = max(0, 1 - (100-10)/50) = max(0, 1 - 1.8) = 0.0
        block = Drawing(bbox=BBox(100, 100, 900, 900), id=1)
        assert rule.calculate(block, context) == 0.0


class TestDividerRules:
    def test_vertical_divider(self, context: RuleContext):
        rule = IsVerticalDividerRule(
            max_thickness=5, min_length_ratio=0.5, edge_margin=10
        )
        # Thin vertical line in center, 60% height
        block = Drawing(bbox=BBox(500, 200, 502, 800), id=1)
        assert rule.calculate(block, context) == 1.0

    def test_vertical_divider_too_thick(self, context: RuleContext):
        rule = IsVerticalDividerRule(
            max_thickness=5, min_length_ratio=0.5, edge_margin=10
        )
        block = Drawing(bbox=BBox(500, 200, 510, 800), id=1)
        assert rule.calculate(block, context) == 0.0

    def test_horizontal_divider(self, context: RuleContext):
        rule = IsHorizontalDividerRule(
            max_thickness=5, min_length_ratio=0.5, edge_margin=10
        )
        # Thin horizontal line in center, 60% width
        block = Drawing(bbox=BBox(200, 500, 800, 502), id=1)
        assert rule.calculate(block, context) == 1.0


class TestTextContainerFitRule:
    def test_perfect_fit(self, context: RuleContext):
        rule = TextContainerFitRule()
        text = Text(bbox=BBox(10, 10, 20, 20), text="1", id=1)  # Area 100
        # Drawing 20x20 = 400 area. Ratio 4.0.
        # Logic: 2.0 <= ratio <= 4.0 -> score 1.0
        drawing = Drawing(bbox=BBox(5, 5, 25, 25), id=2)
        context.page_data.blocks = [text, drawing]

        assert rule.calculate(text, context) == 1.0

    def test_no_container(self, context: RuleContext):
        rule = TextContainerFitRule()
        text = Text(bbox=BBox(10, 10, 20, 20), text="1", id=1)
        # Drawing disjoint
        drawing = Drawing(bbox=BBox(100, 100, 120, 120), id=2)
        context.page_data.blocks = [text, drawing]

        assert rule.calculate(text, context) == 0.0


class TestSizeRatioRule:
    def test_ideal_ratio(self, context: RuleContext):
        rule = SizeRatioRule(ideal_ratio=0.1, min_ratio=0.05, max_ratio=0.2)
        # 100x100 on 1000x1000 page -> ratio 0.1
        block = Drawing(bbox=BBox(0, 0, 100, 100), id=1)
        assert rule.calculate(block, context) == 1.0

    def test_boundary_ratio(self, context: RuleContext):
        rule = SizeRatioRule(ideal_ratio=0.1, min_ratio=0.05, max_ratio=0.2)
        # 50x50 -> ratio 0.05. Should be 0.0 per triangular logic?
        # Logic: (ratio - min) / (ideal - min) -> (0.05-0.05)/(0.1-0.05) = 0.0
        block = Drawing(bbox=BBox(0, 0, 50, 50), id=1)
        assert rule.calculate(block, context) == 0.0
