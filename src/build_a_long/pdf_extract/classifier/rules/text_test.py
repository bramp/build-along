"""Tests for text rules."""

from __future__ import annotations

import pytest

from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.classifier.rules.base import RuleContext
from build_a_long.pdf_extract.classifier.rules.text import (
    BagNumberFontSizeRule,
    BagNumberTextRule,
    FontSizeMatch,
    FontSizeRangeRule,
    PageNumberTextRule,
    PageNumberValueMatch,
    PartCountTextRule,
    PartNumberTextRule,
    PieceLengthValueRule,
    RegexMatch,
    StepNumberTextRule,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Text


@pytest.fixture
def context() -> RuleContext:
    """Create a dummy RuleContext."""
    return RuleContext(
        page_data=PageData(
            page_number=5,
            bbox=BBox(0, 0, 1000, 1000),
            blocks=[],
        ),
        config=ClassifierConfig(),
    )


class TestRegexMatch:
    def test_match(self, context: RuleContext):
        rule = RegexMatch(r"^\d+$")
        block = Text(bbox=BBox(0, 0, 10, 10), text="123", id=1)
        assert rule.calculate(block, context) == 1.0

    def test_no_match(self, context: RuleContext):
        rule = RegexMatch(r"^\d+$")
        block = Text(bbox=BBox(0, 0, 10, 10), text="abc", id=1)
        assert rule.calculate(block, context) == 0.0


class TestFontSizeMatch:
    def test_exact_match(self, context: RuleContext):
        rule = FontSizeMatch(target_size=10.0)
        block = Text(bbox=BBox(0, 0, 10, 10), text="foo", font_size=10.0, id=1)
        assert rule.calculate(block, context) == 1.0

    def test_close_match(self, context: RuleContext):
        rule = FontSizeMatch(target_size=10.0)
        # 10% diff -> score = 1 - 0.1*2 = 0.8
        block = Text(bbox=BBox(0, 0, 10, 10), text="foo", font_size=11.0, id=1)
        assert rule.calculate(block, context) == pytest.approx(0.8)

    def test_no_hint(self, context: RuleContext):
        rule = FontSizeMatch(target_size=None)
        block = Text(bbox=BBox(0, 0, 10, 10), text="foo", font_size=10.0, id=1)
        assert rule.calculate(block, context) is None


class TestFontSizeRangeRule:
    def test_in_range(self, context: RuleContext):
        rule = FontSizeRangeRule(min_size=10.0, max_size=20.0, tolerance=1.0)
        block = Text(bbox=BBox(0, 0, 10, 10), text="foo", font_size=15.0, id=1)
        assert rule.calculate(block, context) == 1.0

    def test_near_boundary(self, context: RuleContext):
        rule = FontSizeRangeRule(min_size=10.0, max_size=20.0, tolerance=1.0)
        # Between 9.0 and 11.0
        block = Text(bbox=BBox(0, 0, 10, 10), text="foo", font_size=10.5, id=1)
        assert rule.calculate(block, context) == 0.7

    def test_out_of_range(self, context: RuleContext):
        rule = FontSizeRangeRule(min_size=10.0, max_size=20.0, tolerance=1.0)
        block = Text(bbox=BBox(0, 0, 10, 10), text="foo", font_size=5.0, id=1)
        assert rule.calculate(block, context) == 0.0


class TestPageNumberValueMatch:
    def test_match(self, context: RuleContext):
        rule = PageNumberValueMatch()
        # Page number is 5 in context
        block = Text(bbox=BBox(0, 0, 10, 10), text="5", id=1)
        assert rule.calculate(block, context) == 1.0

    def test_mismatch(self, context: RuleContext):
        rule = PageNumberValueMatch()
        # Page number is 5, text is 7. Diff=2. Score = 1 - 0.2 = 0.8
        block = Text(bbox=BBox(0, 0, 10, 10), text="7", id=1)
        assert rule.calculate(block, context) == 0.8


class TestTextPatternRules:
    def test_page_number_text(self, context: RuleContext):
        rule = PageNumberTextRule()
        assert (
            rule.calculate(Text(bbox=BBox(0, 0, 0, 0), text="5", id=1), context) == 1.0
        )
        assert (
            rule.calculate(Text(bbox=BBox(0, 0, 0, 0), text="05", id=1), context) == 0.9
        )
        assert (
            rule.calculate(Text(bbox=BBox(0, 0, 0, 0), text="foo", id=1), context)
            == 0.0
        )

    def test_part_count_text(self, context: RuleContext):
        rule = PartCountTextRule()
        assert (
            rule.calculate(Text(bbox=BBox(0, 0, 0, 0), text="2x", id=1), context) == 1.0
        )
        assert (
            rule.calculate(Text(bbox=BBox(0, 0, 0, 0), text="2", id=1), context) == 0.0
        )

    def test_part_number_text(self, context: RuleContext):
        rule = PartNumberTextRule()
        # 7 digits is preferred
        score = rule.calculate(
            Text(bbox=BBox(0, 0, 0, 0), text="1234567", id=1), context
        )
        assert score is not None
        assert score > 0.9
        # 4 digits is low score
        score = rule.calculate(Text(bbox=BBox(0, 0, 0, 0), text="1234", id=1), context)
        assert score is not None
        assert score < 0.6

    def test_bag_number_text(self, context: RuleContext):
        rule = BagNumberTextRule()
        assert (
            rule.calculate(Text(bbox=BBox(0, 0, 0, 0), text="1", id=1), context) == 1.0
        )
        assert (
            rule.calculate(Text(bbox=BBox(0, 0, 0, 0), text="foo", id=1), context)
            == 0.0
        )

    def test_step_number_text(self, context: RuleContext):
        rule = StepNumberTextRule()
        assert (
            rule.calculate(Text(bbox=BBox(0, 0, 0, 0), text="1", id=1), context) == 1.0
        )
        assert (
            rule.calculate(Text(bbox=BBox(0, 0, 0, 0), text="foo", id=1), context)
            == 0.0
        )

    def test_piece_length_value(self, context: RuleContext):
        rule = PieceLengthValueRule()
        assert (
            rule.calculate(Text(bbox=BBox(0, 0, 0, 0), text="4", id=1), context) == 1.0
        )
        assert (
            rule.calculate(Text(bbox=BBox(0, 0, 0, 0), text="33", id=1), context) == 0.0
        )
        assert (
            rule.calculate(Text(bbox=BBox(0, 0, 0, 0), text="0", id=1), context) == 0.0
        )


class TestBagNumberFontSizeRule:
    def test_large_font(self, context: RuleContext):
        rule = BagNumberFontSizeRule()
        block = Text(bbox=BBox(0, 0, 0, 0), text="1", font_size=40.0, id=1)
        assert rule.calculate(block, context) == 1.0

    def test_small_font(self, context: RuleContext):
        rule = BagNumberFontSizeRule()
        block = Text(bbox=BBox(0, 0, 0, 0), text="1", font_size=20.0, id=1)
        assert rule.calculate(block, context) == 0.0
