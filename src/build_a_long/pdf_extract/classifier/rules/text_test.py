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
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("123", 1.0),
            ("abc", 0.0),
            ("12a", 0.0),
            ("", 0.0),
        ],
    )
    def test_regex_match(
        self, context: RuleContext, text: str, expected: float
    ) -> None:
        rule = RegexMatch(r"^\d+$")
        block = Text(bbox=BBox(0, 0, 10, 10), text=text, id=1)
        assert rule.calculate(block, context) == expected


class TestFontSizeMatch:
    @pytest.mark.parametrize(
        "font_size,target_size,expected",
        [
            (10.0, 10.0, 1.0),  # Exact match
            (11.0, 10.0, 0.8),  # 10% diff -> score = 1 - 0.1*2 = 0.8
            (10.0, None, None),  # No hint
        ],
    )
    def test_font_size_match(
        self,
        context: RuleContext,
        font_size: float,
        target_size: float | None,
        expected: float | None,
    ) -> None:
        rule = FontSizeMatch(target_size=target_size)
        block = Text(bbox=BBox(0, 0, 10, 10), text="foo", font_size=font_size, id=1)
        if expected is None:
            assert rule.calculate(block, context) is None
        else:
            assert rule.calculate(block, context) == pytest.approx(expected)


class TestFontSizeRangeRule:
    @pytest.mark.parametrize(
        "font_size,expected",
        [
            (15.0, 1.0),  # In range
            (10.5, 0.7),  # Near boundary (between 9.0 and 11.0)
            (5.0, 0.0),  # Out of range
        ],
    )
    def test_font_size_range(
        self, context: RuleContext, font_size: float, expected: float
    ) -> None:
        rule = FontSizeRangeRule(min_size=10.0, max_size=20.0, tolerance=1.0)
        block = Text(bbox=BBox(0, 0, 10, 10), text="foo", font_size=font_size, id=1)
        assert rule.calculate(block, context) == pytest.approx(expected)


class TestPageNumberValueMatch:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("5", 1.0),  # Exact match (page number is 5)
            ("7", 0.8),  # Mismatch (5 vs 7, diff=2, score=1-0.2=0.8)
            ("105", 0.0),  # Large mismatch
        ],
    )
    def test_page_number_value(
        self, context: RuleContext, text: str, expected: float
    ) -> None:
        rule = PageNumberValueMatch()
        block = Text(bbox=BBox(0, 0, 10, 10), text=text, id=1)
        assert rule.calculate(block, context) == pytest.approx(expected)


class TestTextPatternRules:
    @pytest.mark.parametrize(
        "rule_cls,text,expected_score",
        [
            # Page Number
            (PageNumberTextRule, "5", 1.0),
            (PageNumberTextRule, "05", 0.9),
            (PageNumberTextRule, "foo", 0.0),
            # Part Count
            (PartCountTextRule, "2x", 1.0),
            (PartCountTextRule, "2", 0.0),
            # Bag Number
            (BagNumberTextRule, "1", 1.0),
            (BagNumberTextRule, "foo", 0.0),
            # Step Number
            (StepNumberTextRule, "1", 1.0),
            (StepNumberTextRule, "foo", 0.0),
            # Piece Length
            (PieceLengthValueRule, "4", 1.0),
            (PieceLengthValueRule, "33", 0.0),
            (PieceLengthValueRule, "0", 0.0),
        ],
    )
    def test_text_patterns(
        self,
        context: RuleContext,
        rule_cls: type,
        text: str,
        expected_score: float,
    ) -> None:
        rule = rule_cls()
        block = Text(bbox=BBox(0, 0, 0, 0), text=text, id=1)
        assert rule.calculate(block, context) == pytest.approx(expected_score)

    @pytest.mark.parametrize(
        "text,min_score,max_score",
        [
            ("1234567", 0.9, 1.0),  # Preferred (7 digits)
            ("1234", 0.0, 0.6),  # Low score (4 digits)
        ],
    )
    def test_part_number_text(
        self,
        context: RuleContext,
        text: str,
        min_score: float,
        max_score: float,
    ) -> None:
        rule = PartNumberTextRule()
        score = rule.calculate(Text(bbox=BBox(0, 0, 0, 0), text=text, id=1), context)
        assert score is not None
        assert min_score <= score <= max_score


class TestBagNumberFontSizeRule:
    @pytest.mark.parametrize(
        "font_size,expected",
        [
            (40.0, 1.0),  # Large font
            (20.0, 0.0),  # Small font
        ],
    )
    def test_bag_number_font_size(
        self, context: RuleContext, font_size: float, expected: float
    ) -> None:
        rule = BagNumberFontSizeRule()
        block = Text(bbox=BBox(0, 0, 0, 0), text="1", font_size=font_size, id=1)
        assert rule.calculate(block, context) == expected
