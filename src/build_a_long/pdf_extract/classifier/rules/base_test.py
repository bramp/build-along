"""Tests for base rules."""

from __future__ import annotations

import pytest

from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.classifier.rules.base import (
    IsInstanceFilter,
    MaxScoreRule,
    Rule,
    RuleContext,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Block, Drawing, Image, Text


@pytest.fixture
def context() -> RuleContext:
    """Create a dummy RuleContext."""
    return RuleContext(
        page_data=PageData(
            page_number=1,
            bbox=BBox(0, 0, 1000, 1000),
            blocks=[],
        ),
        config=ClassifierConfig(),
    )


class AlwaysOneRule(Rule):
    """Rule that always returns 1.0."""

    def __init__(self, name: str = "AlwaysOne"):
        self.name = name

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        return 1.0


class AlwaysZeroRule(Rule):
    """Rule that always returns 0.0."""

    def __init__(self, name: str = "AlwaysZero"):
        self.name = name

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        return 0.0


class AlwaysNoneRule(Rule):
    """Rule that always returns None (skipped)."""

    def __init__(self, name: str = "AlwaysNone"):
        self.name = name

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        return None


class TestIsInstanceFilter:
    """Tests for IsInstanceFilter."""

    def test_single_type_match(self, context: RuleContext):
        rule = IsInstanceFilter(Text)
        text_block = Text(bbox=BBox(0, 0, 10, 10), text="foo", id=1)
        assert rule.calculate(text_block, context) == 1.0

    def test_single_type_mismatch(self, context: RuleContext):
        rule = IsInstanceFilter(Text)
        drawing_block = Drawing(bbox=BBox(0, 0, 10, 10), id=1)
        assert rule.calculate(drawing_block, context) == 0.0

    def test_tuple_types_match(self, context: RuleContext):
        rule = IsInstanceFilter((Text, Drawing))
        text_block = Text(bbox=BBox(0, 0, 10, 10), text="foo", id=1)
        drawing_block = Drawing(bbox=BBox(0, 0, 10, 10), id=2)
        assert rule.calculate(text_block, context) == 1.0
        assert rule.calculate(drawing_block, context) == 1.0

    def test_tuple_types_mismatch(self, context: RuleContext):
        rule = IsInstanceFilter((Text, Drawing))
        image_block = Image(bbox=BBox(0, 0, 10, 10), id=1)
        assert rule.calculate(image_block, context) == 0.0


class TestMaxScoreRule:
    """Tests for MaxScoreRule."""

    def test_returns_max_score(self, context: RuleContext):
        rule = MaxScoreRule(
            rules=[
                AlwaysOneRule(),
                AlwaysZeroRule(),
            ]
        )
        block = Text(bbox=BBox(0, 0, 10, 10), text="foo", id=1)
        assert rule.calculate(block, context) == 1.0

    def test_ignores_none_scores(self, context: RuleContext):
        rule = MaxScoreRule(
            rules=[
                AlwaysNoneRule(),
                AlwaysZeroRule(),
            ]
        )
        block = Text(bbox=BBox(0, 0, 10, 10), text="foo", id=1)
        assert rule.calculate(block, context) == 0.0

    def test_returns_none_if_all_none(self, context: RuleContext):
        rule = MaxScoreRule(
            rules=[
                AlwaysNoneRule(),
                AlwaysNoneRule(),
            ]
        )
        block = Text(bbox=BBox(0, 0, 10, 10), text="foo", id=1)
        assert rule.calculate(block, context) is None

    def test_empty_rules_returns_none(self, context: RuleContext):
        rule = MaxScoreRule(rules=[])
        block = Text(bbox=BBox(0, 0, 10, 10), text="foo", id=1)
        assert rule.calculate(block, context) is None
