"""Tests for visual rules."""

from __future__ import annotations

import pytest

from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.classifier.rules.base import RuleContext
from build_a_long.pdf_extract.classifier.rules.visual import StrokeColorScore
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Drawing


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


class TestStrokeColorScore:
    def test_white_stroke(self, context: RuleContext):
        rule = StrokeColorScore()
        block = Drawing(bbox=BBox(0, 0, 10, 10), stroke_color=(1.0, 1.0, 1.0), id=1)
        assert rule.calculate(block, context) == 1.0

    def test_light_gray_stroke(self, context: RuleContext):
        rule = StrokeColorScore()
        block = Drawing(bbox=BBox(0, 0, 10, 10), stroke_color=(0.8, 0.8, 0.8), id=1)
        assert rule.calculate(block, context) == 0.7

    def test_dark_stroke(self, context: RuleContext):
        rule = StrokeColorScore()
        block = Drawing(bbox=BBox(0, 0, 10, 10), stroke_color=(0.0, 0.0, 0.0), id=1)
        assert rule.calculate(block, context) == 0.3

    def test_white_fill_fallback(self, context: RuleContext):
        rule = StrokeColorScore()
        # No stroke, but white fill
        block = Drawing(
            bbox=BBox(0, 0, 10, 10),
            fill_color=(1.0, 1.0, 1.0),
            stroke_color=None,
            id=1,
        )
        assert rule.calculate(block, context) == 0.8

    def test_no_color(self, context: RuleContext):
        rule = StrokeColorScore()
        block = Drawing(
            bbox=BBox(0, 0, 10, 10), fill_color=None, stroke_color=None, id=1
        )
        assert rule.calculate(block, context) == 0.0
