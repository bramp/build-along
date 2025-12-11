"""Tests for ProgressBarIndicatorClassifier."""

import pytest

from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.classifier.pages.progress_bar_indicator_classifier import (
    ProgressBarIndicatorClassifier,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    ProgressBarIndicator,
)
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image, Text


@pytest.fixture
def classifier() -> ProgressBarIndicatorClassifier:
    return ProgressBarIndicatorClassifier(config=ClassifierConfig())


def make_page_data(blocks: list, page_height: float = 1000.0) -> PageData:
    """Create a PageData with given blocks."""
    return PageData(
        page_number=1,
        bbox=BBox(x0=0, y0=0, x1=800, y1=page_height),
        blocks=blocks,
    )


class TestProgressBarIndicatorClassifier:
    """Tests for progress bar indicator classification."""

    def test_finds_valid_indicator(self, classifier: ProgressBarIndicatorClassifier):
        """Test that a valid indicator is found."""
        # 12x12 square at bottom of page
        page_height = 1000.0
        indicator = Drawing(
            id=1, bbox=BBox(100, page_height - 20, 112, page_height - 8)
        )

        page = make_page_data([indicator], page_height=page_height)
        result = ClassificationResult(page_data=page)
        classifier._score(result)

        candidate = result.get_candidate_for_block(indicator, "progress_bar_indicator")
        assert candidate is not None
        assert candidate.score > 0.5

        built = classifier.build(candidate, result)
        assert isinstance(built, ProgressBarIndicator)
        assert built.bbox == indicator.bbox

    def test_finds_image_indicator(self, classifier: ProgressBarIndicatorClassifier):
        """Test that an image indicator is found."""
        page_height = 1000.0
        indicator = Image(id=1, bbox=BBox(100, page_height - 20, 112, page_height - 8))

        page = make_page_data([indicator], page_height=page_height)
        result = ClassificationResult(page_data=page)
        classifier._score(result)

        candidate = result.get_candidate_for_block(indicator, "progress_bar_indicator")
        assert candidate is not None

    def test_rejects_non_square(self, classifier: ProgressBarIndicatorClassifier):
        """Test that non-square elements are rejected."""
        page_height = 1000.0
        # 12x24 rectangle (aspect ratio 2.0)
        indicator = Drawing(
            id=1, bbox=BBox(100, page_height - 30, 112, page_height - 6)
        )

        page = make_page_data([indicator], page_height=page_height)
        result = ClassificationResult(page_data=page)
        classifier._score(result)

        candidate = result.get_candidate_for_block(indicator, "progress_bar_indicator")
        # Should be rejected or have very low score
        if candidate:
            assert candidate.score < 0.1

    def test_rejects_wrong_size(self, classifier: ProgressBarIndicatorClassifier):
        """Test that wrong sized elements are rejected."""
        page_height = 1000.0
        # 50x50 square (too large)
        indicator = Drawing(
            id=1, bbox=BBox(100, page_height - 60, 150, page_height - 10)
        )

        page = make_page_data([indicator], page_height=page_height)
        result = ClassificationResult(page_data=page)
        classifier._score(result)

        candidate = result.get_candidate_for_block(indicator, "progress_bar_indicator")
        if candidate:
            # SizeRangeRule handles hard cutoffs now, returning 0.0
            assert candidate.score == 0.0

    def test_rejects_wrong_position(self, classifier: ProgressBarIndicatorClassifier):
        """Test that elements not at bottom are rejected."""
        page_height = 1000.0
        # 12x12 square at top of page
        indicator = Drawing(id=1, bbox=BBox(100, 100, 112, 112))

        page = make_page_data([indicator], page_height=page_height)
        result = ClassificationResult(page_data=page)
        classifier._score(result)

        candidate = result.get_candidate_for_block(indicator, "progress_bar_indicator")
        if candidate:
            assert candidate.score == 0.0

    def test_rejects_text(self, classifier: ProgressBarIndicatorClassifier):
        """Test that text elements are rejected."""
        page_height = 1000.0
        text = Text(
            id=1,
            bbox=BBox(100, page_height - 20, 112, page_height - 8),
            text="12",
            font_size=12.0,
        )

        page = make_page_data([text], page_height=page_height)
        result = ClassificationResult(page_data=page)
        classifier._score(result)

        candidate = result.get_candidate_for_block(text, "progress_bar_indicator")
        assert candidate is None
