"""Unit tests for PieceLengthClassifier."""

import pytest

from build_a_long.pdf_extract.classifier import (
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.classifier.parts.piece_length_classifier import (
    PieceLengthClassifier,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import PieceLength
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Text


@pytest.fixture
def classifier() -> PieceLengthClassifier:
    return PieceLengthClassifier(config=ClassifierConfig())


def make_page_data(blocks: list) -> PageData:
    """Create a PageData with given blocks."""
    return PageData(
        page_number=1,
        bbox=BBox(x0=0, y0=0, x1=600, y1=500),
        blocks=blocks,
    )


class TestPieceLengthClassifier:
    """Tests for piece length classification."""

    def test_end_to_end_classification(self, classifier: PieceLengthClassifier) -> None:
        """Test full classification pipeline with piece lengths."""
        # Create a simple page with text in circles
        text1 = Text(id=1, bbox=BBox(10, 10, 15, 15), text="4", font_size=8.0)
        # Small circle around text1
        circle1 = Drawing(id=2, bbox=BBox(8, 8, 17, 17))

        text2 = Text(id=3, bbox=BBox(30, 30, 35, 35), text="12", font_size=8.0)
        # Small circle around text2
        circle2 = Drawing(id=4, bbox=BBox(28, 28, 37, 37))

        # Text not in a circle
        text3 = Text(id=5, bbox=BBox(50, 50, 55, 55), text="5", font_size=8.0)

        # Text with invalid value
        text4 = Text(id=6, bbox=BBox(70, 70, 75, 75), text="ABC", font_size=8.0)
        circle4 = Drawing(id=7, bbox=BBox(68, 68, 77, 77))

        page = make_page_data([text1, circle1, text2, circle2, text3, text4, circle4])

        result = ClassificationResult(page_data=page)
        classifier._score(result)

        # Check text1 -> PieceLength(4)
        candidate1 = result.get_candidate_for_block(text1, "piece_length")
        assert candidate1 is not None
        assert candidate1.score > 0.5
        pl1 = classifier.build(candidate1, result)
        assert isinstance(pl1, PieceLength)
        assert pl1.value == 4

        # Check text2 -> PieceLength(12)
        candidate2 = result.get_candidate_for_block(text2, "piece_length")
        assert candidate2 is not None
        assert candidate2.score > 0.5
        pl2 = classifier.build(candidate2, result)
        assert isinstance(pl2, PieceLength)
        assert pl2.value == 12

        # Check text3 -> No candidate (no containing drawing)
        candidate3 = result.get_candidate_for_block(text3, "piece_length")
        # TextContainerFitRule will return 0.0, so total score will be low/zero
        assert candidate3 is None or candidate3.score < 0.1

        # Check text4 -> No candidate (invalid value)
        candidate4 = result.get_candidate_for_block(text4, "piece_length")
        # PieceLengthValueRule will return 0.0
        assert candidate4 is None or candidate4.score < 0.1

    def test_prefers_smaller_drawing_over_page_background(
        self, classifier: PieceLengthClassifier
    ) -> None:
        """Test that classifier prefers small circle over page background."""
        text = Text(id=1, bbox=BBox(100, 100, 105, 105), text="8", font_size=8.0)

        # Small circle (should be selected as container fit)
        small_circle = Drawing(id=2, bbox=BBox(98, 98, 107, 107))

        # Page-sized background (too large ratio)
        page_background = Drawing(id=3, bbox=BBox(0, 0, 500, 500))

        page = make_page_data([text, page_background, small_circle])

        result = ClassificationResult(page_data=page)
        classifier._score(result)

        candidate = result.get_candidate_for_block(text, "piece_length")
        assert candidate is not None
        # Score should be high because TextContainerFitRule finds the small circle
        assert candidate.score > 0.8

        pl = classifier.build(candidate, result)
        assert isinstance(pl, PieceLength)
        assert pl.value == 8
        # The bbox should include the small circle
        assert pl.bbox.contains(small_circle.bbox)

    def test_font_size_scoring(self, classifier: PieceLengthClassifier) -> None:
        """Test that font size affects scoring."""
        # Setup hints: part_count_size=10.0
        classifier.config.font_size_hints.part_count_size = 10.0

        # Good font size (10.0)
        text_good = Text(id=1, bbox=BBox(10, 10, 15, 15), text="4", font_size=10.0)
        circle_good = Drawing(id=2, bbox=BBox(8, 8, 17, 17))

        # Bad font size (too large, e.g. 50.0)
        text_bad = Text(id=3, bbox=BBox(50, 50, 80, 80), text="4", font_size=50.0)
        circle_bad = Drawing(id=4, bbox=BBox(40, 40, 90, 90))

        page = make_page_data([text_good, circle_good, text_bad, circle_bad])
        result = ClassificationResult(page_data=page)
        classifier._score(result)

        # Good candidate should have high score
        cand_good = result.get_candidate_for_block(text_good, "piece_length")
        assert cand_good is not None
        assert cand_good.score > 0.9

        # Bad candidate should have lower score (FontSizeMatch fails)
        cand_bad = result.get_candidate_for_block(text_bad, "piece_length")
        assert cand_bad is not None
        # Score should be significantly lower than good candidate
        # 1.0 (Value) + 1.0 (Fit) + 0.0 (Font) / 3.0 approx 0.67
        assert cand_bad.score < cand_good.score
