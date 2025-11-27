"""Unit tests for PieceLengthClassifier."""

import pytest

from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.classifier.piece_length_classifier import (
    PieceLengthClassifier,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import PieceLength
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Text


@pytest.fixture
def classifier() -> PieceLengthClassifier:
    return PieceLengthClassifier(config=ClassifierConfig())


class TestPieceLengthClassifier:
    """Tests for piece length classification."""

    def test_parse_piece_length_value_valid(
        self, classifier: PieceLengthClassifier
    ) -> None:
        """Test parsing valid piece length values."""
        # Test valid range
        for value in [1, 4, 16, 32]:
            text = Text(id=1, bbox=BBox(0, 0, 10, 10), text=str(value))
            result = classifier._parse_piece_length_value(text)
            assert result == value

    def test_parse_piece_length_value_invalid(
        self, classifier: PieceLengthClassifier
    ) -> None:
        """Test parsing invalid piece length values."""
        # Too small
        text = Text(id=1, bbox=BBox(0, 0, 10, 10), text="0")
        assert classifier._parse_piece_length_value(text) is None

        # Too large
        text = Text(id=1, bbox=BBox(0, 0, 10, 10), text="100")
        assert classifier._parse_piece_length_value(text) is None

        # Not a number
        text = Text(id=1, bbox=BBox(0, 0, 10, 10), text="ABC")
        assert classifier._parse_piece_length_value(text) is None

        # Empty
        text = Text(id=1, bbox=BBox(0, 0, 10, 10), text="")
        assert classifier._parse_piece_length_value(text) is None

    def test_find_smallest_containing_drawing(
        self, classifier: PieceLengthClassifier
    ) -> None:
        """Test finding the smallest drawing containing text."""
        text = Text(id=1, bbox=BBox(100, 100, 105, 105), text="4")

        # Create multiple nested drawings
        small = Drawing(id=2, bbox=BBox(98, 98, 107, 107))  # 9x9
        medium = Drawing(id=3, bbox=BBox(90, 90, 120, 120))  # 30x30
        large = Drawing(id=4, bbox=BBox(0, 0, 500, 500))  # 500x500

        drawings = [large, medium, small]  # Random order

        result = classifier._find_smallest_containing_drawing(text, drawings)
        assert result == small

    def test_find_smallest_containing_drawing_none(
        self, classifier: PieceLengthClassifier
    ) -> None:
        """Test when text is not contained in any drawing."""
        text = Text(id=1, bbox=BBox(100, 100, 105, 105), text="4")
        drawing = Drawing(id=2, bbox=BBox(200, 200, 300, 300))

        result = classifier._find_smallest_containing_drawing(text, [drawing])
        assert result is None

    def test_score_drawing_fit_perfect(self, classifier: PieceLengthClassifier) -> None:
        """Test scoring when drawing is perfectly sized around text."""
        # Text: 10x10 = 100 area
        text = Text(id=1, bbox=BBox(10, 10, 20, 20), text="4")
        # Drawing: 15x15 = 225 area, ratio = 2.25 (ideal)
        drawing = Drawing(id=2, bbox=BBox(7.5, 7.5, 22.5, 22.5))

        score = classifier._score_drawing_fit(text, drawing)
        assert score == 1.0

    def test_score_drawing_fit_slightly_small(
        self, classifier: PieceLengthClassifier
    ) -> None:
        """Test scoring when drawing is slightly smaller than ideal."""
        text = Text(id=1, bbox=BBox(10, 10, 20, 20), text="4")
        # Drawing: 12x12 = 144 area, ratio = 1.44
        drawing = Drawing(id=2, bbox=BBox(9, 9, 21, 21))

        score = classifier._score_drawing_fit(text, drawing)
        assert score == 0.8

    def test_score_drawing_fit_too_large(
        self, classifier: PieceLengthClassifier
    ) -> None:
        """Test scoring when drawing is much larger than text."""
        text = Text(id=1, bbox=BBox(10, 10, 20, 20), text="4")
        # Drawing: 30x30 = 900 area, ratio = 9.0
        drawing = Drawing(id=2, bbox=BBox(0, 0, 30, 30))

        score = classifier._score_drawing_fit(text, drawing)
        assert score == 0.6

    def test_score_drawing_fit_page_sized(
        self, classifier: PieceLengthClassifier
    ) -> None:
        """Test scoring when drawing is page-sized background."""
        text = Text(id=1, bbox=BBox(10, 10, 20, 20), text="4")
        # Drawing: 500x500 = 250000 area, ratio = 2500.0
        drawing = Drawing(id=2, bbox=BBox(0, 0, 500, 500))

        score = classifier._score_drawing_fit(text, drawing)
        assert score == 0.1

    def test_score_font_size_ideal_range(
        self, classifier: PieceLengthClassifier
    ) -> None:
        """Test font size scoring for exact match."""
        classifier.config.font_size_hints.part_count_size = 6.0
        classifier.config.font_size_hints.step_number_size = 16.0

        # Exact match to part_count_size gets 1.0
        text = Text(id=1, bbox=BBox(0, 0, 10, 10), text="4", font_size=6.0)

        score = classifier._score_piece_length_font_size(text)
        assert score == 1.0

    def test_score_font_size_too_large(self, classifier: PieceLengthClassifier) -> None:
        """Test font size scoring for too large font.

        Fonts significantly larger than part_count_size get a score of 0.0.
        """
        classifier.config.font_size_hints.part_count_size = 6.0
        classifier.config.font_size_hints.step_number_size = 16.0

        # font_size=20.0 vs expected=6.0: diff_ratio=14/6=2.33,
        # score=max(0, 1-2.33*2)=0.0
        text = Text(id=1, bbox=BBox(0, 0, 10, 10), text="4", font_size=20.0)

        score = classifier._score_piece_length_font_size(text)
        assert score == 0.0

    def test_end_to_end_classification(self, classifier: PieceLengthClassifier) -> None:
        """Test full classification pipeline with piece lengths."""
        # Create a simple page with text in circles
        text1 = Text(id=1, bbox=BBox(10, 10, 15, 15), text="4", font_size=8.0)
        circle1 = Drawing(id=2, bbox=BBox(8, 8, 17, 17))

        text2 = Text(id=3, bbox=BBox(30, 30, 35, 35), text="12", font_size=8.0)
        circle2 = Drawing(id=4, bbox=BBox(28, 28, 37, 37))

        # Text not in a circle
        text3 = Text(id=5, bbox=BBox(50, 50, 55, 55), text="5", font_size=8.0)

        page = PageData(
            page_number=1,
            blocks=[text1, circle1, text2, circle2, text3],
            bbox=BBox(0, 0, 100, 100),
        )

        result = ClassificationResult(page_data=page)
        classifier.score(result)

        # Check that text1 and text2 are classified as piece_length
        candidate1 = result.get_candidate_for_block(text1, "piece_length")
        assert candidate1 is not None
        constructed_pl1 = result.build(candidate1)
        assert isinstance(constructed_pl1, PieceLength)
        assert constructed_pl1.value == 4

        candidate2 = result.get_candidate_for_block(text2, "piece_length")
        assert candidate2 is not None
        constructed_pl2 = result.build(candidate2)
        assert isinstance(constructed_pl2, PieceLength)
        assert constructed_pl2.value == 12

        # text3 should not be classified (no circle)
        candidate3 = result.get_candidate_for_block(text3, "piece_length")
        assert candidate3 is None or candidate3.constructed is None

    def test_rejects_non_numeric_text(self, classifier: PieceLengthClassifier) -> None:
        """Test that non-numeric text is rejected."""
        text = Text(id=1, bbox=BBox(10, 10, 15, 15), text="ABC", font_size=8.0)
        drawing = Drawing(id=2, bbox=BBox(8, 8, 17, 17))

        page = PageData(
            page_number=1,
            blocks=[text, drawing],
            bbox=BBox(0, 0, 100, 100),
        )

        result = ClassificationResult(page_data=page)
        classifier.score(result)
        candidate = result.get_candidate_for_block(text, "piece_length")

        # Should either have no candidate or a candidate with no constructed element
        if candidate is not None:
            assert candidate.constructed is None

    def test_rejects_out_of_range_numbers(
        self, classifier: PieceLengthClassifier
    ) -> None:
        """Test that numbers outside 1-32 range are rejected."""
        # Too small
        text_small = Text(id=1, bbox=BBox(10, 10, 15, 15), text="0", font_size=8.0)
        drawing1 = Drawing(id=2, bbox=BBox(8, 8, 17, 17))

        # Too large
        text_large = Text(id=3, bbox=BBox(30, 30, 35, 35), text="100", font_size=8.0)
        drawing2 = Drawing(id=4, bbox=BBox(28, 28, 37, 37))

        page = PageData(
            page_number=1,
            blocks=[text_small, drawing1, text_large, drawing2],
            bbox=BBox(0, 0, 100, 100),
        )

        result = ClassificationResult(page_data=page)
        classifier.score(result)

        candidate1 = result.get_candidate_for_block(text_small, "piece_length")
        candidate2 = result.get_candidate_for_block(text_large, "piece_length")

        # Both should be rejected
        if candidate1 is not None:
            assert candidate1.constructed is None
        if candidate2 is not None:
            assert candidate2.constructed is None

    def test_prefers_smaller_drawing_over_page_background(
        self, classifier: PieceLengthClassifier
    ) -> None:
        """Test that classifier prefers small circle over page background."""
        text = Text(id=1, bbox=BBox(100, 100, 105, 105), text="8", font_size=8.0)

        # Small circle (should be selected)
        small_circle = Drawing(id=2, bbox=BBox(98, 98, 107, 107))

        # Page-sized background (should be ignored)
        page_background = Drawing(id=3, bbox=BBox(0, 0, 500, 500))

        page = PageData(
            page_number=1,
            blocks=[text, page_background, small_circle],  # Random order
            bbox=BBox(0, 0, 500, 500),
        )

        result = ClassificationResult(page_data=page)
        classifier.score(result)

        candidate = result.get_candidate_for_block(text, "piece_length")

        assert candidate is not None
        constructed_pl = result.build(candidate)
        assert isinstance(constructed_pl, PieceLength)
        assert constructed_pl.value == 8
        # Should have high score due to small, well-fitting circle
        assert candidate.score > 0.5
