"""Tests for the page number classifier."""

import pytest

from build_a_long.pdf_extract.classifier import (
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.classifier.pages.page_number_classifier import (
    PageNumberClassifier,
)
from build_a_long.pdf_extract.classifier.text import FontSizeHints
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import PageNumber
from build_a_long.pdf_extract.extractor.page_blocks import (
    Text,
)


@pytest.fixture
def classifier() -> PageNumberClassifier:
    return PageNumberClassifier(config=ClassifierConfig())


class TestPageNumberClassification:
    """Tests for page number detection and scoring."""

    def test_no_page_numbers_on_empty_page(
        self, classifier: PageNumberClassifier
    ) -> None:
        """Test that classification succeeds on a page with no elements."""
        page_data = PageData(
            page_number=1,
            blocks=[],
            bbox=BBox(0, 0, 100, 200),
        )
        result = ClassificationResult(page_data=page_data)
        classifier.score(result)

        # Should not create any candidates
        candidates = result.get_candidates("page_number")
        assert not candidates

    def test_single_page_number_bottom_left(
        self, classifier: PageNumberClassifier
    ) -> None:
        """Test identifying a page number in the bottom-left corner."""
        page_bbox = BBox(0, 0, 100, 200)
        page_number_text = Text(
            id=0,
            bbox=BBox(5, 190, 15, 198),  # Bottom-left position
            text="1",
        )

        page_data = PageData(
            page_number=1,
            blocks=[page_number_text],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)
        classifier.score(result)

        # Check the page_number candidate exists and has a good score
        candidate = result.get_candidate_for_block(page_number_text, "page_number")
        assert candidate is not None
        page_number = result.build(candidate)
        assert isinstance(page_number, PageNumber)
        assert page_number.value == 1
        assert page_number.bbox == page_number_text.bbox
        assert candidate.score > 0.5

    def test_single_page_number_bottom_right(
        self, classifier: PageNumberClassifier
    ) -> None:
        """Test identifying a page number in the bottom-right corner."""
        page_bbox = BBox(0, 0, 100, 200)
        page_number_text = Text(
            id=0,
            bbox=BBox(90, 190, 98, 198),  # Bottom-right position
            text="5",
        )

        page_data = PageData(
            page_number=1,
            blocks=[page_number_text],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)
        classifier.score(result)

        # Check the page_number candidate exists and has a good score
        candidate = result.get_candidate_for_block(page_number_text, "page_number")
        assert candidate is not None
        page_number = result.build(candidate)
        assert isinstance(page_number, PageNumber)
        assert page_number.value == 5
        assert page_number.bbox == page_number_text.bbox
        assert candidate.score > 0.5

    def test_multiple_candidates_prefer_corners(
        self, classifier: PageNumberClassifier
    ) -> None:
        """Test that corner elements score higher than center ones."""
        page_bbox = BBox(0, 0, 100, 200)

        # Element in center-bottom (less preferred)
        center_text = Text(
            id=0,
            bbox=BBox(45, 190, 55, 198),
            text="2",
        )

        # Element in corner (more preferred)
        corner_text = Text(
            id=1,
            bbox=BBox(5, 190, 15, 198),
            text="3",
        )

        page_data = PageData(
            page_number=1,
            blocks=[center_text, corner_text],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)
        classifier.score(result)

        # Check scores from ClassificationResult
        corner_candidate = result.get_candidate_for_block(corner_text, "page_number")
        center_candidate = result.get_candidate_for_block(center_text, "page_number")
        assert corner_candidate is not None
        assert center_candidate is not None
        assert corner_candidate.score > center_candidate.score

    def test_prefer_numeric_match_to_page_index(
        self, classifier: PageNumberClassifier
    ) -> None:
        """Test that page numbers matching PageData.page_number score higher."""
        page_bbox = BBox(0, 0, 100, 200)
        # Two numbers, both near bottom, but only one matches the page number 7
        txt6 = Text(id=0, bbox=BBox(10, 190, 14, 196), text="6")
        txt7 = Text(id=1, bbox=BBox(90, 190, 94, 196), text="7")

        page_data = PageData(
            page_number=7,
            blocks=[txt6, txt7],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)
        classifier.score(result)

        # Check that txt7 scored higher than txt6
        candidate7 = result.get_candidate_for_block(txt7, "page_number")
        candidate6 = result.get_candidate_for_block(txt6, "page_number")
        assert candidate7 is not None
        assert candidate6 is not None
        assert candidate7.score > candidate6.score

    def test_non_numeric_text_scores_low(
        self, classifier: PageNumberClassifier
    ) -> None:
        """Test that non-numeric text scores low."""
        page_bbox = BBox(0, 0, 100, 200)
        text_block = Text(
            id=0,
            bbox=BBox(5, 190, 50, 198),  # Bottom-left position
            text="Hello World",
        )

        page_data = PageData(
            page_number=1,
            blocks=[text_block],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)
        classifier.score(result)

        # Should not be labeled due to text pattern (position is good but text is bad)
        # Low-scoring candidates (< 0.5) are not created to reduce debug spam
        candidate = result.get_candidate_for_block(text_block, "page_number")
        assert candidate is None  # No candidate created due to low score

    def test_two_page_numbers_only_one_constructed(
        self, classifier: PageNumberClassifier
    ) -> None:
        """Test that when two valid page numbers exist, both candidates are created
        but construction logic (usually in PageClassifier) would pick one.
        Here we verify both are valid candidates.
        """
        page_bbox = BBox(0, 0, 100, 200)

        # Two page numbers in bottom corners (both valid positions)
        left_page_num = Text(
            id=0,
            bbox=BBox(5, 190, 15, 198),  # Bottom-left corner
            text="42",
        )
        right_page_num = Text(
            id=1,
            bbox=BBox(85, 190, 95, 198),  # Bottom-right corner
            text="42",
        )

        page_data = PageData(
            page_number=42,
            blocks=[left_page_num, right_page_num],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)
        classifier.score(result)

        # Both should have candidates created
        left_candidate = result.get_candidate_for_block(left_page_num, "page_number")
        right_candidate = result.get_candidate_for_block(right_page_num, "page_number")
        assert left_candidate is not None
        assert right_candidate is not None

        # Both should be constructible
        assert result.build(left_candidate) is not None
        assert result.build(right_candidate) is not None

    def test_page_number_with_font_hints(self) -> None:
        """Test that PageNumberClassifier uses font size hints."""
        hints = FontSizeHints(
            part_count_size=None,
            catalog_part_count_size=None,
            catalog_element_id_size=None,
            step_number_size=None,
            step_repeat_size=None,
            page_number_size=8.0,
            remaining_font_sizes={},
        )
        config = ClassifierConfig(font_size_hints=hints)
        classifier = PageNumberClassifier(config)

        matching_text = Text(text="1", bbox=BBox(10, 90, 18, 98), id=1)
        different_text = Text(text="2", bbox=BBox(80, 90, 92, 102), id=2)

        page_data = PageData(
            page_number=1,
            bbox=BBox(0, 0, 100, 100),
            blocks=[matching_text, different_text],
        )

        result = ClassificationResult(page_data=page_data)
        classifier.score(result)

        # Construct all candidates
        candidates = result.get_candidates("page_number")
        for candidate in candidates:
            if candidate.constructed is None:
                result.build(candidate)

        assert len(candidates) == 2

        matching_candidate = next(
            c for c in candidates if matching_text in c.source_blocks
        )
        different_candidate = next(
            c for c in candidates if different_text in c.source_blocks
        )

        assert matching_candidate.score > different_candidate.score, (
            f"Matching score ({matching_candidate.score}) should be higher than "
            f"different score ({different_candidate.score})"
        )
