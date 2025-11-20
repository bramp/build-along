"""Tests for the page number classifier."""

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import (
    Text,
)


class TestPageNumberClassification:
    """Tests for page number detection and scoring."""

    def test_no_page_numbers_on_empty_page(self) -> None:
        """Test that classification succeeds on a page with no elements."""
        page_data = PageData(
            page_number=1,
            blocks=[],
            bbox=BBox(0, 0, 100, 200),
        )
        # Run end-to-end classification; should not raise any errors
        classify_elements(page_data)

    def test_single_page_number_bottom_left(self) -> None:
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

        results = classify_elements(page_data)

        # Check that the Page was built with the correct page_number
        assert results.page is not None
        assert results.page.page_number is not None
        assert results.page.page_number.value == 1
        assert results.page.page_number.bbox == page_number_text.bbox

        # Check the page_number candidate exists and has a good score
        candidate = results.get_candidate_for_block(page_number_text, "page_number")
        assert candidate is not None
        assert candidate.constructed is not None
        assert candidate.score > 0.5

    def test_single_page_number_bottom_right(self) -> None:
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

        result = classify_elements(page_data)

        # Check that the Page was built with the correct page_number
        assert result.page is not None
        assert result.page.page_number is not None
        assert result.page.page_number.value == 5
        assert result.page.page_number.bbox == page_number_text.bbox

        # Check the page_number candidate exists and has a good score
        candidate = result.get_candidate_for_block(page_number_text, "page_number")
        assert candidate is not None
        assert candidate.constructed is not None
        assert candidate.score > 0.5

    def test_multiple_candidates_prefer_corners(self) -> None:
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

        result = classify_elements(page_data)

        # Check that the Page was built with the corner text (higher score)
        assert result.page is not None
        assert result.page.page_number is not None
        assert result.page.page_number.value == 3
        assert result.page.page_number.bbox == corner_text.bbox

        # Check scores from ClassificationResult
        corner_candidate = result.get_candidate_for_block(corner_text, "page_number")
        center_candidate = result.get_candidate_for_block(center_text, "page_number")
        assert corner_candidate is not None
        assert center_candidate is not None
        assert corner_candidate.score > center_candidate.score

    def test_prefer_numeric_match_to_page_index(self) -> None:
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

        result = classify_elements(page_data)

        # Check that txt7 was selected (matches page number)
        assert result.page is not None
        assert result.page.page_number is not None
        assert result.page.page_number.value == 7
        assert result.page.page_number.bbox == txt7.bbox

    def test_non_numeric_text_scores_low(self) -> None:
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

        result = classify_elements(page_data)

        # Should not be labeled due to text pattern (position is good but text is bad)
        # Check that no successful page_number candidates exist with this block as source
        candidate = result.get_candidate_for_block(text_block, "page_number")
        assert candidate is not None
        assert candidate.constructed is None

        # Check that score is low
        assert candidate.score < 0.5
