"""Tests for the element classifier."""

from build_a_long.bounding_box_extractor.classifier.classifier import (
    _score_page_number_text,
    _classify_page_number,
    classify_elements,
)
from build_a_long.bounding_box_extractor.extractor import PageData
from build_a_long.bounding_box_extractor.extractor.bbox import BBox
from build_a_long.bounding_box_extractor.extractor.page_elements import (
    Root,
    Text,
)


class TestScorePageNumberText:
    """Tests for the _score_page_number_text function."""

    def test_simple_numbers(self) -> None:
        """Test simple numeric page numbers."""
        assert _score_page_number_text("1") == 1.0
        assert _score_page_number_text("5") == 1.0
        assert _score_page_number_text("42") == 1.0
        assert _score_page_number_text("123") == 1.0

    def test_leading_zeros(self) -> None:
        """Test page numbers with leading zeros."""
        assert _score_page_number_text("01") == 0.95
        assert _score_page_number_text("001") == 0.95
        assert _score_page_number_text("005") == 0.95

    def test_formatted_page_numbers(self) -> None:
        """Test formatted page numbers."""
        assert _score_page_number_text("Page 5") == 0.85
        assert _score_page_number_text("page 42") == 0.85
        assert _score_page_number_text("p.5") == 0.85
        assert _score_page_number_text("P.123") == 0.85
        assert _score_page_number_text("p 7") == 0.85

    def test_whitespace_handling(self) -> None:
        """Test that whitespace is properly handled."""
        assert _score_page_number_text("  5  ") == 1.0
        assert _score_page_number_text("\t42\n") == 1.0

    def test_non_page_numbers(self) -> None:
        """Test that non-page-number text is rejected."""
        assert _score_page_number_text("hello") == 0.0
        assert _score_page_number_text("Step 3") == 0.0
        assert _score_page_number_text("1234") == 0.0  # Too many digits
        assert _score_page_number_text("12.5") == 0.0  # Decimal
        assert _score_page_number_text("") == 0.0


class TestClassifyPageNumber:
    """Tests for the _classify_page_number function."""

    def test_no_elements(self) -> None:
        """Test classification with no elements."""
        page_data = PageData(
            page_number=1,
            root=Root(bbox=BBox(0, 0, 100, 200)),
            elements=[],
        )
        # First calculate scores (would normally be done by classify_elements)
        from build_a_long.bounding_box_extractor.classifier.classifier import (
            _calculate_page_number_scores,
        )

        _calculate_page_number_scores(page_data)
        _classify_page_number(page_data)
        # Should not raise any errors

    def test_single_page_number_bottom_left(self) -> None:
        """Test identifying a page number in the bottom-left corner."""
        page_bbox = BBox(0, 0, 100, 200)
        page_number_text = Text(
            bbox=BBox(5, 190, 15, 198),  # Bottom-left position
            text="1",
        )

        page_data = PageData(
            page_number=1,
            root=Root(bbox=page_bbox),
            elements=[page_number_text],
        )

        # Calculate scores first
        from build_a_long.bounding_box_extractor.classifier.classifier import (
            _calculate_page_number_scores,
        )

        _calculate_page_number_scores(page_data)
        _classify_page_number(page_data)

        assert page_number_text.label == "page_number"
        assert "page_number" in page_number_text.label_scores
        assert page_number_text.label_scores["page_number"] > 0.5

    def test_single_page_number_bottom_right(self) -> None:
        """Test identifying a page number in the bottom-right corner."""
        page_bbox = BBox(0, 0, 100, 200)
        page_number_text = Text(
            bbox=BBox(90, 190, 98, 198),  # Bottom-right position
            text="5",
        )

        page_data = PageData(
            page_number=1,
            root=Root(bbox=page_bbox),
            elements=[page_number_text],
        )

        from build_a_long.bounding_box_extractor.classifier.classifier import (
            _calculate_page_number_scores,
        )

        _calculate_page_number_scores(page_data)
        _classify_page_number(page_data)

        assert page_number_text.label == "page_number"
        assert "page_number" in page_number_text.label_scores

    def test_multiple_candidates_prefer_corners(self) -> None:
        """Test that corner elements score higher than center ones."""
        page_bbox = BBox(0, 0, 100, 200)

        # Element in center-bottom (less preferred)
        center_text = Text(
            bbox=BBox(45, 190, 55, 198),
            text="2",
        )

        # Element in corner (more preferred)
        corner_text = Text(
            bbox=BBox(5, 190, 15, 198),
            text="3",
        )

        page_data = PageData(
            page_number=1,
            root=Root(bbox=page_bbox),
            elements=[center_text, corner_text],
        )

        from build_a_long.bounding_box_extractor.classifier.classifier import (
            _calculate_page_number_scores,
        )

        _calculate_page_number_scores(page_data)

        # Corner should have higher score
        assert (
            corner_text.label_scores["page_number"]
            > center_text.label_scores["page_number"]
        )

        _classify_page_number(page_data)
        assert corner_text.label == "page_number"
        assert center_text.label is None

    def test_not_in_bottom_region(self) -> None:
        """Test that elements outside bottom region score lower due to position."""
        page_bbox = BBox(0, 0, 100, 200)
        top_text = Text(
            bbox=BBox(5, 10, 15, 18),  # Top of page
            text="1",
        )

        page_data = PageData(
            page_number=1,
            root=Root(bbox=page_bbox),
            elements=[top_text],
        )

        from build_a_long.bounding_box_extractor.classifier.classifier import (
            _calculate_page_number_scores,
        )

        _calculate_page_number_scores(page_data)

        # Should have score dominated by text (position score is 0.0)
        # Score = 0.7 * 1.0 (text) + 0.3 * 0.0 (position) = 0.7
        assert top_text.label_scores["page_number"] == 0.7

        _classify_page_number(page_data)
        # Still gets labeled since it's the only candidate with score > threshold
        # In real scenarios, there would be other elements with better positions
        assert top_text.label == "page_number"

    def test_non_numeric_text_scores_low(self) -> None:
        """Test that non-numeric text scores low."""
        page_bbox = BBox(0, 0, 100, 200)
        text_element = Text(
            bbox=BBox(5, 190, 50, 198),  # Bottom-left position
            text="Hello World",
        )

        page_data = PageData(
            page_number=1,
            root=Root(bbox=page_bbox),
            elements=[text_element],
        )

        from build_a_long.bounding_box_extractor.classifier.classifier import (
            _calculate_page_number_scores,
        )

        _calculate_page_number_scores(page_data)

        # Should have low score due to text pattern (position is good but text is bad)
        assert text_element.label_scores["page_number"] < 0.5

        _classify_page_number(page_data)
        assert text_element.label is None


class TestClassifyElements:
    """Tests for the main classify_elements function."""

    def test_classify_multiple_pages(self) -> None:
        """Test classification across multiple pages."""
        pages = []
        for i in range(1, 4):
            page_bbox = BBox(0, 0, 100, 200)
            page_number_text = Text(
                bbox=BBox(5, 190, 15, 198),
                text=str(i),
            )

            page_data = PageData(
                page_number=i,
                root=Root(bbox=page_bbox),
                elements=[page_number_text],
            )
            pages.append(page_data)

        classify_elements(pages)

        # Verify all pages have their page numbers labeled and scored
        for page_data in pages:
            labeled_elements = [
                e
                for e in page_data.elements
                if isinstance(e, Text) and e.label == "page_number"
            ]
            assert len(labeled_elements) == 1
            # Check that scores were calculated
            assert "page_number" in labeled_elements[0].label_scores
            assert labeled_elements[0].label_scores["page_number"] > 0.5

    def test_empty_pages_list(self) -> None:
        """Test with an empty list of pages."""
        classify_elements([])
        # Should not raise any errors
