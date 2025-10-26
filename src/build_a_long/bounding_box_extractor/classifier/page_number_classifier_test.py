"""Tests for the page number classifier."""

from build_a_long.bounding_box_extractor.classifier.classifier import classify_elements
from build_a_long.bounding_box_extractor.classifier.page_number_classifier import (
    PageNumberClassifier,
)
from build_a_long.bounding_box_extractor.classifier.types import ClassifierConfig
from build_a_long.bounding_box_extractor.extractor import PageData
from build_a_long.bounding_box_extractor.extractor.bbox import BBox
from build_a_long.bounding_box_extractor.extractor.page_elements import (
    Drawing,
    Text,
)


class TestScorePageNumberText:
    """Tests for the _score_page_number_text function."""

    def test_simple_numbers(self) -> None:
        """Test simple numeric page numbers."""
        pn = PageNumberClassifier(ClassifierConfig(), classifier=None)  # type: ignore[arg-type]
        assert pn._score_page_number_text("1") == 1.0
        assert pn._score_page_number_text("5") == 1.0
        assert pn._score_page_number_text("42") == 1.0
        assert pn._score_page_number_text("123") == 1.0

    def test_leading_zeros(self) -> None:
        """Test page numbers with leading zeros."""
        pn = PageNumberClassifier(ClassifierConfig(), classifier=None)  # type: ignore[arg-type]
        assert pn._score_page_number_text("01") == 0.95
        assert pn._score_page_number_text("001") == 0.95
        assert pn._score_page_number_text("005") == 0.95

    def test_whitespace_handling(self) -> None:
        """Test that whitespace is properly handled."""
        pn = PageNumberClassifier(ClassifierConfig(), classifier=None)  # type: ignore[arg-type]
        assert pn._score_page_number_text("  5  ") == 1.0
        assert pn._score_page_number_text("\t42\n") == 1.0

    def test_non_page_numbers(self) -> None:
        """Test that non-page-number text is rejected."""
        pn = PageNumberClassifier(ClassifierConfig(), classifier=None)  # type: ignore[arg-type]
        assert pn._score_page_number_text("hello") == 0.0
        assert pn._score_page_number_text("Step 3") == 0.0
        assert pn._score_page_number_text("1234") == 0.0  # Too many digits
        assert pn._score_page_number_text("12.5") == 0.0  # Decimal
        assert pn._score_page_number_text("") == 0.0


class TestClassifyPageNumber:
    """Tests for the _classify_page_number function."""

    def test_no_elements(self) -> None:
        """Test classification with no elements."""
        page_data = PageData(
            page_number=1,
            elements=[],
            bbox=BBox(0, 0, 100, 200),
        )
        # Run end-to-end classification; should not raise any errors
        classify_elements([page_data])

    def test_single_page_number_bottom_left(self) -> None:
        """Test identifying a page number in the bottom-left corner."""
        page_bbox = BBox(0, 0, 100, 200)
        page_number_text = Text(
            bbox=BBox(5, 190, 15, 198),  # Bottom-left position
            text="1",
        )

        page_data = PageData(
            page_number=1,
            elements=[page_number_text],
            bbox=page_bbox,
        )

        classify_elements([page_data])

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
            elements=[page_number_text],
            bbox=page_bbox,
        )

        classify_elements([page_data])

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
            elements=[center_text, corner_text],
            bbox=page_bbox,
        )

        classify_elements([page_data])

        # Corner should have higher score
        assert (
            corner_text.label_scores["page_number"]
            > center_text.label_scores["page_number"]
        )
        assert corner_text.label == "page_number"
        assert center_text.label is None

    def test_prefer_numeric_match_to_page_index(self) -> None:
        """Prefer element whose numeric value equals PageData.page_number."""
        page_bbox = BBox(0, 0, 100, 200)
        # Two numbers, both near bottom, but only one matches the page number 7
        txt6 = Text(bbox=BBox(10, 190, 14, 196), text="6")
        txt7 = Text(bbox=BBox(90, 190, 94, 196), text="7")

        page_data = PageData(
            page_number=7,
            elements=[txt6, txt7],
            bbox=page_bbox,
        )

        classify_elements([page_data])

        assert txt7.label == "page_number"
        assert txt6.label is None

    def test_remove_near_duplicate_bboxes(self) -> None:
        """After choosing page number, remove nearly identical shadow/duplicate elements."""
        page_bbox = BBox(0, 0, 100, 200)
        # Chosen page number
        pn = Text(bbox=BBox(10, 190, 14, 196), text="3")
        # Very similar drawing (e.g., stroke/shadow) almost same bbox
        dup = Drawing(bbox=BBox(10.2, 190.1, 14.1, 195.9))

        page_data = PageData(
            page_number=3,
            elements=[pn, dup],
            bbox=page_bbox,
        )

        classify_elements([page_data])

        # Page number kept and labeled; duplicate marked as deleted
        assert pn.label == "page_number"
        assert pn in page_data.elements
        assert dup in page_data.elements
        assert dup.deleted is True
        assert pn.deleted is False

    def test_non_numeric_text_scores_low(self) -> None:
        """Test that non-numeric text scores low."""
        page_bbox = BBox(0, 0, 100, 200)
        text_element = Text(
            bbox=BBox(5, 190, 50, 198),  # Bottom-left position
            text="Hello World",
        )

        page_data = PageData(
            page_number=1,
            elements=[text_element],
            bbox=page_bbox,
        )

        classify_elements([page_data])

        # Should have low score due to text pattern (position is good but text is bad)
        assert text_element.label_scores["page_number"] < 0.5
        assert text_element.label is None
