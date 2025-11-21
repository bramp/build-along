"""Tests for the bag number classifier."""

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import BagNumber
from build_a_long.pdf_extract.extractor.page_blocks import Text


class TestBagNumberClassification:
    """Tests for bag number detection and scoring."""

    def test_no_bag_numbers_on_empty_page(self) -> None:
        """Test that classification succeeds on a page with no elements."""
        page_data = PageData(
            page_number=1,
            blocks=[],
            bbox=BBox(0, 0, 100, 200),
        )
        # Run end-to-end classification; should not raise any errors
        classify_elements(page_data)

    def test_single_bag_number_top_left(self) -> None:
        """Test identifying a bag number in the top-left area."""
        page_bbox = BBox(0, 0, 552.76, 496.06)
        bag_number_text = Text(
            id=0,
            bbox=BBox(113.93, 104.82, 148.61, 169.89),  # Top-left position
            text="1",
            font_name="CeraPro-Medium",
            font_size=50.0,
        )

        page_data = PageData(
            page_number=10,
            blocks=[bag_number_text],
            bbox=page_bbox,
        )

        result = classify_elements(page_data)

        # Check the bag_number candidate exists and has a good score
        candidate = result.get_candidate_for_block(bag_number_text, "bag_number")
        assert candidate is not None
        assert candidate.constructed is not None
        assert isinstance(candidate.constructed, BagNumber)
        bag_number = candidate.constructed
        assert bag_number.value == 1
        assert candidate.score > 0.5

    def test_bag_number_requires_numeric_text(self) -> None:
        """Test that non-numeric text is not classified as bag number."""
        page_bbox = BBox(0, 0, 552.76, 496.06)
        text_block = Text(
            id=0,
            bbox=BBox(113.93, 104.82, 148.61, 169.89),
            text="ABC",
            font_size=50.0,
        )

        page_data = PageData(
            page_number=10,
            blocks=[text_block],
            bbox=page_bbox,
        )

        result = classify_elements(page_data)

        # Should not classify non-numeric text as bag number
        candidate = result.get_candidate_for_block(text_block, "bag_number")
        assert candidate is None

    def test_bag_number_rejects_bottom_of_page(self) -> None:
        """Test that numbers at bottom of page are not classified as bag numbers."""
        page_bbox = BBox(0, 0, 552.76, 496.06)
        # Place text at bottom (y > 40% of page height)
        bottom_text = Text(
            id=0,
            bbox=BBox(113.93, 400, 148.61, 450),  # Near bottom
            text="1",
            font_size=50.0,
        )

        page_data = PageData(
            page_number=10,
            blocks=[bottom_text],
            bbox=page_bbox,
        )

        result = classify_elements(page_data)

        # Should not classify bottom text as bag number
        candidate = result.get_candidate_for_block(bottom_text, "bag_number")
        assert candidate is None

    def test_bag_number_accepts_values_1_to_99(self) -> None:
        """Test that bag numbers accept values from 1 to 99."""
        page_bbox = BBox(0, 0, 552.76, 496.06)

        for value in [1, 2, 5, 10, 50, 99]:
            bag_number_text = Text(
                id=0,
                bbox=BBox(113.93, 104.82, 148.61, 169.89),
                text=str(value),
                font_size=50.0,
            )

            page_data = PageData(
                page_number=10,
                blocks=[bag_number_text],
                bbox=page_bbox,
            )

            result = classify_elements(page_data)
            candidate = result.get_candidate_for_block(bag_number_text, "bag_number")
            assert candidate is not None
            assert candidate.constructed is not None
            assert isinstance(candidate.constructed, BagNumber)
            bag_number = candidate.constructed
            assert bag_number.value == value

    def test_bag_number_rejects_zero_and_large_numbers(self) -> None:
        """Test that bag numbers reject 0 and numbers > 99."""
        page_bbox = BBox(0, 0, 552.76, 496.06)

        for invalid_value in ["0", "100", "999"]:
            text_block = Text(
                id=0,
                bbox=BBox(113.93, 104.82, 148.61, 169.89),
                text=invalid_value,
                font_size=50.0,
            )

            page_data = PageData(
                page_number=10,
                blocks=[text_block],
                bbox=page_bbox,
            )

            result = classify_elements(page_data)
            candidate = result.get_candidate_for_block(text_block, "bag_number")
            assert candidate is None

    def test_bag_number_prefers_larger_font_sizes(self) -> None:
        """Test that only large font sizes are accepted as bag numbers."""
        page_bbox = BBox(0, 0, 552.76, 496.06)

        # Small font size (should be rejected - likely step number)
        small_text = Text(
            id=0,
            bbox=BBox(113.93, 104.82, 120, 110),
            text="1",
            font_size=26.0,  # Too small for bag number
        )

        # Large font size (should be accepted)
        large_text = Text(
            id=1,
            bbox=BBox(200, 100, 240, 160),
            text="2",
            font_size=50.0,  # Large - typical bag number
        )

        page_data = PageData(
            page_number=10,
            blocks=[small_text, large_text],
            bbox=page_bbox,
        )

        result = classify_elements(page_data)

        small_candidate = result.get_candidate_for_block(small_text, "bag_number")
        large_candidate = result.get_candidate_for_block(large_text, "bag_number")

        # Small font should be rejected, large should be accepted
        assert small_candidate is None, (
            "Small font sizes should not create bag number candidates"
        )
        assert large_candidate is not None, (
            "Large font sizes should create bag number candidates"
        )
