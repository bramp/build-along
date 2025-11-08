"""Tests for font_size_hints module."""

from collections import Counter

from build_a_long.pdf_extract.classifier.font_size_hints import FontSizeHints
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Text


def test_from_pages_with_all_sizes() -> None:
    """Test hint extraction when we have all 3 top sizes."""
    # Create mock pages with text blocks
    page1 = PageData(
        page_number=1,
        bbox=BBox(0, 0, 100, 100),
        blocks=[
            Text(bbox=BBox(10, 10, 20, 20), text="2x", font_size=10.0, id=1),
            Text(bbox=BBox(10, 30, 20, 40), text="3x", font_size=10.0, id=2),
            Text(bbox=BBox(10, 50, 20, 60), text="4x", font_size=12.0, id=3),
            Text(bbox=BBox(10, 70, 20, 80), text="5x", font_size=14.0, id=4),
            Text(
                bbox=BBox(30, 10, 40, 20), text="99", font_size=16.0, id=5
            ),  # Integer not matching page ±1
        ],
    )

    hints = FontSizeHints.from_pages([page1])

    assert hints.part_count_size == 10.0  # Most common (2 occurrences)
    assert hints.catalog_part_count_size == 12.0  # 2nd most common
    assert hints.step_number_size == 14.0  # 3rd most common

    # Remaining should only have "other integers" (not part counts or page numbers)
    assert hints.remaining_font_sizes == Counter({16.0: 1})


def test_from_pages_with_two_sizes() -> None:
    """Test hint extraction when we only have 2 part count sizes."""
    page1 = PageData(
        page_number=1,
        bbox=BBox(0, 0, 100, 100),
        blocks=[
            Text(bbox=BBox(10, 10, 20, 20), text="2x", font_size=10.0, id=1),
            Text(bbox=BBox(10, 30, 20, 40), text="3x", font_size=12.0, id=2),
        ],
    )

    hints = FontSizeHints.from_pages([page1])

    assert hints.part_count_size == 10.0
    assert hints.catalog_part_count_size == 12.0
    assert hints.step_number_size is None  # Not enough data

    # Remaining should be empty since both sizes are known
    assert hints.remaining_font_sizes == Counter()


def test_from_pages_with_one_size() -> None:
    """Test hint extraction when we only have 1 part count size."""
    page1 = PageData(
        page_number=1,
        bbox=BBox(0, 0, 100, 100),
        blocks=[
            Text(bbox=BBox(10, 10, 20, 20), text="2x", font_size=10.0, id=1),
        ],
    )

    hints = FontSizeHints.from_pages([page1])

    assert hints.part_count_size == 10.0
    assert hints.catalog_part_count_size is None
    assert hints.step_number_size is None

    # Remaining should be empty since the only size is known
    assert hints.remaining_font_sizes == Counter()


def test_from_pages_with_no_part_counts() -> None:
    """Test hint extraction when there are no part count patterns."""
    page1 = PageData(
        page_number=1,
        bbox=BBox(0, 0, 100, 100),
        blocks=[
            Text(
                bbox=BBox(10, 10, 20, 20), text="10", font_size=10.0, id=1
            ),  # Integer not matching page ±1
            Text(
                bbox=BBox(10, 30, 20, 40), text="20", font_size=12.0, id=2
            ),  # Integer not matching page ±1
        ],
    )

    hints = FontSizeHints.from_pages([page1])

    assert hints.part_count_size is None
    assert hints.catalog_part_count_size is None
    assert hints.step_number_size is None

    # All "other integer" sizes should remain
    assert hints.remaining_font_sizes == Counter({10.0: 1, 12.0: 1})


def test_remaining_font_sizes_preserves_counts() -> None:
    """Test that remaining font sizes preserve their original counts."""
    page1 = PageData(
        page_number=1,
        bbox=BBox(0, 0, 100, 100),
        blocks=[
            Text(bbox=BBox(10, 10, 20, 20), text="2x", font_size=10.0, id=1),
            Text(bbox=BBox(10, 30, 20, 40), text="3x", font_size=12.0, id=2),
            Text(bbox=BBox(10, 50, 20, 60), text="50", font_size=14.0, id=3),  # Integer
            Text(bbox=BBox(10, 70, 20, 80), text="60", font_size=14.0, id=4),  # Integer
            Text(bbox=BBox(30, 10, 40, 20), text="70", font_size=16.0, id=5),  # Integer
        ],
    )

    hints = FontSizeHints.from_pages([page1])

    # Only "other integer" sizes should remain (not part counts)
    assert hints.remaining_font_sizes == Counter({14.0: 2, 16.0: 1})
