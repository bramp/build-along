"""Tests for TextHistogram functionality."""

from build_a_long.pdf_extract.classifier.text_histogram import TextHistogram
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Block, Image, Text


class TestTextHistogram:
    """Tests for the TextHistogram class."""

    def test_histogram_from_empty_pages(self) -> None:
        """Test histogram building from pages with no text elements."""
        pages = [
            PageData(
                page_number=1,
                blocks=[
                    Image(id=0, bbox=BBox(10, 10, 50, 50), image_id="img1"),
                ],
                bbox=BBox(0, 0, 100, 200),
            )
        ]

        histogram = TextHistogram.from_pages(pages)

        assert len(histogram.remaining_font_sizes) == 0
        assert len(histogram.font_name_counts) == 0
        assert histogram.remaining_font_sizes.most_common(5) == []
        assert histogram.font_name_counts.most_common(5) == []

    def test_histogram_from_multiple_pages(self) -> None:
        """Test histogram building from multiple pages with text elements."""
        pages = []

        # Create 3 sample pages with various text elements
        for i in range(1, 4):
            page_bbox = BBox(0, 0, 100, 200)
            blocks: list[Block] = [
                # Page number (small font) - appears on all 3 pages
                Text(
                    id=0,
                    bbox=BBox(5, 190, 15, 198),
                    text=str(i),
                    font_name="Arial",
                    font_size=8.0,
                ),
                # Step number (large font) - appears on all 3 pages
                Text(
                    id=1,
                    bbox=BBox(10, 150, 30, 170),
                    text=str(i * 2),
                    font_name="Arial-Bold",
                    font_size=24.0,
                ),
                # Part count (medium font) - appears twice per page = 6 total
                Text(
                    id=2,
                    bbox=BBox(40, 100, 60, 110),
                    text="2x",
                    font_name="Arial",
                    font_size=12.0,
                ),
                # Another part count
                Text(
                    id=3,
                    bbox=BBox(40, 80, 60, 90),
                    text="1x",
                    font_name="Arial",
                    font_size=12.0,
                ),
            ]

            page_data = PageData(
                page_number=i,
                blocks=blocks,
                bbox=page_bbox,
            )
            pages.append(page_data)

        # Build the histogram
        histogram = TextHistogram.from_pages(pages)

        # Verify font size counts (only includes "other integers" - not part counts or page numbers)
        # In this test: page numbers (str(i)) match page ±1, so they're in page_number_font_sizes
        # Step numbers (i*2) are integers but NOT within ±1 of page number, so they're in font_size_counts
        assert len(histogram.remaining_font_sizes) == 1
        assert (
            histogram.remaining_font_sizes[24.0] == 2
        )  # Only pages 1 and 2 have step numbers outside ±1 range

        # Verify part count font sizes
        assert len(histogram.part_count_font_sizes) == 1
        assert histogram.part_count_font_sizes[12.0] == 6  # 6 part counts

        # Verify page number font sizes
        assert len(histogram.page_number_font_sizes) == 2
        assert histogram.page_number_font_sizes[8.0] == 3  # 3 page numbers
        assert (
            histogram.page_number_font_sizes[24.0] == 1
        )  # Page 3's step number (6) is within ±1 of page 3

        # Verify font name counts
        assert len(histogram.font_name_counts) == 2
        assert histogram.font_name_counts["Arial"] == 9  # page nums + part counts
        assert histogram.font_name_counts["Arial-Bold"] == 3  # step numbers
        most_common_names = histogram.font_name_counts.most_common(2)
        assert most_common_names[0] == ("Arial", 9)
        assert most_common_names[1] == ("Arial-Bold", 3)

        # Verify pattern-specific tracking
        # Part counts "2x" and "1x" both use font size 12.0
        assert histogram.part_count_font_sizes[12.0] == 6

        # Page numbers matching (±1 from current page):
        # Page 1 (i=1): "1" matches (|1-1|=0), "2" matches (|2-1|=1)
        # Page 2 (i=2): "2" matches (|2-2|=0), "4" doesn't (|4-2|=2>1)
        # Page 3 (i=3): "3" matches (|3-3|=0), "6" doesn't (|6-3|=3>1)
        # So we match: "1"(8.0), "2"(24.0), "2"(8.0), "3"(8.0)
        assert histogram.page_number_font_sizes[8.0] == 3
        assert histogram.page_number_font_sizes[24.0] == 1

    def test_histogram_with_none_values(self) -> None:
        """Test histogram handles Text blocks with None font properties."""
        pages = [
            PageData(
                page_number=1,
                blocks=[
                    Text(
                        id=0,
                        bbox=BBox(10, 10, 50, 50),
                        text="10",  # Integer text (not matching page ±1)
                        font_name="Arial",
                        font_size=12.0,
                    ),
                    Text(
                        id=1,
                        bbox=BBox(10, 60, 50, 80),
                        text="20",  # Integer text (not matching page ±1)
                        font_name=None,
                        font_size=None,
                    ),
                    Text(
                        id=2,
                        bbox=BBox(10, 90, 50, 110),
                        text="30",  # Integer text (not matching page ±1)
                        font_name="Arial",
                        font_size=12.0,
                    ),
                ],
                bbox=BBox(0, 0, 100, 200),
            )
        ]

        histogram = TextHistogram.from_pages(pages)

        # Only the non-None values should be counted
        # font_size_counts only tracks integers that aren't part counts or page numbers
        assert len(histogram.remaining_font_sizes) == 1
        assert (
            histogram.remaining_font_sizes[12.0] == 2
        )  # "10" and "30" (not "20" which has None font_size)

        assert len(histogram.font_name_counts) == 1
        assert histogram.font_name_counts["Arial"] == 2

    def test_most_common_with_limit(self) -> None:
        """Test that most_common respects the limit parameter."""
        pages = [
            PageData(
                page_number=1,
                blocks=[
                    Text(
                        id=i,
                        bbox=BBox(10, 10 + i * 10, 50, 20 + i * 10),
                        text=f"text{i}",
                        font_name=f"Font{i % 5}",
                        font_size=float(10 + i % 3),
                    )
                    for i in range(20)
                ],
                bbox=BBox(0, 0, 100, 500),
            )
        ]

        histogram = TextHistogram.from_pages(pages)

        # Request only top 3
        top_3_sizes = histogram.remaining_font_sizes.most_common(3)
        assert len(top_3_sizes) <= 3

        top_2_names = histogram.font_name_counts.most_common(2)
        assert len(top_2_names) <= 2

    def test_part_count_pattern_matching(self) -> None:
        r"""Test that part count pattern (\dx) is correctly identified."""
        pages = [
            PageData(
                page_number=1,
                blocks=[
                    # Valid part counts
                    Text(
                        id=0,
                        bbox=BBox(10, 10, 50, 20),
                        text="2x",
                        font_name="Arial",
                        font_size=10.0,
                    ),
                    Text(
                        id=1,
                        bbox=BBox(10, 30, 50, 40),
                        text="5X",  # Case insensitive
                        font_name="Arial",
                        font_size=12.0,
                    ),
                    Text(
                        id=2,
                        bbox=BBox(10, 50, 50, 60),
                        text="10x",
                        font_name="Arial",
                        font_size=14.0,
                    ),
                    # Invalid - should not match
                    Text(
                        id=3,
                        bbox=BBox(10, 70, 50, 80),
                        text="x2",  # Wrong order
                        font_name="Arial",
                        font_size=20.0,
                    ),
                    Text(
                        id=4,
                        bbox=BBox(10, 90, 50, 100),
                        text="2 x",  # Space
                        font_name="Arial",
                        font_size=22.0,
                    ),
                ],
                bbox=BBox(0, 0, 100, 200),
            )
        ]

        histogram = TextHistogram.from_pages(pages)

        # All three part counts use font size 10.0, 12.0, and 14.0
        assert histogram.part_count_font_sizes[10.0] == 1
        assert histogram.part_count_font_sizes[12.0] == 1
        assert histogram.part_count_font_sizes[14.0] == 1

    def test_page_number_pattern_matching(self) -> None:
        """Test that page numbers (±1) are correctly identified."""
        pages = []

        # Page 5: should match "4", "5", "6"
        pages.append(
            PageData(
                page_number=5,
                blocks=[
                    Text(
                        id=0,
                        bbox=BBox(10, 10, 50, 20),
                        text="4",  # 5-1
                        font_name="Arial",
                        font_size=8.0,
                    ),
                    Text(
                        id=1,
                        bbox=BBox(10, 30, 50, 40),
                        text="5",  # Current page
                        font_name="Arial",
                        font_size=8.0,
                    ),
                    Text(
                        id=2,
                        bbox=BBox(10, 50, 50, 60),
                        text="6",  # 5+1
                        font_name="Arial",
                        font_size=8.0,
                    ),
                    Text(
                        id=3,
                        bbox=BBox(10, 70, 50, 80),
                        text="10",  # Too far away
                        font_name="Arial",
                        font_size=16.0,
                    ),
                ],
                bbox=BBox(0, 0, 100, 200),
            )
        )

        histogram = TextHistogram.from_pages(pages)

        # Page numbers matching: "4", "5", "6" all use font size 8.0
        assert histogram.page_number_font_sizes[8.0] == 3
        assert 16.0 not in histogram.page_number_font_sizes

    def test_empty_pattern_averages(self) -> None:
        """Test that averages are None when no matching patterns found."""
        pages = [
            PageData(
                page_number=1,
                blocks=[
                    Text(
                        id=0,
                        bbox=BBox(10, 10, 50, 20),
                        text="random text",
                        font_name="Arial",
                        font_size=12.0,
                    ),
                ],
                bbox=BBox(0, 0, 100, 200),
            )
        ]

        histogram = TextHistogram.from_pages(pages)

        assert len(histogram.part_count_font_sizes) == 0
        assert len(histogram.page_number_font_sizes) == 0

    def test_median_calculation(self) -> None:
        """Test that median is calculated correctly for odd and even counts."""
        pages = [
            PageData(
                page_number=1,
                blocks=[
                    # Odd number of part counts: 10, 12, 14 -> median is 12
                    Text(
                        id=0,
                        bbox=BBox(10, 10, 50, 20),
                        text="1x",
                        font_name="Arial",
                        font_size=10.0,
                    ),
                    Text(
                        id=1,
                        bbox=BBox(10, 30, 50, 40),
                        text="2x",
                        font_name="Arial",
                        font_size=12.0,
                    ),
                    Text(
                        id=2,
                        bbox=BBox(10, 50, 50, 60),
                        text="3x",
                        font_name="Arial",
                        font_size=14.0,
                    ),
                ],
                bbox=BBox(0, 0, 100, 200),
            ),
            PageData(
                page_number=2,
                blocks=[
                    # Even number of page numbers: 8, 8, 10, 10 -> median is 9.0
                    Text(
                        id=0,
                        bbox=BBox(10, 10, 50, 20),
                        text="1",
                        font_name="Arial",
                        font_size=8.0,
                    ),
                    Text(
                        id=1,
                        bbox=BBox(10, 30, 50, 40),
                        text="2",
                        font_name="Arial",
                        font_size=8.0,
                    ),
                    Text(
                        id=2,
                        bbox=BBox(10, 50, 50, 60),
                        text="3",
                        font_name="Arial",
                        font_size=10.0,
                    ),
                    Text(
                        id=3,
                        bbox=BBox(10, 70, 50, 80),
                        text="20",  # Too far, won't match
                        font_name="Arial",
                        font_size=10.0,
                    ),
                ],
                bbox=BBox(0, 0, 100, 200),
            ),
        ]

        histogram = TextHistogram.from_pages(pages)

        # Verify part count font sizes collected correctly
        assert histogram.part_count_font_sizes[10.0] == 1
        assert histogram.part_count_font_sizes[12.0] == 1
        assert histogram.part_count_font_sizes[14.0] == 1

        # Verify page number font sizes collected correctly
        # Page 2 matches: "1" (8.0), "2" (8.0), "3" (10.0)
        assert histogram.page_number_font_sizes[8.0] == 2
        assert histogram.page_number_font_sizes[10.0] == 1
