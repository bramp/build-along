"""Extract font size hints from text histogram for classification."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from build_a_long.pdf_extract.classifier.text_histogram import (
    TextHistogram,
)
from build_a_long.pdf_extract.extractor import PageData


@dataclass(frozen=True)
class FontSizeHints:
    """Font size hints derived from text histogram analysis.

    This class analyzes the TextHistogram to identify the most common font sizes
    for specific element types (part counts, catalog part counts, step numbers).
    It removes these "known" sizes from the general histogram to help identify
    other element types.
    """

    part_count_size: float | None
    """Most common font size for part counts (e.g., '2x', '3x')"""

    catalog_part_count_size: float | None
    """Most common font size for part counts (catalog listings)"""

    catalog_element_id_size: float | None
    """Most common font size for element IDs (catalog listings)"""

    step_number_size: float | None
    """Most common font size for step numbers"""

    step_repeat_size: float | None
    """Most common font size for step repeat numbers"""

    page_number_size: float | None
    """Most common font size for page numbers (catalog listings)"""

    remaining_font_sizes: Counter[float]
    """Font size distribution after removing known sizes"""

    @classmethod
    def from_pages(cls, pages: list[PageData]) -> FontSizeHints:
        """Extract font size hints from multiple pages.

        This method:
        1. Builds a histogram of text properties across all pages
        2. Identifies the top 3 most common part count font sizes
        3. Assigns them as part_count, catalog_part_count, and step_number sizes
        4. Creates a remaining font size histogram excluding these known sizes

        Args:
            pages: List of PageData objects to analyze.

        Returns:
            FontSizeHints with identified sizes and remaining histogram.
        """

        # Build histogram from pages
        histogram = TextHistogram.from_pages(pages)

        # Get the top 3 most common part count font sizes
        top_part_count_sizes = histogram.part_count_font_sizes.most_common(None)

        part_count_size = (
            top_part_count_sizes[0][0] if len(top_part_count_sizes) >= 1 else None
        )
        catalog_part_count_size = (
            top_part_count_sizes[1][0] if len(top_part_count_sizes) >= 2 else None
        )
        step_number_size = (
            top_part_count_sizes[2][0] if len(top_part_count_sizes) >= 3 else None
        )
        step_repeat_size = (
            top_part_count_sizes[3][0] if len(top_part_count_sizes) >= 4 else None
        )

        element_id_size = (
            histogram.element_id_font_sizes.most_common(1)[0][0]
            if histogram.element_id_font_sizes
            else None
        )

        page_number_size = (
            histogram.page_number_font_sizes.most_common(1)[0][0]
            if histogram.page_number_font_sizes
            else None
        )

        return cls(
            part_count_size=part_count_size,
            catalog_part_count_size=catalog_part_count_size,
            step_number_size=step_number_size,
            step_repeat_size=step_repeat_size,
            catalog_element_id_size=element_id_size,
            page_number_size=page_number_size,
            remaining_font_sizes=histogram.remaining_font_sizes,
        )
