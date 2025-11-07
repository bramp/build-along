"""Text histogram for analyzing font properties across pages."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.page_blocks import Text


@dataclass
class TextHistogram:
    """Global statistics about text elements across all pages.

    This data structure collects font size and font name distributions
    to help guide classification decisions.
    """

    font_size_counts: Counter[float]
    font_name_counts: Counter[str]
    part_count_font_sizes: Counter[float]
    r"""Font sizes for text matching \dx pattern (e.g., '2x', '3x')"""

    page_number_font_sizes: Counter[float]
    """Font sizes for text matching page numbers (±1 from current page)"""

    @classmethod
    def from_pages(cls, pages: list[PageData]) -> TextHistogram:
        """Build a histogram from all text elements across all pages.

        Args:
            pages: List of PageData objects to analyze.

        Returns:
            A TextHistogram containing font size and name distributions.
        """
        font_size_counts: Counter[float] = Counter()
        font_name_counts: Counter[str] = Counter()

        # Pattern for part counts like "2x", "3x", etc.
        part_count_pattern = re.compile(r"^\d+x$", re.IGNORECASE)

        # Track font sizes for specific patterns
        part_count_font_size_counter: Counter[float] = Counter()
        page_number_font_size_counter: Counter[float] = Counter()

        for page in pages:
            for block in page.blocks:
                if isinstance(block, Text):
                    if block.font_size is not None:
                        font_size_counts[block.font_size] += 1

                        # Check if text matches part count pattern (\dx)
                        if part_count_pattern.match(block.text.strip()):
                            part_count_font_size_counter[block.font_size] += 1

                        # Check if text matches page number (±1 from current)
                        text_stripped = block.text.strip()
                        if text_stripped.isdigit():
                            text_num = int(text_stripped)
                            page_num = page.page_number
                            if abs(text_num - page_num) <= 1:
                                page_number_font_size_counter[block.font_size] += 1

                    if block.font_name is not None:
                        font_name_counts[block.font_name] += 1

        return cls(
            font_size_counts=font_size_counts,
            font_name_counts=font_name_counts,
            part_count_font_sizes=part_count_font_size_counter,
            page_number_font_sizes=page_number_font_size_counter,
        )
