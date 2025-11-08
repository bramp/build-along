"""Text histogram for analyzing font properties across pages."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.page_blocks import Text


@dataclass
class TextHistogram:
    """Global statistics about numeric text elements across all pages.

    This data structure collects font size and font name distributions
    to help guide classification decisions.
    """

    font_name_counts: Counter[str]

    part_count_font_sizes: Counter[float]
    r"""Font sizes for text matching \dx pattern (e.g., '2x', '3x')"""

    page_number_font_sizes: Counter[float]
    """Font sizes for text matching page numbers (±1 from current page)"""

    element_id_font_sizes: Counter[float]
    """Font sizes for text matching Element IDs (6-7 digit numbers)"""

    remaining_font_sizes: Counter[float]
    """Font sizes for all other integer text elements"""

    @classmethod
    def from_pages(cls, pages: list[PageData]) -> TextHistogram:
        """Build a histogram from all text elements across all pages.

        Args:
            pages: List of PageData objects to analyze.

        Returns:
            A TextHistogram containing font size and name distributions.
        """
        font_name_counts: Counter[str] = Counter()

        # TODO Ensure this matches the part count classifier (used elsewhere)
        # Pattern for part counts like "2x", "3x", etc.
        part_count_pattern = re.compile(r"^\d+x$", re.IGNORECASE)

        # Size of fonts that match part count pattern
        part_count_font_size_counter: Counter[float] = Counter()

        # Size of fonts that match page numbers
        page_number_font_size_counter: Counter[float] = Counter()

        # Size of fonts that match Element IDs (6-7 digit numbers)
        element_id_font_size_counter: Counter[float] = Counter()

        # Size of all other integers fonts
        font_size_counts: Counter[float] = Counter()

        # Track font sizes for specific patterns

        for page in pages:
            for block in page.blocks:
                if not isinstance(block, Text):
                    continue

                if block.font_name is not None:
                    font_name_counts[block.font_name] += 1

                if block.font_size is not None:
                    text_stripped = block.text.strip()

                    # Check if text matches part count pattern (\dx)
                    if part_count_pattern.match(text_stripped):
                        part_count_font_size_counter[block.font_size] += 1

                    elif text_stripped.isdigit():
                        text_num = int(text_stripped)
                        num_digits = len(text_stripped)

                        # Check if text matches Element ID (6-7 digit number)
                        if 6 <= num_digits <= 7:
                            element_id_font_size_counter[block.font_size] += 1
                        # Check if text matches page number (±1 from current)
                        elif abs(text_num - page.page_number) <= 1:
                            page_number_font_size_counter[block.font_size] += 1
                        else:
                            font_size_counts[block.font_size] += 1

        return cls(
            font_name_counts=font_name_counts,
            part_count_font_sizes=part_count_font_size_counter,
            page_number_font_sizes=page_number_font_size_counter,
            element_id_font_sizes=element_id_font_size_counter,
            remaining_font_sizes=font_size_counts,
        )
