"""Text histogram for analyzing font properties across pages."""

from __future__ import annotations

import re
from collections import Counter

from pydantic import BaseModel, ConfigDict

from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.page_blocks import Text


class TextHistogram(BaseModel):
    """Global statistics about numeric text elements across all pages.

    This data structure collects font size and font name distributions
    to help guide classification decisions.
    """

    model_config = ConfigDict(frozen=True)

    font_name_counts: Counter[str]

    part_count_font_sizes: Counter[float]
    r"""Font sizes for text matching \dx pattern (e.g., '2x', '3x')"""

    step_number_font_sizes: Counter[float]
    """Font sizes for text matching step numbers (1-999, excluding page numbers)"""

    page_number_font_sizes: Counter[float]
    """Font sizes for text matching page numbers (±1 from current page)"""

    element_id_font_sizes: Counter[float]
    """Font sizes for text matching Element IDs (6-7 digit numbers)"""

    remaining_font_sizes: Counter[float]
    """Font sizes for all other integer text elements"""

    @classmethod
    def empty(cls) -> TextHistogram:
        """Create an empty TextHistogram with zero counts.

        Returns:
            A TextHistogram with all counters initialized to empty.
        """
        return TextHistogram(
            font_name_counts=Counter(),
            part_count_font_sizes=Counter(),
            step_number_font_sizes=Counter(),
            page_number_font_sizes=Counter(),
            element_id_font_sizes=Counter(),
            remaining_font_sizes=Counter(),
        )

    def update(self, other: TextHistogram) -> None:
        """Update this histogram by adding counts from another histogram.

        Args:
            other: Another TextHistogram whose counts will be added to this one.
        """
        self.font_name_counts.update(other.font_name_counts)
        self.part_count_font_sizes.update(other.part_count_font_sizes)
        self.step_number_font_sizes.update(other.step_number_font_sizes)
        self.page_number_font_sizes.update(other.page_number_font_sizes)
        self.element_id_font_sizes.update(other.element_id_font_sizes)
        self.remaining_font_sizes.update(other.remaining_font_sizes)

    @classmethod
    def from_pages(cls, pages: list[PageData]) -> TextHistogram:
        """Build a histogram from all text elements across all pages.

        Args:
            pages: List of PageData objects to analyze.

        Returns:
            A TextHistogram containing font size and name distributions.
        """
        histogram = TextHistogram.empty()

        # TODO Ensure this matches the part count classifier (used elsewhere)
        # Pattern for part counts like "2x", "3x", etc.
        part_count_pattern = re.compile(r"^\d+x$", re.IGNORECASE)

        for page in pages:
            for block in page.blocks:
                if not isinstance(block, Text):
                    continue

                if block.font_name is not None:
                    histogram.font_name_counts[block.font_name] += 1

                if block.font_size is not None:
                    text_stripped = block.text.strip()

                    # Check if text matches part count pattern (\dx)
                    if part_count_pattern.match(text_stripped):
                        histogram.part_count_font_sizes[block.font_size] += 1

                    elif text_stripped.isdigit():
                        text_num = int(text_stripped)
                        num_digits = len(text_stripped)

                        # Check if text matches Element ID (6-7 digit number)
                        if 6 <= num_digits <= 7:
                            histogram.element_id_font_sizes[block.font_size] += 1
                        # Check if text matches page number (±1 from current)
                        elif abs(text_num - page.page_number) <= 1:
                            histogram.page_number_font_sizes[block.font_size] += 1
                        # Check if text matches step number pattern (1-999)
                        elif 1 <= text_num <= 999:
                            histogram.step_number_font_sizes[block.font_size] += 1
                        else:
                            histogram.remaining_font_sizes[block.font_size] += 1

        return histogram
