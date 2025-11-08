"""Extract font size hints from text histogram for classification."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

from build_a_long.pdf_extract.classifier.text_histogram import (
    TextHistogram,
)
from build_a_long.pdf_extract.extractor import PageData

logger = logging.getLogger(__name__)


# Minimum number of occurrences required to confidently identify a font size
MIN_SAMPLES = 3


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
        """Extract font size hints from multiple pages with spatial awareness.

        This method uses a section-based approach to handle LEGO instruction booklets:
        - Instruction pages (first 2/3): Used for main part counts and step numbers
        - Catalog pages (last 1/3): Used for catalog-specific sizes (if present)

        This prevents catalog pages from overwhelming instruction page statistics
        due to their higher density of part counts and smaller font sizes.

        The method:
        1. Splits pages into instruction and catalog sections
        2. Analyzes each section separately
        3. Extracts main sizes from instruction pages
        4. Extracts catalog-specific sizes from catalog pages
        5. Validates that catalog sizes are smaller than instruction sizes
        6. Requires minimum sample counts for confidence

        Args:
            pages: List of PageData objects to analyze.

        Returns:
            FontSizeHints with identified sizes and remaining histogram.
        """
        if not pages:
            # Handle empty page list
            return cls(
                part_count_size=None,
                catalog_part_count_size=None,
                catalog_element_id_size=None,
                step_number_size=None,
                step_repeat_size=None,
                page_number_size=None,
                remaining_font_sizes=Counter(),
            )

        # Split pages into instruction section (first 2/3) and catalog
        # section (last 1/3). Catalog pages appear at the end of LEGO
        # instruction booklets
        split_point = len(pages) * 2 // 3
        instruction_pages = pages[:split_point] if split_point > 0 else pages
        catalog_pages = pages[split_point:] if split_point < len(pages) else []

        logger.debug(
            f"Analyzing {len(pages)} pages: "
            f"{len(instruction_pages)} instruction pages, "
            f"{len(catalog_pages)} catalog pages"
        )

        # Build separate histograms for each section
        instruction_histogram = TextHistogram.from_pages(instruction_pages)
        catalog_histogram = (
            TextHistogram.from_pages(catalog_pages) if catalog_pages else None
        )

        # Extract part_count_size from instruction pages (most reliable)
        # This is the main font size used for part counts like "2x", "3x" in build steps
        part_count_size = cls._extract_size_with_minimum(
            instruction_histogram.part_count_font_sizes, "part_count"
        )

        # Extract step_number_size from instruction pages
        # Look for the second most common part count size as it often
        # represents step numbers
        top_instruction_sizes = instruction_histogram.part_count_font_sizes.most_common(
            4
        )
        step_number_size = cls._extract_nth_size_with_minimum(
            top_instruction_sizes, 1, "step_number"
        )
        step_repeat_size = cls._extract_nth_size_with_minimum(
            top_instruction_sizes, 2, "step_repeat"
        )

        # Extract catalog-specific sizes from catalog section
        catalog_part_count_size = None
        catalog_element_id_size = None

        if catalog_histogram:
            # Get the most common part count size in catalog section
            candidate_catalog_size = cls._extract_size_with_minimum(
                catalog_histogram.part_count_font_sizes, "catalog_part_count"
            )

            # Validate that catalog sizes are typically smaller than instruction sizes
            # This prevents misidentification when catalog has different layout
            if candidate_catalog_size and part_count_size:
                if candidate_catalog_size <= part_count_size:
                    catalog_part_count_size = candidate_catalog_size
                else:
                    logger.warning(
                        f"Catalog part count size ({candidate_catalog_size}) is larger "
                        f"than instruction part count size ({part_count_size}). "
                        f"Ignoring catalog size as it may be misidentified."
                    )
            else:
                catalog_part_count_size = candidate_catalog_size

            # Extract element ID size from catalog
            catalog_element_id_size = cls._extract_size_with_minimum(
                catalog_histogram.element_id_font_sizes, "catalog_element_id"
            )

        # Extract page number size from full document
        # Use instruction pages primarily, fall back to catalog if needed
        page_number_size = cls._extract_size_with_minimum(
            instruction_histogram.page_number_font_sizes, "page_number"
        )
        if not page_number_size and catalog_histogram:
            page_number_size = cls._extract_size_with_minimum(
                catalog_histogram.page_number_font_sizes, "page_number"
            )

        # Build combined remaining font sizes from instruction pages
        # (catalog pages are less relevant for other classifications)
        remaining_font_sizes = instruction_histogram.remaining_font_sizes.copy()

        # Log the extracted hints for debugging
        part_count_samples = (
            instruction_histogram.part_count_font_sizes.get(part_count_size, 0)
            if part_count_size
            else 0
        )
        catalog_part_count_samples = (
            catalog_histogram.part_count_font_sizes.get(catalog_part_count_size, 0)
            if catalog_histogram and catalog_part_count_size
            else 0
        )

        logger.info(
            f"Font size hints extracted: "
            f"part_count={part_count_size} (n={part_count_samples}), "
            f"catalog_part_count={catalog_part_count_size} "
            f"(n={catalog_part_count_samples}), "
            f"step_number={step_number_size}, "
            f"step_repeat={step_repeat_size}, "
            f"catalog_element_id={catalog_element_id_size}, "
            f"page_number={page_number_size}"
        )

        return cls(
            part_count_size=part_count_size,
            catalog_part_count_size=catalog_part_count_size,
            catalog_element_id_size=catalog_element_id_size,
            step_number_size=step_number_size,
            step_repeat_size=step_repeat_size,
            page_number_size=page_number_size,
            remaining_font_sizes=remaining_font_sizes,
        )

    @staticmethod
    def _extract_size_with_minimum(counter: Counter[float], label: str) -> float | None:
        """Extract the most common size if it meets minimum sample threshold.

        Args:
            counter: Counter of font sizes.
            label: Label for logging purposes.

        Returns:
            Most common font size or None if insufficient samples.
        """
        if not counter:
            return None

        most_common = counter.most_common(1)
        if not most_common:
            return None

        size, count = most_common[0]

        if count < MIN_SAMPLES:
            logger.debug(
                f"Insufficient samples for {label}: "
                f"size={size}, count={count} (min={MIN_SAMPLES})"
            )
            return None

        return size

    @staticmethod
    def _extract_nth_size_with_minimum(
        top_sizes: list[tuple[float, int]], n: int, label: str
    ) -> float | None:
        """Extract the nth most common size if it meets minimum sample threshold.

        Args:
            top_sizes: List of (size, count) tuples from most_common().
            n: Zero-based index of the size to extract.
            label: Label for logging purposes.

        Returns:
            Nth most common font size or None if insufficient samples or out of range.
        """
        if len(top_sizes) <= n:
            return None

        size, count = top_sizes[n]

        if count < MIN_SAMPLES:
            logger.debug(
                f"Insufficient samples for {label}: "
                f"size={size}, count={count} (min={MIN_SAMPLES})"
            )
            return None

        return size
