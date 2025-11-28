"""Extract font size hints from text histogram for classification."""

from __future__ import annotations

import logging
from collections import Counter

from pydantic import BaseModel

from build_a_long.pdf_extract.classifier.constants import (
    CATALOG_ELEMENT_ID_THRESHOLD,
)
from build_a_long.pdf_extract.classifier.text.text_histogram import TextHistogram
from build_a_long.pdf_extract.extractor import PageData

logger = logging.getLogger(__name__)


# Minimum number of occurrences required to confidently identify a font size
MIN_SAMPLES = 3


class FontSizeHints(BaseModel):
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

    remaining_font_sizes: dict[str, int]
    """Font size distribution after removing known sizes (float keys as strings)"""

    @classmethod
    def empty(cls) -> FontSizeHints:
        """Create an empty FontSizeHints with no hints.

        This is useful as a default when no pages are available for analysis.

        Returns:
            FontSizeHints with all sizes set to None
        """
        return FontSizeHints(
            part_count_size=None,
            catalog_part_count_size=None,
            catalog_element_id_size=None,
            step_number_size=None,
            step_repeat_size=None,
            page_number_size=None,
            remaining_font_sizes={},
        )

    # TODO add a default() method that returns a hint with reasonable values.

    @classmethod
    def from_pages(cls, pages: list[PageData]) -> FontSizeHints:
        """Extract font size hints from multiple pages.

        This method analyzes pages to distinguish between instruction pages and
        catalog pages based on the presence of catalog element IDs:
        - Pages with >3 catalog element IDs are considered catalog pages
        - Part counts on catalog pages become catalog_part_count_size
        - Part counts on instruction pages become part_count_size

        The method:
        1. Separates pages into instruction and catalog based on element ID count
        2. Analyzes each type separately
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
            return FontSizeHints.empty()

        # Build histograms page by page to classify instruction vs catalog pages
        # If a page has any element_id_font_sizes, it's a catalog page
        instruction_histogram = TextHistogram.empty()
        catalog_histogram = TextHistogram.empty()
        all_histogram = TextHistogram.empty()

        instruction_page_count = 0
        catalog_page_count = 0

        for page in pages:
            page_histogram = TextHistogram.from_pages([page])

            # Always add to all_histogram
            all_histogram.update(page_histogram)

            # If page has many element IDs, classify as catalog
            element_id_count = sum(page_histogram.element_id_font_sizes.values())
            if element_id_count > CATALOG_ELEMENT_ID_THRESHOLD:
                catalog_histogram.update(page_histogram)
                catalog_page_count += 1
            else:
                instruction_histogram.update(page_histogram)
                instruction_page_count += 1

        logger.debug(
            f"Analyzing {len(pages)} pages: "
            f"{instruction_page_count} instruction pages, "
            f"{catalog_page_count} catalog pages"
        )

        # Extract part_count_size from instruction pages (most reliable)
        # This is the main font size used for part counts like "2x", "3x" in build steps
        part_count_size = None
        step_number_size = None
        step_repeat_size = None

        if instruction_page_count > 0:
            part_count_size = cls._extract_size_with_minimum(
                instruction_histogram.part_count_font_sizes, "part_count"
            )

            # Extract step_number_size from instruction pages
            # Look for the second most common part count size as it often
            # represents step numbers
            top_instruction_sizes = (
                instruction_histogram.part_count_font_sizes.most_common(4)
            )
            step_number_size = cls._extract_nth_size_with_minimum(
                top_instruction_sizes, 1, "step_number"
            )
            step_repeat_size = cls._extract_nth_size_with_minimum(
                top_instruction_sizes, 2, "step_repeat"
            )

        # Extract catalog-specific sizes from catalog pages
        catalog_part_count_size = None
        catalog_element_id_size = None

        if catalog_page_count > 0:
            # Get the most common part count size in catalog pages
            candidate_catalog_size = cls._extract_size_with_minimum(
                catalog_histogram.part_count_font_sizes, "catalog_part_count"
            )

            # Validate that catalog sizes are typically smaller than instruction sizes
            # This prevents misidentification when catalog has different layout
            catalog_part_count_size = candidate_catalog_size
            if (
                candidate_catalog_size
                and part_count_size
                and candidate_catalog_size > part_count_size
            ):
                logger.warning(
                    f"Catalog part count size ({candidate_catalog_size}) is larger "
                    f"than instruction part count size ({part_count_size}). "
                    f"This may indicate misidentification."
                )

            # Extract element ID size from catalog
            catalog_element_id_size = cls._extract_size_with_minimum(
                catalog_histogram.element_id_font_sizes, "catalog_element_id"
            )

        # Extract page number size and remaining font sizes from all pages
        page_number_size = cls._extract_size_with_minimum(
            all_histogram.page_number_font_sizes, "page_number"
        )
        remaining_font_sizes = all_histogram.remaining_font_sizes.copy()

        # Log the extracted hints for debugging
        part_count_samples = (
            instruction_histogram.part_count_font_sizes.get(part_count_size, 0)
            if instruction_page_count > 0 and part_count_size
            else 0
        )
        catalog_part_count_samples = (
            catalog_histogram.part_count_font_sizes.get(catalog_part_count_size, 0)
            if catalog_page_count > 0 and catalog_part_count_size
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

        return FontSizeHints(
            part_count_size=part_count_size,
            catalog_part_count_size=catalog_part_count_size,
            catalog_element_id_size=catalog_element_id_size,
            step_number_size=step_number_size,
            step_repeat_size=step_repeat_size,
            page_number_size=page_number_size,
            remaining_font_sizes={str(k): v for k, v in remaining_font_sizes.items()},
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
