"""Extract page type hints from preliminary classification analysis.

This module provides PageHints to help determine page types (INSTRUCTION, CATALOG, INFO)
during a pre-pass before full classification.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from pydantic import BaseModel

from build_a_long.pdf_extract.classifier.text_histogram import TextHistogram
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.lego_page_elements import Page

logger = logging.getLogger(__name__)

PageType = Page.PageType


class PageHint(BaseModel):
    """Hint about a single page's type based on preliminary analysis.

    Attributes:
        page_number: The page number being analyzed
        confidences: Confidence scores (0.0-1.0) for each PageType
        part_number_count: Number of part numbers detected (catalog indicator)
        part_count_count: Number of part counts detected
        step_number_count: Number of step numbers detected (instruction indicator)
    """

    page_number: int
    confidences: dict[PageType, float]
    part_number_count: int
    part_count_count: int
    step_number_count: int

    @property
    def page_type(self) -> PageType:
        """Get the most likely page type based on highest confidence.

        Returns:
            PageType with the highest confidence score
        """
        return max(self.confidences.items(), key=lambda x: x[1])[0]

    @property
    def confidence(self) -> float:
        """Get the confidence for the most likely page type.

        Returns:
            Highest confidence score
        """
        return max(self.confidences.values())

    @property
    def is_instruction(self) -> bool:
        """Check if this is likely an instruction page.

        Returns:
            True if INSTRUCTION confidence > 0.8
        """
        return self.confidences.get(PageType.INSTRUCTION, 0.0) > 0.8

    @property
    def is_catalog(self) -> bool:
        """Check if this is likely a catalog page.

        Returns:
            True if CATALOG confidence > 0.8
        """
        return self.confidences.get(PageType.CATALOG, 0.0) > 0.8

    @property
    def is_info(self) -> bool:
        """Check if this is likely an info page.

        Returns:
            True if INFO confidence > 0.8
        """
        return self.confidences.get(PageType.INFO, 0.0) > 0.8


# TODO We should expand PageHints to find step numbers on each page, and try and
# make appropriate runs of steps. This can help the StepClassifier later.
class PageHints(BaseModel):
    """Page type hints derived from preliminary analysis of all pages.

    This class analyzes pages in a pre-pass to determine their types:
    - CATALOG pages: Have many part numbers (element IDs), typically >3
    - INSTRUCTION pages: Have step numbers and part counts but few/no part numbers
    - INFO pages: Have minimal structured content

    This helps downstream classifiers make better decisions, especially:
    - PartsClassifier: Can skip catalog pages or use different logic
    - PageClassifier: Can use hints to determine category
    """

    hints: dict[int, PageHint]
    """Mapping from page number to page hint"""

    # Threshold for classifying a page as CATALOG based on part number count
    # Pages with more than this many element IDs are considered catalog pages
    CATALOG_ELEMENT_ID_THRESHOLD: ClassVar[int] = 3

    @classmethod
    def empty(cls) -> PageHints:
        """Create empty PageHints with no hints.

        Returns:
            PageHints with empty hints dict
        """
        return PageHints(hints={})

    @classmethod
    def from_pages(cls, pages: list[PageData]) -> PageHints:
        """Extract page type hints from multiple pages.

        This method performs a lightweight analysis to classify pages:
        1. Build text histogram for each page
        2. Count potential part numbers (element IDs)
        3. Count potential part counts
        4. Count potential step numbers
        5. Classify based on these counts

        Heuristics:
        - CATALOG: >=10 element IDs (part numbers)
        - INSTRUCTION: >=1 step number OR (part counts > 0 and element IDs < 10)
        - INFO: Everything else

        Args:
            pages: List of PageData objects to analyze

        Returns:
            PageHints with type hints for each page
        """
        if not pages:
            return cls.empty()

        hints: dict[int, PageHint] = {}

        for page in pages:
            page_histogram = TextHistogram.from_pages([page])

            # Count indicators
            part_number_count = sum(page_histogram.element_id_font_sizes.values())
            part_count_count = sum(page_histogram.part_count_font_sizes.values())
            # Step numbers are harder to detect in pre-pass, use heuristic
            # (second most common part_count size)
            step_number_count = 0
            if part_count_count > 0:
                top_sizes = page_histogram.part_count_font_sizes.most_common(3)
                if len(top_sizes) >= 2 and 1 <= top_sizes[1][1] <= 5:
                    # Second most common has count between 1-5, likely step numbers
                    step_number_count = top_sizes[1][1]

            # Classify page type based on indicators
            # Calculate confidence for each page type
            confidences: dict[PageType, float] = {}

            # CATALOG confidence: based on number of part numbers
            if part_number_count > cls.CATALOG_ELEMENT_ID_THRESHOLD:
                # Catalog indicator
                confidences[PageType.CATALOG] = min(
                    0.95, 0.6 + (part_number_count / 100)
                )
            elif part_number_count > 0:
                # Some part numbers, but not many
                confidences[PageType.CATALOG] = min(0.5, 0.2 + (part_number_count / 50))
            else:
                # No part numbers
                confidences[PageType.CATALOG] = 0.0

            # INSTRUCTION confidence: based on step numbers and part counts
            if step_number_count > 0 and part_count_count > 0:
                # Strong instruction indicator: has both steps and parts
                confidences[PageType.INSTRUCTION] = 0.9
            elif step_number_count > 0:
                # Has step numbers
                confidences[PageType.INSTRUCTION] = 0.8
            elif part_count_count > 5 and part_number_count < 10:
                # Has part counts but not many part numbers
                confidences[PageType.INSTRUCTION] = 0.7
            elif part_count_count > 0 and part_number_count == 0:
                # Has part counts and no part numbers
                confidences[PageType.INSTRUCTION] = 0.6
            else:
                # Weak instruction indicator
                confidences[PageType.INSTRUCTION] = 0.0

            # INFO confidence: based on lack of structured content
            if part_count_count == 0 and part_number_count == 0:
                # No structured content
                confidences[PageType.INFO] = 0.8
            elif part_count_count < 3 and part_number_count < 3:
                # Minimal structured content
                confidences[PageType.INFO] = 0.5
            else:
                # Has significant structured content
                confidences[PageType.INFO] = 0.0

            hints[page.page_number] = PageHint(
                page_number=page.page_number,
                confidences=confidences,
                part_number_count=part_number_count,
                part_count_count=part_count_count,
                step_number_count=step_number_count,
            )

            # Get the most likely page type for logging
            page_type = max(confidences.items(), key=lambda x: x[1])[0]
            confidence = confidences[page_type]

            logger.debug(
                f"Page {page.page_number}: {page_type.value} "
                f"(confidence={confidence:.2f}, "
                f"part_numbers={part_number_count}, "
                f"part_counts={part_count_count}, "
                f"step_numbers={step_number_count})"
            )

        # Log summary
        type_counts = {}
        for hint in hints.values():
            type_counts[hint.page_type] = type_counts.get(hint.page_type, 0) + 1

        logger.info(
            f"Page hints extracted: {len(hints)} pages - "
            + ", ".join(
                f"{t.value}={c}"
                for t, c in sorted(type_counts.items(), key=lambda x: x[0].value)
            )
        )

        return PageHints(hints=hints)

    def get_hint(self, page_number: int) -> PageHint | None:
        """Get hint for a specific page.

        Args:
            page_number: Page number to get hint for

        Returns:
            PageHint if available, None otherwise
        """
        return self.hints.get(page_number)

    def is_catalog_page(self, page_number: int) -> bool:
        """Check if page is likely a catalog page.

        Args:
            page_number: Page number to check

        Returns:
            True if page is classified as catalog with reasonable confidence
        """
        hint = self.get_hint(page_number)
        return (
            hint is not None
            and hint.page_type == PageType.CATALOG
            and hint.confidence >= 0.5
        )

    def is_instruction_page(self, page_number: int) -> bool:
        """Check if page is likely an instruction page.

        Args:
            page_number: Page number to check

        Returns:
            True if page is classified as instruction with reasonable confidence
        """
        hint = self.get_hint(page_number)
        return (
            hint is not None
            and hint.page_type == PageType.INSTRUCTION
            and hint.confidence >= 0.5
        )
