"""PageHint represents classification hints for a single page."""

from __future__ import annotations

from pydantic import BaseModel

from build_a_long.pdf_extract.extractor.lego_page_elements import Page

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
