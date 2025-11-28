"""Batch classification result class."""

from __future__ import annotations

import logging

from pydantic import BaseModel

from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.text import TextHistogram
from build_a_long.pdf_extract.extractor.lego_page_elements import Manual

logger = logging.getLogger(__name__)


class BatchClassificationResult(BaseModel):
    """Results from classifying multiple pages together.

    This class holds both the per-page classification results and the
    global text histogram computed across all pages.
    """

    results: list[ClassificationResult]
    """Per-page classification results, one for each input page"""

    histogram: TextHistogram
    """Global text histogram computed across all pages"""

    @property
    def manual(self) -> Manual:
        """Construct a Manual from the classification results.

        Returns:
            Manual containing all successfully classified pages, sorted by PDF page
        """
        pages = []
        for result in self.results:
            page = result.page
            if page:
                pages.append(page)
            else:
                logger.warning(
                    "No valid page for page %s", result.page_data.page_number
                )
        return Manual(pages=pages)
