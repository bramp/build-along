"""
Page edge classifier.

Purpose
-------
Identify elements (Drawing or Image) that are entirely contained within the
edge margin of the page. These are typically artifacts like borders, bleed
lines, or print marks that should be classified as part of the background.

Heuristic
---------
- Drawing or Image elements
- Entirely contained within edge_margin from any page edge
- Left edge: block.x1 <= page.x0 + margin
- Right edge: block.x0 >= page.x1 - margin
- Top edge: block.y1 <= page.y0 + margin
- Bottom edge: block.y0 >= page.y1 - margin

Debugging
---------
Enable with `LOG_LEVEL=DEBUG` for structured logs.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.rule_based_classifier import (
    RuleBasedClassifier,
)
from build_a_long.pdf_extract.classifier.rules import (
    IsInstanceFilter,
    PageEdgeFilter,
    Rule,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import LegoPageElements
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image

log = logging.getLogger(__name__)


class PageEdgeClassifier(RuleBasedClassifier):
    """Classifier for page-edge artifacts.

    Identifies Drawing or Image elements that are entirely contained within
    the edge margin of the page. These are typically borders, bleed lines,
    or print marks that should be treated as background.
    """

    output: ClassVar[str] = "page_edge"
    requires: ClassVar[frozenset[str]] = frozenset()  # No dependencies

    @property
    def rules(self) -> list[Rule]:
        """Rules for page-edge element detection."""
        config = self.config.background
        return [
            IsInstanceFilter((Drawing, Image)),
            PageEdgeFilter(
                margin=config.edge_margin,
                name="PageEdge",
            ),
        ]

    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        """Page-edge elements are intermediate - never built directly.

        The BackgroundClassifier will consume these candidates and build
        the final Background element.
        """
        raise NotImplementedError(
            "PageEdgeClassifier candidates are consumed by "
            "BackgroundClassifier and should not be built directly."
        )
