"""
Full-page background classifier.

Purpose
-------
Identify large Drawing elements that cover most of the page area and form
the visual backdrop for LEGO instruction content.

Heuristic
---------
- Drawing elements that cover >85% of the page area
- Should be at or near page boundaries (within edge_margin)
- Typically have a gray fill color

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
    CoverageRule,
    EdgeProximityRule,
    IsInstanceFilter,
    Rule,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import LegoPageElements
from build_a_long.pdf_extract.extractor.page_blocks import Drawing

log = logging.getLogger(__name__)


class FullPageBackgroundClassifier(RuleBasedClassifier):
    """Classifier for full-page background drawings.

    Identifies Drawing elements that cover most of the page and are at or near
    page boundaries. These form the visual backdrop for instruction content.
    """

    output: ClassVar[str] = "full_page_background"
    requires: ClassVar[frozenset[str]] = frozenset()  # No dependencies

    @property
    def rules(self) -> list[Rule]:
        """Rules for full-page background detection."""
        config = self.config.background
        return [
            IsInstanceFilter(Drawing),
            CoverageRule(
                min_ratio=config.min_coverage_ratio,
                weight=0.7,
                required=True,
                name="Coverage",
            ),
            EdgeProximityRule(
                threshold=config.edge_margin,
                weight=0.3,
                required=True,
                name="EdgeProximity",
            ),
        ]

    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        """Full-page backgrounds are intermediate - never built directly.

        The BackgroundClassifier will consume these candidates and build
        the final Background element.
        """
        raise NotImplementedError(
            "FullPageBackgroundClassifier candidates are consumed by "
            "BackgroundClassifier and should not be built directly."
        )
