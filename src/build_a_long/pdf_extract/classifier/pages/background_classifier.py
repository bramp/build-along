"""
Background classifier.

Purpose
-------
Identify the full-page background elements on LEGO instruction pages.
Backgrounds are large rectangles (typically gray) that cover most or all of
the page and form the visual backdrop for the instruction content.

Heuristic
---------
- Look for Drawing elements that cover most of the page area (>85%)
- Typically have a gray fill color
- Should be at or near page boundaries
- There should be only one background per page

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
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Background,
)
from build_a_long.pdf_extract.extractor.page_blocks import Drawing

log = logging.getLogger(__name__)


class BackgroundClassifier(RuleBasedClassifier):
    """Classifier for background elements on instruction pages."""

    output: ClassVar[str] = "background"
    requires: ClassVar[frozenset[str]] = frozenset()  # No dependencies

    @property
    def rules(self) -> list[Rule]:
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
            # TODO: Add a rule for fill color (typically gray)
        ]

    def build(self, candidate: Candidate, result: ClassificationResult) -> Background:
        """Construct a Background element from a single candidate."""
        # The source block should be the Drawing block that was classified
        drawing_block = next(
            (b for b in candidate.source_blocks if isinstance(b, Drawing)), None
        )
        assert drawing_block is not None
        assert isinstance(drawing_block, Drawing)

        # Extract fill color as RGB tuple
        fill_color: tuple[float, float, float] | None = None
        if drawing_block.fill_color is not None and len(drawing_block.fill_color) >= 3:
            fill_color = (
                drawing_block.fill_color[0],
                drawing_block.fill_color[1],
                drawing_block.fill_color[2],
            )

        return Background(
            bbox=candidate.bbox,
            fill_color=fill_color,
        )
