"""
Progress bar indicator classifier.

Purpose
-------
Identify the progress indicator element within a progress bar. This is a circular
graphic on top of the progress bar that shows how far through the instructions
the reader is.

Heuristic
---------
- Look for roughly square Drawing/Image elements (circular indicator)
- Size should be in the range of 8-25 pixels (typical circle size)
- Located at the bottom of the page (same area as progress bars)
- Aspect ratio close to 1.0 (square/circular shape)

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
    AspectRatioRule,
    InBottomBandFilter,
    IsInstanceFilter,
    Rule,
    SizeRangeRule,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    ProgressBarIndicator,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Drawing,
    Image,
)

log = logging.getLogger(__name__)


class ProgressBarIndicatorClassifier(RuleBasedClassifier):
    """Classifier for progress bar indicators.

    This classifier identifies roughly square/circular elements at the bottom
    of pages that could be progress bar indicators. The actual indicator is
    selected and consumed by the ProgressBarClassifier when building the
    progress bar.
    """

    output: ClassVar[str] = "progress_bar_indicator"
    requires: ClassVar[frozenset[str]] = frozenset()

    @property
    def rules(self) -> list[Rule]:
        cfg = self.config.progress_bar_indicator
        return [
            # Only consider Drawing and Image elements
            IsInstanceFilter((Drawing, Image)),
            # Should be in bottom portion of page
            InBottomBandFilter(
                threshold_ratio=cfg.max_bottom_margin_ratio,
                name="position_band",
            ),
            # Check size constraints
            SizeRangeRule(
                min_width=cfg.min_size,
                max_width=cfg.max_size,
                min_height=cfg.min_size,
                max_height=cfg.max_size,
                weight=0.2,
                required=True,
                name="size",
            ),
            # Check aspect ratio (should be roughly square for a circle)
            AspectRatioRule(
                min_ratio=1.0,  # Lower bound handled by max_aspect_ratio logic
                max_ratio=cfg.max_aspect_ratio,
                weight=0.5,
                required=True,
                name="aspect_ratio",
            ),
            # Ideally we'd have a position score too, but InBottomBandFilter handles the hard constraint
            # and position score was mostly about being *in* the band.
        ]

    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> ProgressBarIndicator:
        """Construct a ProgressBarIndicator element from a single candidate."""
        return ProgressBarIndicator(
            bbox=candidate.bbox,
        )
