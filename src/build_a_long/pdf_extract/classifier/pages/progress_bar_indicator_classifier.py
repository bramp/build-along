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
from collections.abc import Sequence
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
    SizePreferenceScore,
)
from build_a_long.pdf_extract.classifier.rules.scale import LinearScale
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    ProgressBarIndicator,
)
from build_a_long.pdf_extract.extractor.page_blocks import Blocks, Drawing, Image

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
    def effects_margin(self) -> float | None:
        """Use the configured shadow margin for finding indicator effects."""
        return self.config.progress_bar.indicator_shadow_margin

    @property
    def effects_block_types(self) -> tuple[type[Blocks], ...]:
        """Progress bar indicators are Drawing or Image blocks."""
        return (Drawing, Image)

    @property
    def rules(self) -> Sequence[Rule]:
        cfg = self.config.progress_bar
        return [
            # Only consider Drawing and Image elements
            IsInstanceFilter((Drawing, Image)),
            # Should be in bottom portion of page
            InBottomBandFilter(
                threshold_ratio=cfg.indicator_max_bottom_margin_ratio,
                name="position_band",
            ),
            # Check size constraints, prefer larger (outer circle over inner fill)
            # Triangular scale: 0.0 at min, 1.0 at max, 0.0 beyond max
            SizePreferenceScore(
                scale=LinearScale(
                    {
                        cfg.indicator_min_size: 0.0,
                        cfg.indicator_max_size: 1.0,
                        cfg.indicator_max_size * 1.5: 0.0,  # Drop to 0.0 for oversized
                    }
                ),
                weight=0.5,
                required=True,
                name="size",
            ),
            # Check aspect ratio (should be roughly square for a circle)
            AspectRatioRule(
                min_ratio=cfg.indicator_min_aspect_ratio,
                max_ratio=cfg.indicator_max_aspect_ratio,
                weight=0.5,
                required=True,
                name="aspect_ratio",
            ),
            # Ideally we'd have a position score too, but InBottomBandFilter handles
            # the hard constraint and position score was mostly about being *in*
            # the band.
        ]

    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> ProgressBarIndicator:
        """Construct a ProgressBarIndicator element from a single candidate.

        Shadow blocks around the indicator are automatically consumed via
        _get_additional_source_blocks in the scoring phase.
        """
        return ProgressBarIndicator(bbox=candidate.bbox)
