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

from build_a_long.pdf_extract.classifier.block_filter import (
    find_image_shadow_effects,
)
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
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    ProgressBarIndicator,
)
from build_a_long.pdf_extract.extractor.page_blocks import BBox, Drawing, Image

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
            SizePreferenceScore(
                min_size=cfg.indicator_min_size,
                target_size=cfg.indicator_max_size,  # Target max - prefer largest
                max_size=cfg.indicator_max_size,
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

    # Note: We intentionally do NOT override _get_additional_source_blocks here.
    # Shadow blocks are consumed during build() to avoid conflicts between
    # indicator and bar over shared blocks.

    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> ProgressBarIndicator:
        """Construct a ProgressBarIndicator element from a single candidate.

        Consumes shadow blocks around the indicator during build.
        """
        cfg = self.config.progress_bar

        # Find and consume shadow blocks around the indicator
        primary_block = candidate.source_blocks[0]
        if isinstance(primary_block, (Drawing, Image)):
            all_blocks = result.page_data.blocks
            effects, _ = find_image_shadow_effects(
                primary_block, all_blocks, margin=cfg.indicator_shadow_margin
            )

            if effects:
                log.debug(
                    f"Progress bar indicator at {primary_block.bbox} consumed "
                    f"{len(effects)} shadow effect blocks."
                )
                candidate.source_blocks.extend(effects)
                candidate.bbox = BBox.union_all(
                    [b.bbox for b in candidate.source_blocks]
                )

        return ProgressBarIndicator(bbox=candidate.bbox)
