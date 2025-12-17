"""
Progress bar classifier.

Purpose
-------
Identify progress bars at the bottom of instruction pages. Progress bars are
typically horizontal elements spanning most of the page width, located near
the page number at the bottom of the page.

Heuristic
---------
- Look for Drawing/Image elements near the bottom of the page
- Must span a significant portion of the page width (e.g., >50%)
- Should be relatively thin vertically (height << width)
- Located near the page number or bottom margin
- May consist of multiple adjacent elements forming a single visual bar

Debugging
---------
Enable with `LOG_LEVEL=DEBUG` for structured logs.
"""

from __future__ import annotations

import logging

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.config import ProgressBarConfig
from build_a_long.pdf_extract.classifier.rule_based_classifier import (
    RuleBasedClassifier,
    RuleScore,
)
from build_a_long.pdf_extract.classifier.rules import (
    BottomPositionScore,
    ContinuousAspectRatioScore,
    IsInstanceFilter,
    PageNumberProximityScore,
    Rule,
    WidthCoverageScore,
)
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    ProgressBar,
    ProgressBarIndicator,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Block,
    Blocks,
    Drawing,
    Image,
)

log = logging.getLogger(__name__)


class ProgressBarClassifier(RuleBasedClassifier):
    """Classifier for progress bars on instruction pages."""

    output = "progress_bar"
    requires = frozenset({"page_number", "progress_bar_indicator"})

    @property
    def min_score(self) -> float:
        return self.config.progress_bar.min_score

    @property
    def rules(self) -> list[Rule]:
        config: ProgressBarConfig = self.config.progress_bar
        return [
            IsInstanceFilter((Drawing, Image)),
            BottomPositionScore(
                max_bottom_margin_ratio=config.max_bottom_margin_ratio,
                weight=1.0,
                name="position_score",
            ),
            # TODO Do we need this rule? Being in the bottom band may be sufficent
            PageNumberProximityScore(
                proximity_ratio=config.max_page_number_proximity_ratio,
                weight=0.2,
                name="page_number_proximity_score",
            ),
            WidthCoverageScore(
                min_width_ratio=config.min_width_ratio,
                max_score_width_ratio=config.max_score_width_ratio,
                weight=1.0,
                name="width_score",
            ),
            ContinuousAspectRatioScore(
                min_ratio=config.min_aspect_ratio,
                ideal_ratio=config.ideal_aspect_ratio,
                weight=1.0,
                name="aspect_ratio_score",
            ),
        ]

    def _get_additional_source_blocks(
        self, block: Block, result: ClassificationResult
    ) -> list[Blocks]:
        """Find overlapping blocks to include in source_blocks."""
        if not isinstance(block, Drawing | Image):
            return []

        page_bbox = result.page_data.bbox
        assert page_bbox is not None
        clipped_bbox = block.bbox.clip_to(page_bbox)
        config: ProgressBarConfig = self.config.progress_bar

        return self._find_overlapping_blocks(block, clipped_bbox, result, config)

    def build(self, candidate: Candidate, result: ClassificationResult) -> ProgressBar:
        """Construct a ProgressBar element from a single candidate."""
        # Get score details
        detail_score = candidate.score_details
        assert isinstance(detail_score, RuleScore)

        # Get the config for ProgressBarClassifier
        config: ProgressBarConfig = self.config.progress_bar

        # Calculate properties from candidate bbox (which is unclipped block.bbox)
        page_bbox = result.page_data.bbox
        assert page_bbox is not None
        clipped_bbox = candidate.bbox.clip_to(page_bbox)
        original_width = candidate.bbox.width
        bar_start_x = candidate.bbox.x0

        # Find and build the indicator at build time
        indicator, progress = self._find_and_build_indicator(
            clipped_bbox,
            bar_start_x,
            original_width,
            result,
            config,
        )

        # Construct the ProgressBar element
        return ProgressBar(
            bbox=clipped_bbox,
            progress=progress,
            full_width=original_width,
            indicator=indicator,
        )

    def _find_and_build_indicator(
        self,
        bar_bbox: BBox,
        bar_start_x: float,
        bar_full_width: float,
        result: ClassificationResult,
        config: ProgressBarConfig,
    ) -> tuple[ProgressBarIndicator | None, float | None]:
        """Find and build a progress bar indicator for this progress bar.

        Looks for progress_bar_indicator candidates that are vertically aligned
        with the progress bar and selects the one furthest to the right (showing
        most progress).

        Args:
            bar_bbox: The clipped bounding box of the progress bar
            bar_start_x: The starting X position of the progress bar
            bar_full_width: The original unclipped width of the progress bar
            result: Classification result containing indicator candidates
            config: ProgressBarConfig instance
        Returns:
            A tuple of (indicator, progress) where:
            - indicator: The built ProgressBarIndicator, or None
            - progress: The calculated progress (0.0-1.0), or None if not found
        """
        # Get available indicator candidates
        indicator_candidates = result.get_scored_candidates(
            "progress_bar_indicator",
            valid_only=False,
            exclude_failed=True,
        )

        bar_height = bar_bbox.height
        bar_center_y = (bar_bbox.y0 + bar_bbox.y1) / 2

        best_candidate: Candidate | None = None
        best_score: float = -1.0

        for cand in indicator_candidates:
            # Skip if already built (consumed by another progress bar)
            if cand.constructed is not None:
                continue

            cand_bbox = cand.bbox

            # Indicator must be at least as tall as the bar to avoid false positives
            if cand_bbox.height < bar_height:
                continue

            # Check if the candidate's center Y is aligned with the bar's center Y
            cand_center_y = (cand_bbox.y0 + cand_bbox.y1) / 2
            if abs(cand_center_y - bar_center_y) > bar_height:
                continue

            # Must be horizontally within or near the progress bar
            indicator_x = (cand_bbox.x0 + cand_bbox.x1) / 2
            bar_end_x = bar_start_x + bar_full_width
            if (
                indicator_x < bar_start_x - config.indicator_search_margin
                or indicator_x > bar_end_x + config.indicator_search_margin
            ):
                continue

            # Keep the indicator with the highest score (most circular shape)
            if cand.score > best_score:
                best_candidate = cand
                best_score = cand.score

        if best_candidate is None:
            return None, None

        # Calculate progress based on indicator position
        best_indicator_x = (best_candidate.bbox.x0 + best_candidate.bbox.x1) / 2
        progress = (best_indicator_x - bar_start_x) / bar_full_width
        progress = max(0.0, min(1.0, progress))

        log.debug(
            "Found progress indicator candidate at x=%.1f, bar_start=%.1f, "
            "full_width=%.1f, progress=%.1%%",
            best_indicator_x,
            bar_start_x,
            bar_full_width,
            progress * 100,
        )

        # Build the indicator
        try:
            indicator_elem = result.build(best_candidate)
            assert isinstance(indicator_elem, ProgressBarIndicator)
            return indicator_elem, progress
        except Exception as e:
            log.debug(
                "[progress_bar] Failed to build indicator at %s: %s",
                best_candidate.bbox,
                e,
            )
            return None, None

    def _find_overlapping_blocks(
        self,
        bar_block: Drawing | Image,
        bar_bbox: BBox,
        result: ClassificationResult,
        config: ProgressBarConfig,
    ) -> list[Blocks]:
        """Find all Drawing/Image blocks that are contained within the progress bar.

        This captures all visual elements that are part of the progress bar
        visualization, including:
        - The colored progress section on the left
        - Inner/outer borders
        - Progress indicator elements
        - Any decorative elements within the bar area

        Only includes blocks that are fully or mostly contained within the
        progress bar's vertical extent to avoid capturing unrelated elements
        like page backgrounds or vertical dividers.

        Args:
            bar_block: The main progress bar drawing/image block
            bar_bbox: The clipped bounding box of the progress bar
            result: The classification result containing all page blocks
            config: ProgressBarConfig instance
        Returns:
            List of blocks that are contained within the progress bar area
        """
        overlapping: list[Blocks] = []

        # Expand the bbox slightly vertically to catch elements that extend
        # a bit beyond (like the indicator)
        expanded_bbox = bar_bbox.expand(config.overlap_expansion_margin)

        for block in result.page_data.blocks:
            # Only consider Drawing and Image elements
            if not isinstance(block, Drawing | Image):
                continue

            # Skip the bar itself
            if block is bar_block:
                continue

            block_bbox = block.bbox

            # Block must be mostly within the progress bar's vertical extent
            # This filters out full-page backgrounds and vertical dividers
            if block_bbox.y0 < expanded_bbox.y0 or block_bbox.y1 > expanded_bbox.y1:
                continue

            # Block must overlap horizontally with the progress bar
            if block_bbox.x1 < bar_bbox.x0 or block_bbox.x0 > bar_bbox.x1:
                continue

            overlapping.append(block)

        return overlapping
