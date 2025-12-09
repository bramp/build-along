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
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    ProgressBar,
    ProgressBarIndicator,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Blocks,
    Drawing,
    Image,
)

log = logging.getLogger(__name__)


class _ProgressBarScore(Score):
    """Internal score representation for progress bar classification."""

    position_score: float
    """Score based on position at bottom of page (0.0-1.0)."""

    width_score: float
    """Score based on how much of the page width it spans (0.0-1.0)."""

    aspect_ratio_score: float
    """Score based on horizontal aspect ratio (wide and thin) (0.0-1.0)."""

    original_width: float
    """Original width before clipping to page boundaries."""

    clipped_bbox: BBox
    """Bounding box clipped to page boundaries."""

    bar_start_x: float
    """The starting X position of the progress bar."""

    def score(self) -> Weight:
        """Calculate final weighted score from components."""
        # Equal weighting for all components
        score = (self.position_score + self.width_score + self.aspect_ratio_score) / 3.0
        return score


class ProgressBarClassifier(LabelClassifier):
    """Classifier for progress bars on instruction pages."""

    output = "progress_bar"
    requires = frozenset({"page_number", "progress_bar_indicator"})

    def _score(self, result: ClassificationResult) -> None:
        """Score Drawing/Image elements and create candidates."""
        page_data = result.page_data
        page_bbox = page_data.bbox
        assert page_bbox is not None

        # Get the config for ProgressBarClassifier
        config: ProgressBarConfig = self.config.progress_bar

        # Get page number location if available to help with positioning
        page_number_bbox = self._get_page_number_bbox(result)

        for block in page_data.blocks:
            # Only consider Drawing and Image elements
            if not isinstance(block, Drawing | Image):
                continue

            #  Score the block
            position_score = self._score_bottom_position(
                block.bbox, page_bbox, page_number_bbox, config
            )

            # Skip if not in bottom 20% of page
            if position_score == 0.0:
                continue

            width_score = self._score_width_coverage(block.bbox, page_bbox, config)
            aspect_ratio_score = self._score_aspect_ratio(block.bbox, config)

            # Must have minimum width (at least 30% of page width)
            if width_score == 0.0:
                continue

            # Must have aspect ratio suggesting horizontal bar (at least 3:1)
            if aspect_ratio_score == 0.0:
                continue

            # Clip the bbox to page boundaries
            original_width = block.bbox.width
            clipped_bbox = block.bbox.clip_to(page_bbox)

            # Find all overlapping blocks within the progress bar area
            overlapping_blocks = self._find_overlapping_blocks(
                block, clipped_bbox, result, config
            )

            score_details = _ProgressBarScore(
                position_score=position_score,
                width_score=width_score,
                aspect_ratio_score=aspect_ratio_score,
                original_width=original_width,
                clipped_bbox=clipped_bbox,
                bar_start_x=block.bbox.x0,
            )

            combined = score_details.score()

            # Build source_blocks list including all overlapping blocks
            source_blocks: list[Blocks] = [block]
            source_blocks.extend(overlapping_blocks)

            # Store candidate
            result.add_candidate(
                Candidate(
                    bbox=clipped_bbox,
                    label="progress_bar",
                    score=combined,
                    score_details=score_details,
                    source_blocks=source_blocks,
                ),
            )

    def build(self, candidate: Candidate, result: ClassificationResult) -> ProgressBar:
        """Construct a ProgressBar element from a single candidate."""
        # Get score details
        detail_score = candidate.score_details
        assert isinstance(detail_score, _ProgressBarScore)

        # Get the config for ProgressBarClassifier
        config: ProgressBarConfig = self.config.progress_bar

        # Find and build the indicator at build time
        indicator, progress = self._find_and_build_indicator(
            detail_score.clipped_bbox,
            detail_score.bar_start_x,
            detail_score.original_width,
            result,
            config,
        )

        # Construct the ProgressBar element
        return ProgressBar(
            bbox=detail_score.clipped_bbox,
            progress=progress,
            full_width=detail_score.original_width,
            indicator=indicator,
        )

    def _get_page_number_bbox(self, result: ClassificationResult) -> BBox | None:
        """Get the bbox of the page number if it has been classified."""
        page_number_candidates = result.get_scored_candidates(
            "page_number", valid_only=False, exclude_failed=True
        )

        if page_number_candidates:
            # Assume the highest scoring candidate is the page number
            return page_number_candidates[0].bbox

        return None

    def _score_bottom_position(
        self,
        bbox: BBox,
        page_bbox: BBox,
        page_number_bbox: BBox | None,
        config: ProgressBarConfig,
    ) -> float:
        """Score based on position at bottom of page.

        Returns higher scores for elements at the bottom of the page,
        especially near the page number.
        """
        page_height = page_bbox.height
        element_bottom = bbox.y1

        # Calculate distance from bottom of page
        bottom_distance = page_bbox.y1 - element_bottom
        bottom_margin_ratio = bottom_distance / page_height

        # Should be in bottom 20% of page
        if bottom_margin_ratio > config.bottom_margin_threshold:
            return 0.0

        # Score based on proximity to bottom (closer = better)
        position_score = 1.0 - (bottom_margin_ratio / config.bottom_margin_threshold)

        # Boost score if near page number
        if page_number_bbox is not None:
            # Check horizontal distance to page number
            horizontal_distance = min(
                abs(bbox.x0 - page_number_bbox.x1),
                abs(bbox.x1 - page_number_bbox.x0),
            )
            if (
                horizontal_distance
                < page_bbox.width * config.page_number_proximity_threshold
            ):
                position_score = min(
                    1.0, position_score * config.page_number_proximity_boost
                )

        return min(1.0, position_score)

    def _score_width_coverage(
        self, bbox: BBox, page_bbox: BBox, config: ProgressBarConfig
    ) -> float:
        """Score based on how much of the page width the element spans.

        Progress bars typically span >50% of the page width.
        """
        width_ratio = bbox.width / page_bbox.width

        # Penalize elements that are too narrow
        if width_ratio < config.min_width_ratio:
            return 0.0

        # Score increases with width, maxing at 80% coverage
        # (some margin is expected on sides)
        if width_ratio >= config.max_score_width_ratio:
            return 1.0

        # Linear interpolation between 0.3 and 0.8
        return (width_ratio - config.min_width_ratio) / (
            config.max_score_width_ratio - config.min_width_ratio
        )

    def _score_aspect_ratio(self, bbox: BBox, config: ProgressBarConfig) -> float:
        """Score based on aspect ratio (should be wide and thin).

        Progress bars are typically very wide relative to their height.
        """
        aspect_ratio = bbox.width / bbox.height if bbox.height > 0 else 0.0

        # Progress bars should be wide and thin
        # Typical aspect ratio might be 10:1 or higher
        if aspect_ratio < config.min_aspect_ratio:  # Too square
            return 0.0

        if aspect_ratio >= config.ideal_aspect_ratio:  # Good aspect ratio
            return 1.0

        # Linear interpolation between 3 and 10
        return (aspect_ratio - config.min_aspect_ratio) / (
            config.ideal_aspect_ratio - config.min_aspect_ratio
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
        expanded_bbox = bar_bbox.expand(config.overlap_expansion)

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
            log.debug(
                "Found overlapping block for progress bar: %s at %s",
                type(block).__name__,
                block.bbox,
            )

        return overlapping
