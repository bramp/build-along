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
from dataclasses import dataclass

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    ProgressBar,
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

    indicator_block: Blocks | None = None
    """The block representing the progress indicator, if found."""

    indicator_progress: float | None = None
    """The calculated progress (0.0-1.0) based on indicator position."""

    def score(self) -> Weight:
        """Calculate final weighted score from components."""
        # Equal weighting for all components
        score = (self.position_score + self.width_score + self.aspect_ratio_score) / 3.0
        return score


@dataclass(frozen=True)
class ProgressBarClassifier(LabelClassifier):
    """Classifier for progress bars on instruction pages."""

    output = "progress_bar"
    requires = frozenset({"page_number"})

    def _score(self, result: ClassificationResult) -> None:
        """Score Drawing/Image elements and create candidates."""
        page_data = result.page_data
        page_bbox = page_data.bbox
        assert page_bbox is not None

        # Get page number location if available to help with positioning
        page_number_bbox = self._get_page_number_bbox(result)

        for block in page_data.blocks:
            # Only consider Drawing and Image elements
            if not isinstance(block, Drawing | Image):
                continue

            #  Score the block
            position_score = self._score_bottom_position(
                block.bbox, page_bbox, page_number_bbox
            )

            # Skip if not in bottom 20% of page
            if position_score == 0.0:
                continue

            width_score = self._score_width_coverage(block.bbox, page_bbox)
            aspect_ratio_score = self._score_aspect_ratio(block.bbox)

            # Must have minimum width (at least 30% of page width)
            if width_score == 0.0:
                continue

            # Must have aspect ratio suggesting horizontal bar (at least 3:1)
            if aspect_ratio_score == 0.0:
                continue

            # Clip the bbox to page boundaries
            original_width = block.bbox.width
            clipped_bbox = block.bbox.clip_to(page_bbox)

            # Find progress indicator within the bar's vertical range
            indicator_block, indicator_progress = self._find_progress_indicator(
                block, result, original_width
            )

            score_details = _ProgressBarScore(
                position_score=position_score,
                width_score=width_score,
                aspect_ratio_score=aspect_ratio_score,
                original_width=original_width,
                clipped_bbox=clipped_bbox,
                indicator_block=indicator_block,
                indicator_progress=indicator_progress,
            )

            combined = score_details.score()

            # Build source_blocks list including indicator if found
            source_blocks: list[Blocks] = [block]
            if indicator_block is not None:
                source_blocks.append(indicator_block)

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

        # Construct the ProgressBar element
        return ProgressBar(
            bbox=detail_score.clipped_bbox,
            progress=detail_score.indicator_progress,
            full_width=detail_score.original_width,
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
        self, bbox: BBox, page_bbox: BBox, page_number_bbox: BBox | None
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
        if bottom_margin_ratio > 0.2:
            return 0.0

        # Score based on proximity to bottom (closer = better)
        position_score = 1.0 - (bottom_margin_ratio / 0.2)

        # Boost score if near page number
        if page_number_bbox is not None:
            # Check horizontal distance to page number
            horizontal_distance = min(
                abs(bbox.x0 - page_number_bbox.x1),
                abs(bbox.x1 - page_number_bbox.x0),
            )
            if horizontal_distance < page_bbox.width * 0.3:
                position_score = min(1.0, position_score * 1.2)

        return min(1.0, position_score)

    def _score_width_coverage(self, bbox: BBox, page_bbox: BBox) -> float:
        """Score based on how much of the page width the element spans.

        Progress bars typically span >50% of the page width.
        """
        width_ratio = bbox.width / page_bbox.width

        # Penalize elements that are too narrow
        if width_ratio < 0.3:
            return 0.0

        # Score increases with width, maxing at 80% coverage
        # (some margin is expected on sides)
        if width_ratio >= 0.8:
            return 1.0

        # Linear interpolation between 0.3 and 0.8
        return (width_ratio - 0.3) / 0.5

    def _score_aspect_ratio(self, bbox: BBox) -> float:
        """Score based on aspect ratio (should be wide and thin).

        Progress bars are typically very wide relative to their height.
        """
        aspect_ratio = bbox.width / bbox.height if bbox.height > 0 else 0.0

        # Progress bars should be wide and thin
        # Typical aspect ratio might be 10:1 or higher
        if aspect_ratio < 3.0:  # Too square
            return 0.0

        if aspect_ratio >= 10.0:  # Good aspect ratio
            return 1.0

        # Linear interpolation between 3 and 10
        return (aspect_ratio - 3.0) / 7.0

    def _find_progress_indicator(
        self,
        bar_block: Drawing | Image,
        result: ClassificationResult,
        bar_full_width: float,
    ) -> tuple[Blocks | None, float | None]:
        """Find a progress indicator within the progress bar's vertical range.

        Progress bars often have a small visual indicator (a narrow drawing or
        image element) that shows how far through the instructions the reader is.
        This method searches for such an indicator within the bar's Y-range.

        The indicator must:
        - Be a Drawing or Image element
        - Be narrow (width < 20 pixels)
        - Be at least as tall as the progress bar (to avoid false positives)
        - Have its vertical center aligned with the bar

        Args:
            bar_block: The main progress bar drawing/image block
            result: The classification result containing all page blocks
            bar_full_width: The original unclipped width of the progress bar

        Returns:
            A tuple of (indicator_block, progress) where:
            - indicator_block: The block representing the indicator, or None
            - progress: The calculated progress (0.0-1.0), or None if not found
        """
        bar_bbox = bar_block.bbox
        bar_start_x = bar_bbox.x0
        bar_height = bar_bbox.height

        # Maximum width for an indicator (should be a narrow element)
        max_indicator_width = 20.0

        best_indicator: Blocks | None = None
        best_indicator_x: float | None = None

        for block in result.page_data.blocks:
            # Consider both Drawing and Image elements as potential indicators
            if not isinstance(block, Drawing | Image):
                continue

            # Skip the bar itself
            if block is bar_block:
                continue

            block_bbox = block.bbox

            # Check if block is narrow enough to be an indicator
            if block_bbox.width > max_indicator_width:
                continue

            # Indicator must be at least as tall as the bar to avoid false positives
            if block_bbox.height < bar_height:
                continue

            # Check if the block's center Y is aligned with the bar's center Y
            block_center_y = (block_bbox.y0 + block_bbox.y1) / 2
            bar_center_y = (bar_bbox.y0 + bar_bbox.y1) / 2
            if abs(block_center_y - bar_center_y) > bar_height:
                continue

            # This looks like an indicator - use the center X position
            indicator_x = (block_bbox.x0 + block_bbox.x1) / 2

            # Keep the indicator with the largest X (furthest progress)
            # This handles cases where there might be multiple small elements
            if best_indicator_x is None or indicator_x > best_indicator_x:
                best_indicator = block
                best_indicator_x = indicator_x

        if best_indicator is None or best_indicator_x is None:
            return None, None

        # Calculate progress as position relative to bar start, normalized by width
        # Clamp to 0.0-1.0 range
        progress = (best_indicator_x - bar_start_x) / bar_full_width
        progress = max(0.0, min(1.0, progress))

        log.debug(
            "Found progress indicator at x=%.1f, bar_start=%.1f, "
            "full_width=%.1f, progress=%.1%%",
            best_indicator_x,
            bar_start_x,
            bar_full_width,
            progress * 100,
        )

        return best_indicator, progress
