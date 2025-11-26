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

from build_a_long.pdf_extract.classifier.classification_result import (
    Candidate,
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    LegoPageElements,
    ProgressBar,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Drawing,
    Image,
)

log = logging.getLogger(__name__)


@dataclass
class _ProgressBarScore:
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

    def combined_score(self, config: ClassifierConfig) -> float:
        """Calculate final weighted score from components."""
        # Equal weighting for all components
        score = (self.position_score + self.width_score + self.aspect_ratio_score) / 3.0
        return score


@dataclass(frozen=True)
class ProgressBarClassifier(LabelClassifier):
    """Classifier for progress bars on instruction pages."""

    outputs = frozenset({"progress_bar"})
    requires = frozenset({"page_number"})

    def score(self, result: ClassificationResult) -> None:
        """Score Drawing/Image elements and create candidates WITHOUT construction."""
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

            score_details = _ProgressBarScore(
                position_score=position_score,
                width_score=width_score,
                aspect_ratio_score=aspect_ratio_score,
                original_width=original_width,
                clipped_bbox=clipped_bbox,
            )

            combined = score_details.combined_score(self.config)

            # Store candidate WITHOUT construction
            result.add_candidate(
                "progress_bar",
                Candidate(
                    bbox=clipped_bbox,
                    label="progress_bar",
                    score=combined,
                    score_details=score_details,
                    constructed=None,
                    source_blocks=[block],
                    failure_reason=None,
                ),
            )

    def construct(self, result: ClassificationResult) -> None:
        """Construct ProgressBar elements from candidates."""
        candidates = result.get_candidates("progress_bar")
        for candidate in candidates:
            try:
                elem = self.construct_candidate(candidate, result)
                candidate.constructed = elem
            except Exception as e:
                candidate.failure_reason = str(e)

    def construct_candidate(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        """Construct a ProgressBar element from a single candidate."""
        # Get score details
        detail_score = candidate.score_details
        assert isinstance(detail_score, _ProgressBarScore)

        # Construct the ProgressBar element
        return ProgressBar(
            bbox=detail_score.clipped_bbox,
            progress=None,
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
