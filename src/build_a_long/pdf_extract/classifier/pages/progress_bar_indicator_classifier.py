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

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    ProgressBarIndicator,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Drawing,
    Image,
)

log = logging.getLogger(__name__)


class _ProgressBarIndicatorScore(Score):
    """Internal score representation for progress bar indicator classification."""

    shape_score: float
    """Score based on how square the element is (0.0-1.0, higher = more square)."""

    size_score: float
    """Score based on size being in the ideal range (0.0-1.0)."""

    position_score: float
    """Score based on position at bottom of page (0.0-1.0)."""

    def score(self) -> Weight:
        """Calculate final weighted score from components."""
        # Shape (squareness) is most important for identifying the circle
        return (
            self.shape_score * 0.5 + self.size_score * 0.2 + self.position_score * 0.3
        )


class ProgressBarIndicatorClassifier(LabelClassifier):
    """Classifier for progress bar indicators.

    This classifier identifies roughly square/circular elements at the bottom
    of pages that could be progress bar indicators. The actual indicator is
    selected and consumed by the ProgressBarClassifier when building the
    progress bar.
    """

    output = "progress_bar_indicator"
    requires = frozenset()

    def _score(self, result: ClassificationResult) -> None:
        """Score Drawing/Image elements and create candidates."""
        page_data = result.page_data
        page_bbox = page_data.bbox
        assert page_bbox is not None

        cfg = self.config.progress_bar_indicator

        for block in page_data.blocks:
            # Only consider Drawing and Image elements
            if not isinstance(block, Drawing | Image):
                continue

            bbox = block.bbox

            # Check size constraints
            if bbox.width < cfg.min_size or bbox.height < cfg.min_size:
                continue
            if bbox.width > cfg.max_size or bbox.height > cfg.max_size:
                continue

            # Check aspect ratio (should be roughly square for a circle)
            aspect_ratio = max(bbox.width, bbox.height) / max(
                min(bbox.width, bbox.height), 0.1
            )
            if aspect_ratio > cfg.max_aspect_ratio:
                continue

            # Score based on how square the shape is (1.0 = perfect square)
            shape_score = 1.0 - (aspect_ratio - 1.0) / (cfg.max_aspect_ratio - 1.0)

            # Score based on size - ideal size is around 12-15 pixels
            ideal_size = (cfg.min_size + cfg.max_size) / 2
            avg_size = (bbox.width + bbox.height) / 2
            size_deviation = abs(avg_size - ideal_size) / ideal_size
            size_score = max(0.0, 1.0 - size_deviation)

            # Score based on position at bottom of page
            page_height = page_bbox.height
            element_bottom = bbox.y1
            bottom_distance = page_bbox.y1 - element_bottom
            bottom_margin_ratio = bottom_distance / page_height

            # Should be in bottom portion of page
            if bottom_margin_ratio > cfg.max_bottom_margin_ratio:
                continue

            position_score = 1.0 - (bottom_margin_ratio / cfg.max_bottom_margin_ratio)

            score_details = _ProgressBarIndicatorScore(
                shape_score=shape_score,
                size_score=size_score,
                position_score=position_score,
            )

            combined = score_details.score()

            log.debug(
                "[progress_bar_indicator] Candidate: block id=%d bbox=%s "
                "width=%.1f height=%.1f aspect=%.2f shape=%.2f size=%.2f pos=%.2f",
                block.id,
                bbox,
                bbox.width,
                bbox.height,
                aspect_ratio,
                shape_score,
                size_score,
                position_score,
            )

            result.add_candidate(
                Candidate(
                    bbox=bbox,
                    label="progress_bar_indicator",
                    score=combined,
                    score_details=score_details,
                    source_blocks=[block],
                ),
            )

    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> ProgressBarIndicator:
        """Construct a ProgressBarIndicator element from a single candidate."""
        return ProgressBarIndicator(
            bbox=candidate.bbox,
        )
