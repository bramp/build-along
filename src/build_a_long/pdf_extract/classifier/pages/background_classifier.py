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

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Background,
)
from build_a_long.pdf_extract.extractor.page_blocks import Drawing

log = logging.getLogger(__name__)


class _BackgroundScore(Score):
    """Internal score representation for background classification."""

    coverage_score: float
    """Score based on how much of the page the background covers (0.0-1.0)."""

    position_score: float
    """Score based on how close the element is to page boundaries (0.0-1.0)."""

    fill_color: tuple[float, float, float] | None
    """The RGB fill color of the background, if any."""

    def score(self) -> Weight:
        """Calculate final weighted score from components."""
        # Coverage is most important, position is secondary
        return self.coverage_score * 0.7 + self.position_score * 0.3


class BackgroundClassifier(LabelClassifier):
    """Classifier for background elements on instruction pages."""

    output = "background"
    requires = frozenset()  # No dependencies

    def _score(self, result: ClassificationResult) -> None:
        """Score Drawing elements and create candidates for potential backgrounds."""
        page_data = result.page_data
        page_bbox = page_data.bbox
        assert page_bbox is not None

        config = self.config.background
        min_coverage_ratio = config.min_coverage_ratio
        edge_tolerance = config.edge_tolerance

        page_width = page_bbox.width
        page_height = page_bbox.height
        page_area = page_width * page_height

        for block in page_data.blocks:
            # Only consider Drawing elements
            if not isinstance(block, Drawing):
                continue

            bbox = block.bbox
            width = bbox.width
            height = bbox.height
            block_area = width * height

            # Skip very small elements
            if block_area < page_area * 0.1:
                continue

            # Calculate coverage ratio
            coverage_ratio = block_area / page_area

            # Skip if coverage is too low
            if coverage_ratio < min_coverage_ratio:
                continue

            # Score the coverage (how much of page it covers)
            # Score increases from min_coverage_ratio to 1.0
            coverage_normalized = (coverage_ratio - min_coverage_ratio) / (
                1.0 - min_coverage_ratio
            )
            coverage_score = min(1.0, coverage_normalized * 0.5 + 0.5)

            # Score the position (how close to page boundaries)
            # Calculate distance from each edge
            left_dist = abs(bbox.x0 - page_bbox.x0)
            right_dist = abs(bbox.x1 - page_bbox.x1)
            top_dist = abs(bbox.y0 - page_bbox.y0)
            bottom_dist = abs(bbox.y1 - page_bbox.y1)

            # Average distance from edges
            avg_edge_dist = (left_dist + right_dist + top_dist + bottom_dist) / 4

            # Position score is higher when closer to edges
            if avg_edge_dist <= edge_tolerance:
                position_score = 1.0
            else:
                # Decrease score as distance increases
                position_score = max(0.0, 1.0 - (avg_edge_dist - edge_tolerance) / 50.0)

            # Extract fill color as RGB tuple
            fill_color: tuple[float, float, float] | None = None
            if block.fill_color is not None and len(block.fill_color) >= 3:
                fill_color = (
                    block.fill_color[0],
                    block.fill_color[1],
                    block.fill_color[2],
                )

            score_details = _BackgroundScore(
                coverage_score=coverage_score,
                position_score=position_score,
                fill_color=fill_color,
            )

            combined = score_details.score()

            # Skip if below minimum score threshold
            if combined < config.min_score:
                log.debug(
                    "[background] Rejected block at %s: score=%.2f < min=%.2f",
                    bbox,
                    combined,
                    config.min_score,
                )
                continue

            result.add_candidate(
                Candidate(
                    bbox=bbox,
                    label="background",
                    score=combined,
                    score_details=score_details,
                    source_blocks=[block],
                ),
            )

            log.debug(
                "[background] Candidate at %s: coverage=%.1f%%, "
                "position_score=%.2f, score=%.2f",
                bbox,
                coverage_ratio * 100,
                position_score,
                combined,
            )

    def build(self, candidate: Candidate, result: ClassificationResult) -> Background:
        """Construct a Background element from a single candidate."""
        detail_score = candidate.score_details
        assert isinstance(detail_score, _BackgroundScore)

        return Background(
            bbox=candidate.bbox,
            fill_color=detail_score.fill_color,
        )
