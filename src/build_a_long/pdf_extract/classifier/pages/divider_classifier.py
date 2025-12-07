"""
Divider classifier.

Purpose
-------
Identify visual divider lines that separate sections of a LEGO instruction page.
Dividers are thin lines (typically white strokes) that run vertically or
horizontally across a significant portion of the page (>40% of page
height/width).

Heuristic
---------
- Look for Drawing elements that are thin lines (width or height near 0)
- Must span at least 40% of the page dimension
- Typically have white stroke color (for separating instruction sections)
- Can be vertical (separating left/right columns) or horizontal

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
    Divider,
)
from build_a_long.pdf_extract.extractor.page_blocks import Drawing

log = logging.getLogger(__name__)


class _DividerScore(Score):
    """Internal score representation for divider classification."""

    length_score: float
    """Score based on how much of the page dimension the divider spans (0.0-1.0)."""

    # TODO reconsider the color scoring approach - as not all dividers are white
    color_score: float
    """Score based on stroke color (white = high score)."""

    orientation: Divider.Orientation
    """Whether this is a vertical or horizontal divider."""

    def score(self) -> Weight:
        """Calculate final weighted score from components."""
        # Length is most important, color is secondary
        # TODO make weights configurable
        return self.length_score * 0.7 + self.color_score * 0.3


class DividerClassifier(LabelClassifier):
    """Classifier for divider lines on instruction pages."""

    output = "divider"
    requires = frozenset()  # No dependencies

    def _score(self, result: ClassificationResult) -> None:
        """Score Drawing elements and create candidates for potential dividers."""
        page_data = result.page_data
        page_bbox = page_data.bbox
        assert page_bbox is not None

        config = self.config.divider
        min_length_ratio = config.min_length_ratio
        max_thickness = config.max_thickness

        page_width = page_bbox.width

        # TODO Perhaps remove the progress bar area from consideration?
        page_height = page_bbox.height

        for block in page_data.blocks:
            # Only consider Drawing elements
            if not isinstance(block, Drawing):
                continue

            bbox = block.bbox
            width = bbox.width
            height = bbox.height

            # Skip elements at page boundaries (within edge_margin of page edges)
            # These are typically page borders, not content dividers
            edge_margin = config.edge_margin
            at_left_edge = bbox.x0 <= page_bbox.x0 + edge_margin
            at_right_edge = bbox.x1 >= page_bbox.x1 - edge_margin
            at_top_edge = bbox.y0 <= page_bbox.y0 + edge_margin
            at_bottom_edge = bbox.y1 >= page_bbox.y1 - edge_margin

            # Check for vertical divider (thin width, tall height)
            if width <= max_thickness and height >= page_height * min_length_ratio:
                # Reject vertical dividers at left or right page edges
                if at_left_edge or at_right_edge:
                    log.debug(
                        "[divider] Rejected edge vertical at x=%.1f "
                        "(page edges: %.1f-%.1f)",
                        bbox.x0,
                        page_bbox.x0,
                        page_bbox.x1,
                    )
                    continue
                orientation = Divider.Orientation.VERTICAL
                length_ratio = height / page_height
            # Check for horizontal divider (thin height, wide width)
            elif height <= max_thickness and width >= page_width * min_length_ratio:
                # Reject horizontal dividers at top or bottom page edges
                if at_top_edge or at_bottom_edge:
                    log.debug(
                        "[divider] Rejected edge horizontal at y=%.1f "
                        "(page edges: %.1f-%.1f)",
                        bbox.y0,
                        page_bbox.y0,
                        page_bbox.y1,
                    )
                    continue
                orientation = Divider.Orientation.HORIZONTAL
                length_ratio = width / page_width
            else:
                # Not a divider shape
                continue

            # Score the length (how much of page it spans)
            # Score increases from min_length_ratio (0.5) to 1.0 (1.0)
            normalized = (length_ratio - min_length_ratio) / (1.0 - min_length_ratio)
            length_score = min(1.0, normalized * 0.5 + 0.5)

            # Score the color (prefer white/light strokes)
            color_score = self._score_stroke_color(block)

            score_details = _DividerScore(
                length_score=length_score,
                color_score=color_score,
                orientation=orientation,
            )

            combined = score_details.score()

            # Skip if below minimum score threshold
            if combined < config.min_score:
                log.debug(
                    "[divider] Rejected %s divider at %s: score=%.2f < min=%.2f",
                    orientation.value,
                    bbox,
                    combined,
                    config.min_score,
                )
                continue

            result.add_candidate(
                Candidate(
                    bbox=bbox,
                    label="divider",
                    score=combined,
                    score_details=score_details,
                    source_blocks=[block],
                ),
            )

            log.debug(
                "[divider] Candidate %s divider at %s: length=%.1f%%, "
                "color_score=%.2f, score=%.2f",
                orientation.value,
                bbox,
                length_ratio * 100,
                color_score,
                combined,
            )

    def build(self, candidate: Candidate, result: ClassificationResult) -> Divider:
        """Construct a Divider element from a single candidate."""
        detail_score = candidate.score_details
        assert isinstance(detail_score, _DividerScore)

        return Divider(
            bbox=candidate.bbox,
            orientation=detail_score.orientation,
        )

    def _score_stroke_color(self, block: Drawing) -> float:
        """Score a drawing block based on its stroke color.

        Dividers are typically white lines on LEGO instruction pages.

        Args:
            block: The Drawing block to analyze

        Returns:
            Score from 0.0 to 1.0 where 1.0 is white stroke
        """
        if block.stroke_color is not None:
            r, g, b = block.stroke_color
            # Check if it's white or very light (all channels > 0.9)
            if r > 0.9 and g > 0.9 and b > 0.9:
                return 1.0
            # Light gray is also acceptable
            if r > 0.7 and g > 0.7 and b > 0.7:
                return 0.7
            # Any other stroke color gets a lower score
            return 0.3

        # No stroke color - could still be a divider via fill
        if block.fill_color is not None:
            r, g, b = block.fill_color
            if r > 0.9 and g > 0.9 and b > 0.9:
                return 0.8
            if r > 0.7 and g > 0.7 and b > 0.7:
                return 0.5

        return 0.0
