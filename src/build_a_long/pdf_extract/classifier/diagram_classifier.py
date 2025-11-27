"""
Diagram classifier.

Purpose
-------
Identify diagram regions on instruction pages. Diagrams are the main visual
elements showing assembly instructions for each step. They are typically:
- Drawing or Image elements
- Located to the right of step numbers and parts lists
- Occupy a significant portion of the page area
- Distinct from parts list diagrams (which are smaller)

Heuristic
---------
- Look for Drawing/Image elements that are not already classified as parts
- Must be reasonably large (not tiny decorative elements)
- Should not be part of a parts list, progress bar, etc.
- Typically located in the main content area of the page

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
    Diagram,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Drawing,
    Image,
)

log = logging.getLogger(__name__)


@dataclass
class _DiagramScore:
    """Internal score representation for diagram classification."""

    area_score: float
    """Score based on the diagram area (0.0-1.0)."""

    position_score: float
    """Score based on position in main content area (0.0-1.0)."""

    def combined_score(self, config: ClassifierConfig) -> float:
        """Calculate final weighted score from components."""
        # Equal weighting for all components
        return (self.area_score + self.position_score) / 2.0


@dataclass(frozen=True)
class DiagramClassifier(LabelClassifier):
    """Classifier for diagram regions on instruction pages."""

    output = "diagram"
    requires = frozenset({"progress_bar"})

    def _score(self, result: ClassificationResult) -> None:
        """Score Drawing/Image elements and create candidates."""
        page_data = result.page_data
        page_bbox = page_data.bbox
        assert page_bbox is not None

        # TODO Maybe this should be a page hint, to represent the true content area
        # Get progress bar bbox to filter out overlapping elements
        progress_bar_bbox = self._get_progress_bar_bbox(result)

        for block in page_data.blocks:
            # Only consider Drawing and Image elements
            if not isinstance(block, Drawing | Image):
                continue

            # Skip if overlaps significantly with progress bar
            if progress_bar_bbox and block.bbox.overlaps(progress_bar_bbox):
                continue

            # Skip very large elements that span most of the page
            # (likely borders/backgrounds)
            area_ratio = block.bbox.area / page_bbox.area
            if area_ratio > 0.9:
                continue

            # Score the block
            # TODO Does the score or position actually matter?
            area_score = self._score_area(block.bbox, page_bbox)
            position_score = self._score_position(block.bbox, page_bbox)

            if area_score == 0.0:
                continue

            score_details = _DiagramScore(
                area_score=area_score,
                position_score=position_score,
            )

            combined = score_details.combined_score(self.config)

            result.add_candidate(
                Candidate(
                    bbox=block.bbox,
                    label="diagram",
                    score=combined,
                    score_details=score_details,
                    source_blocks=[block],
                ),
            )

    def build(self, candidate: Candidate, result: ClassificationResult) -> Diagram:
        """Construct a Diagram element from a single candidate."""
        # Diagram construction is trivial - just wrap the bbox
        return Diagram(bbox=candidate.bbox)

    def _get_progress_bar_bbox(self, result: ClassificationResult) -> BBox | None:
        """Get the bounding box of the progress bar if present.

        Returns:
            BBox of the progress bar, or None if not found.
        """
        progress_bar_candidates = result.get_scored_candidates(
            "progress_bar", valid_only=False, exclude_failed=True
        )

        # Return the first progress bar candidate's bbox
        if progress_bar_candidates:
            return progress_bar_candidates[0].bbox

        return None

    def _score_area(self, bbox: BBox, page_bbox: BBox) -> float:
        """Score based on the diagram area relative to page size.

        Step diagrams are typically substantial in size - larger than detail
        callouts or decorative elements. We filter out very small elements
        and prefer larger diagrams.
        """
        area_ratio = bbox.area / page_bbox.area

        # Too small (< 3% of page) - likely decorative, callouts, or noise
        # Step diagrams are usually more substantial
        if area_ratio < 0.03:
            return 0.0

        # Good size range (3-60% of page)
        # Prefer larger diagrams as they're more likely to be main step diagrams
        if area_ratio <= 0.6:
            # Linear scale: 0.5 at 3%, 1.0 at 15%+
            if area_ratio < 0.15:
                return 0.5 + (area_ratio - 0.03) / 0.12 * 0.5
            return 1.0

        # Very large (> 60%) - might span multiple steps, reduce score
        return max(0.0, 1.0 - (area_ratio - 0.6) / 0.4)

    def _score_position(self, bbox: BBox, page_bbox: BBox) -> float:
        """Score based on position within the page.

        Diagrams can appear anywhere in the main content area.
        We penalize only extreme edges to filter out margin elements.
        """
        # Calculate relative position
        x_center = (bbox.x0 + bbox.x1) / 2
        y_center = (bbox.y0 + bbox.y1) / 2

        x_ratio = x_center / page_bbox.width
        y_ratio = y_center / page_bbox.height

        # Accept diagrams almost anywhere (x: 0.05-0.95, y: 0.05-0.95)
        # Penalize only those very close to edges
        x_score = 1.0
        if x_ratio < 0.05:
            x_score = x_ratio / 0.05
        elif x_ratio > 0.95:
            x_score = max(0.0, 1.0 - (x_ratio - 0.95) / 0.05)

        y_score = 1.0
        if y_ratio < 0.05:
            y_score = y_ratio / 0.05
        elif y_ratio > 0.95:
            y_score = max(0.0, 1.0 - (y_ratio - 0.95) / 0.05)

        return (x_score + y_score) / 2.0
