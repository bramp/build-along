"""
Loose part symbol classifier.

Purpose
-------
Identify "loose part" symbols that appear next to OpenBag circles containing
a Part instead of a bag number. The symbol indicates that an extra part is
needed that's not found in the main bag. It is a small, square-ish cluster
of drawings that provides additional visual context for which part to find.

Heuristic
---------
1. Find clusters of drawings/images that are:
   - Located to the right of an OpenBag circle
   - Have a roughly square aspect ratio (0.7 to 1.4)
   - Contain multiple drawing elements (>=3)
2. Score based on proximity to OpenBag circles and aspect ratio

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
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    LoosePartSymbol,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Drawing,
    Image,
)

log = logging.getLogger(__name__)


class _LoosePartSymbolScore(Score):
    """Score for loose part symbol classification."""

    aspect_score: float
    """Score based on how square the cluster is (0.0-1.0)."""

    size_score: float
    """Score based on how close the size is to the ideal (~68.5 points)."""

    def score(self) -> Weight:
        """Calculate final weighted score from components."""
        return (self.aspect_score + self.size_score) / 2.0


class LoosePartSymbolClassifier(LabelClassifier):
    """Classifier for loose part symbols.

    Identifies symbol clusters that appear in the upper portion of pages.
    These are small, square-ish clusters of drawings that provide visual context.
    The OpenBagClassifier will claim matching symbols during its build phase.
    """

    output = "loose_part_symbol"
    requires = frozenset()  # No dependencies - runs early

    def _score(self, result: ClassificationResult) -> None:
        """Find symbol clusters that contain a circular drawing component.

        The symbol icon consists of a circle with a LEGO brick inside,
        plus a small symbol (like a plus sign) in the corner. We look for
        small circular drawings as anchors and build clusters around them.
        """
        page_data = result.page_data

        # Filter to small drawings/images (symbol elements are small)
        # Exclude large blocks that are likely backgrounds or OpenBag circles
        max_block_size = 80.0  # Individual blocks should be smaller than this
        small_blocks = [
            b
            for b in page_data.blocks
            if isinstance(b, Drawing | Image)
            and b.bbox.width < max_block_size
            and b.bbox.height < max_block_size
        ]

        if not small_blocks:
            log.debug("[loose_part_symbol] No small blocks found")
            return

        # Find small circular drawings as potential symbol anchors
        # These are smaller than the OpenBag circles (which are ~200px)
        circle_anchors = [
            b
            for b in small_blocks
            if isinstance(b, Drawing) and self._is_small_circle(b)
        ]

        log.debug(
            "[loose_part_symbol] page=%s small_blocks=%d circle_anchors=%d",
            page_data.page_number,
            len(small_blocks),
            len(circle_anchors),
        )

        # For each circle anchor, find nearby blocks to form a cluster
        for anchor in circle_anchors:
            # Find blocks near this anchor
            cluster_blocks = self._find_nearby_blocks(
                anchor, small_blocks, max_distance=30.0
            )

            if len(cluster_blocks) < 3:
                continue

            # Calculate combined bbox
            symbol_bbox = BBox.union_all([b.bbox for b in cluster_blocks])

            # Check aspect ratio - should be roughly square (0.6 to 1.6)
            aspect = symbol_bbox.width / symbol_bbox.height if symbol_bbox.height else 0
            if not (0.6 <= aspect <= 1.6):
                log.debug(
                    "[loose_part_symbol] Cluster rejected: aspect=%.2f bbox=%s",
                    aspect,
                    symbol_bbox,
                )
                continue

            # Check total size - use config values for ideal size and tolerance
            # Using average of width and height for size calculation
            config = self.config.loose_part_symbol
            ideal_size = config.ideal_size
            tolerance = config.size_tolerance
            min_size = ideal_size * (1.0 - tolerance)
            max_size = ideal_size * (1.0 + tolerance)
            avg_size = (symbol_bbox.width + symbol_bbox.height) / 2.0

            if not (min_size <= avg_size <= max_size):
                log.debug(
                    "[loose_part_symbol] Cluster rejected: "
                    "size=%.1f not in [%.1f, %.1f] bbox=%s",
                    avg_size,
                    min_size,
                    max_size,
                    symbol_bbox,
                )
                continue

            # Score based on aspect ratio (closer to 1.0 is better)
            aspect_score = 1.0 - abs(aspect - 1.0) * 0.5
            # Score based on size (closer to ideal_size is better)
            size_deviation = abs(avg_size - ideal_size) / ideal_size
            size_score = 1.0 - size_deviation

            score_details = _LoosePartSymbolScore(
                aspect_score=aspect_score,
                size_score=size_score,
            )

            result.add_candidate(
                Candidate(
                    bbox=symbol_bbox,
                    label="loose_part_symbol",
                    score=score_details.score(),
                    score_details=score_details,
                    source_blocks=list(cluster_blocks),
                ),
            )

            log.debug(
                "[loose_part_symbol] Found candidate: bbox=%s aspect=%.2f "
                "size=%.1f score=%.2f",
                symbol_bbox,
                aspect,
                avg_size,
                score_details.score(),
            )

    def _is_small_circle(self, drawing: Drawing, max_size: float = 80.0) -> bool:
        """Check if a drawing is a small circle (symbol circle component).

        Args:
            drawing: The drawing to check.
            max_size: Maximum width/height for the circle.

        Returns:
            True if the drawing appears to be a small circular shape.
        """
        if not drawing.items or len(drawing.items) < 4:
            return False

        # Check size - must be small (not an OpenBag circle)
        if drawing.bbox.width > max_size or drawing.bbox.height > max_size:
            return False

        # Must be at least some minimum size
        if drawing.bbox.width < 20 or drawing.bbox.height < 20:
            return False

        # Check aspect ratio - circles should be close to 1:1
        aspect = drawing.bbox.width / drawing.bbox.height if drawing.bbox.height else 0
        if not (0.8 <= aspect <= 1.25):
            return False

        # Count bezier curves ('c' type items) - circles have 4+ curves
        curve_count = sum(1 for item in drawing.items if item[0] == "c")
        return curve_count >= 4

    def _find_nearby_blocks(
        self,
        anchor: Drawing,
        blocks: list[Drawing | Image],
        max_distance: float,
    ) -> list[Drawing | Image]:
        """Find blocks that are near the anchor block.

        Args:
            anchor: The anchor block to search around.
            blocks: All candidate blocks.
            max_distance: Maximum distance from anchor to include.

        Returns:
            List of blocks near the anchor (including the anchor itself).
        """
        result: list[Drawing | Image] = [anchor]
        anchor_bbox = anchor.bbox

        for block in blocks:
            if block is anchor:
                continue

            # Check distance from anchor
            if self._bbox_distance(anchor_bbox, block.bbox) <= max_distance:
                result.append(block)

        return result

    def _bbox_distance(self, bbox1: BBox, bbox2: BBox) -> float:
        """Calculate minimum distance between two bboxes."""
        # Calculate gap between boxes (0 if overlapping)
        x_gap = max(0, max(bbox1.x0, bbox2.x0) - min(bbox1.x1, bbox2.x1))
        y_gap = max(0, max(bbox1.y0, bbox2.y0) - min(bbox1.y1, bbox2.y1))
        return (x_gap**2 + y_gap**2) ** 0.5

    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LoosePartSymbol:
        """Construct a LoosePartSymbol element from a candidate."""
        return LoosePartSymbol(bbox=candidate.bbox)  # type: ignore[return-value]
