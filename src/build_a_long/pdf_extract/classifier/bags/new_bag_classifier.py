"""
New bag classifier.

Purpose
-------
Identify "New Bag" elements on LEGO instruction pages. A NewBag element
consists of an optional bag number (large text) surrounded by a cluster
of images forming a bag icon graphic. This typically appears at the
top-left of a page when a new numbered bag of pieces should be opened.

Some sets use a NewBag graphic without a number, indicating that all
bags should be opened.

Heuristic
---------
1. Look for large circular drawings (the bag icon outline) - these are strong
   signals for NewBag elements
2. Fall back to finding large, square-ish image clusters in the top-left area
3. Score each cluster based on size, aspect ratio, and position
4. Check if any BagNumber candidates are inside the cluster (bonus score)
5. Best-scoring cluster becomes the NewBag candidate

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
from build_a_long.pdf_extract.classifier.score import Score, Weight, find_best_scoring
from build_a_long.pdf_extract.extractor.bbox import (
    BBox,
    filter_by_max_area,
    filter_contained,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    BagNumber,
    NewBag,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Blocks,
    Drawing,
    Image,
)

log = logging.getLogger(__name__)


def _is_circular_drawing(drawing: Drawing, min_size: float = 100.0) -> bool:
    """Check if a drawing is a large circle (bag icon outline).

    Bag icons are surrounded by a circular outline made of bezier curves.
    These circles are typically 150-250 points in diameter.

    Args:
        drawing: The drawing to check.
        min_size: Minimum width/height to consider.

    Returns:
        True if the drawing appears to be a circular outline.
    """
    if not drawing.items or len(drawing.items) < 4:
        return False

    # Check size - must be large enough to be a bag icon outline
    if drawing.bbox.width < min_size or drawing.bbox.height < min_size:
        return False

    # Check aspect ratio - circles should be close to 1:1
    aspect = drawing.bbox.width / drawing.bbox.height if drawing.bbox.height else 0
    if not (0.85 <= aspect <= 1.18):
        return False

    # Count bezier curves ('c' type items) - circles have 4+ curves
    curve_count = sum(1 for item in drawing.items if item[0] == "c")
    return curve_count >= 4


class _NewBagScore(Score):
    """Internal score representation for new bag classification.

    Scores based on circle properties (size, aspect ratio, position).
    Bag number discovery is deferred to build time.
    """

    size_score: float
    """Score based on circle size (0.0-1.0)."""

    aspect_score: float
    """Score based on how circular the shape is (0.0-1.0)."""

    position_score: float
    """Score based on position on page (0.0-1.0)."""

    has_bag_number: bool
    """Whether a BagNumber candidate was found inside the circle."""

    cluster_bbox: BBox
    """Bounding box of the circle (used for bag number lookup)."""

    def score(self) -> Weight:
        """Calculate final weighted score from components."""
        # Base score from circle properties
        base_score = (self.size_score + self.aspect_score + self.position_score) / 3.0

        # Bonus for having a bag number (increases confidence significantly)
        if self.has_bag_number:
            base_score = min(1.0, base_score + 0.2)

        return base_score


class NewBagClassifier(LabelClassifier):
    """Classifier for new bag elements.

    Identifies bag icons by finding large circular outline drawings.
    The circle surrounds the entire bag graphic, and all blocks inside
    are claimed as part of the NewBag element.
    """

    output = "new_bag"
    requires = frozenset({"bag_number"})

    def _score(self, result: ClassificationResult) -> None:
        """Find circular drawings and score them as potential bag icons."""
        config = self.config
        page_data = result.page_data
        page_bbox = page_data.bbox

        # Get bag number candidates (used for scoring bonus)
        bag_number_candidates = result.get_scored_candidates(
            "bag_number", valid_only=False, exclude_failed=True
        )

        # Find large circular drawings (bag icon outlines)
        circle_drawings = [
            block
            for block in page_data.blocks
            if isinstance(block, Drawing)
            and _is_circular_drawing(block, config.new_bag.min_circle_size)
        ]

        log.debug(
            "[new_bag] page=%s bag_number_candidates=%d circle_drawings=%d",
            page_data.page_number,
            len(bag_number_candidates),
            len(circle_drawings),
        )

        # Process each circle as a potential bag icon
        for circle in circle_drawings:
            score_details = self._score_circle(circle, bag_number_candidates, page_bbox)
            if score_details is None:
                continue

            combined = score_details.score()
            if combined < config.new_bag.min_score:
                log.debug(
                    "[new_bag] Skipping circle score=%.2f (below %.2f) bbox=%s",
                    combined,
                    config.new_bag.min_score,
                    circle.bbox,
                )
                continue

            # Find all blocks that overlap with the circle
            # These are the images/drawings inside the bag icon
            overlapping_blocks = self._find_overlapping_blocks(
                circle.bbox, page_data.bbox, page_data.blocks
            )

            result.add_candidate(
                Candidate(
                    bbox=circle.bbox,
                    label="new_bag",
                    score=combined,
                    score_details=score_details,
                    source_blocks=overlapping_blocks,
                ),
            )

            log.debug(
                "[new_bag] candidate size=%.2f aspect=%.2f position=%.2f "
                "has_number=%s score=%.2f blocks=%d bbox=%s",
                score_details.size_score,
                score_details.aspect_score,
                score_details.position_score,
                score_details.has_bag_number,
                combined,
                len(overlapping_blocks),
                circle.bbox,
            )

    def _find_overlapping_blocks(
        self, circle_bbox: BBox, page_bbox: BBox, blocks: list[Blocks]
    ) -> list[Blocks]:
        """Find all blocks that overlap with the circle bbox.

        Filters out large blocks (>50% of page size) to avoid claiming
        full-page backgrounds.

        Args:
            circle_bbox: The bounding box of the circle.
            page_bbox: The page bounding box for size comparison.
            blocks: All blocks on the page.

        Returns:
            List of Drawing/Image blocks that overlap with the circle.
        """
        # Only include drawings and images
        drawing_image_blocks: list[Blocks] = [
            b for b in blocks if isinstance(b, Drawing | Image)
        ]

        # Filter out large blocks (likely backgrounds) - >50% of page size
        small_blocks = filter_by_max_area(
            drawing_image_blocks, max_ratio=0.5, reference_bbox=page_bbox
        )

        # Expand circle bbox slightly to catch blocks on the edge
        # (circle bezier curves can be slightly inside the nominal bbox)
        expanded_bbox = circle_bbox.expand(2.0)

        # Keep only blocks are within the expanded circle bbox
        return filter_contained(small_blocks, expanded_bbox)

    def _score_circle(
        self,
        circle: Drawing,
        bag_number_candidates: list[Candidate],
        page_bbox: BBox,
    ) -> _NewBagScore | None:
        """Score a circular drawing as a potential bag icon outline.

        Args:
            circle: The circular drawing.
            bag_number_candidates: All bag number candidates on the page.
            page_bbox: Page bounding box for position calculations.

        Returns:
            Score details, or None if the circle doesn't meet requirements.
        """
        bbox = circle.bbox

        # Check position - must be in top area (more lenient for circles)
        # Circles can appear anywhere in the top half of the page
        max_y = page_bbox.height * 0.6  # Allow top 60% of page
        if bbox.y0 > max_y:
            log.debug(
                "[new_bag] Circle at %s rejected: y=%.1f > max_y=%.1f",
                bbox,
                bbox.y0,
                max_y,
            )
            return None

        # Score size (ideal is around 200-240)
        ideal_size = 220.0
        avg_size = (bbox.width + bbox.height) / 2.0
        size_score = min(1.0, avg_size / ideal_size)

        # Score aspect ratio (circles should be very close to 1:1)
        aspect = bbox.width / bbox.height if bbox.height else 0
        aspect_score = 1.0 - abs(aspect - 1.0) * 2.0
        aspect_score = max(0.0, aspect_score)

        # Score position (prefer left side, top area)
        x_ratio = bbox.x0 / page_bbox.width if page_bbox.width else 1.0
        y_ratio = bbox.y0 / page_bbox.height if page_bbox.height else 1.0
        # Left side gets higher score
        position_score = 1.0 - (x_ratio * 0.7 + y_ratio * 0.3)

        # Check if bag number exists inside the circle
        has_bag_number = self._has_bag_number_in_cluster(bbox, bag_number_candidates)

        log.debug(
            "[new_bag] Circle score details: size=%.2f (avg=%.1f/ideal=%.1f) "
            "aspect=%.2f (ratio=%.2f) position=%.2f (x=%.2f y=%.2f) "
            "has_number=%s bbox=%s",
            size_score,
            avg_size,
            ideal_size,
            aspect_score,
            aspect,
            position_score,
            x_ratio,
            y_ratio,
            has_bag_number,
            bbox,
        )

        return _NewBagScore(
            size_score=size_score,
            aspect_score=aspect_score,
            position_score=position_score,
            has_bag_number=has_bag_number,
            cluster_bbox=bbox,
        )

    def _has_bag_number_in_cluster(
        self, cluster_bbox: BBox, bag_number_candidates: list[Candidate]
    ) -> bool:
        """Check if a bag number candidate exists inside the cluster bbox.

        Args:
            cluster_bbox: Bounding box of the cluster.
            bag_number_candidates: All bag number candidates on the page.

        Returns:
            True if a bag number candidate is inside the cluster.
        """
        return any(filter_contained(bag_number_candidates, cluster_bbox))

    def build(self, candidate: Candidate, result: ClassificationResult) -> NewBag:
        """Construct a NewBag element from a single candidate.

        Discovers and builds the bag number at build time by finding
        the best-scoring bag number candidate inside the cluster.
        """
        detail_score = candidate.score_details
        assert isinstance(detail_score, _NewBagScore)

        # Find and construct bag number at build time
        bag_number = self._find_and_build_bag_number(detail_score.cluster_bbox, result)

        return NewBag(bbox=detail_score.cluster_bbox, number=bag_number)

    def _find_and_build_bag_number(
        self, cluster_bbox: BBox, result: ClassificationResult
    ) -> BagNumber | None:
        """Find and build the bag number inside the cluster.

        Args:
            cluster_bbox: Bounding box of the cluster.
            result: Classification result for accessing candidates.

        Returns:
            Built BagNumber element, or None if not found.
        """
        bag_number_candidates = result.get_scored_candidates(
            "bag_number", valid_only=False, exclude_failed=True
        )
        contained = list(filter_contained(bag_number_candidates, cluster_bbox))
        best_candidate = find_best_scoring(contained)

        log.debug(
            "[new_bag] Build: looking for bag_number in %s, "
            "found %d candidates, %d contained, best=%s",
            cluster_bbox,
            len(bag_number_candidates),
            len(contained),
            best_candidate.bbox if best_candidate else None,
        )

        if best_candidate is None:
            return None

        bag_number_elem = result.build(best_candidate)
        assert isinstance(bag_number_elem, BagNumber)
        log.debug(
            "[new_bag] Built bag_number=%s at %s",
            bag_number_elem.value,
            bag_number_elem.bbox,
        )
        return bag_number_elem
