"""
Open bag classifier.

Purpose
-------
Identify "Open Bag" elements on LEGO instruction pages. An OpenBag element
consists of an optional bag number (large text) surrounded by a cluster
of images forming a bag icon graphic. This typically appears at the
top-left of a page when a new numbered bag of pieces should be opened.

Some sets use an OpenBag graphic without a number, indicating that all
bags should be opened.

Heuristic
---------
1. Look for large circular drawings (the bag icon outline) - these are strong
   signals for OpenBag elements
2. Fall back to finding large, square-ish image clusters in the top-left area
3. Score each cluster based on size, aspect ratio, and position
4. Check if any BagNumber candidates are inside the cluster (bonus score)
5. Best-scoring cluster becomes the OpenBag candidate

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
    LoosePartSymbol,
    OpenBag,
    Part,
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


class _OpenBagScore(Score):
    """Internal score representation for open bag classification.

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


class OpenBagClassifier(LabelClassifier):
    """Classifier for open bag elements.

    Identifies bag icons by finding large circular outline drawings.
    The circle surrounds the entire bag graphic, and all blocks inside
    are claimed as part of the OpenBag element.
    """

    output = "open_bag"
    requires = frozenset({"bag_number", "part", "loose_part_symbol"})

    def _score(self, result: ClassificationResult) -> None:
        """Find circular drawings and score them as potential bag icons."""
        config = self.config
        page_data = result.page_data
        page_bbox = page_data.bbox

        # Get bag number candidates (used for scoring bonus)
        bag_number_candidates = result.get_scored_candidates("bag_number")

        # Find large circular drawings (bag icon outlines)
        circle_drawings = [
            block
            for block in page_data.blocks
            if isinstance(block, Drawing)
            and _is_circular_drawing(block, config.open_bag.min_circle_size)
        ]

        log.debug(
            "[open_bag] page=%s bag_number_candidates=%d circle_drawings=%d",
            page_data.page_number,
            len(bag_number_candidates),
            len(circle_drawings),
        )

        # Pre-compute block assignments for all circles
        # This ensures blocks are assigned to the correct circle when circles overlap
        block_assignments = self._assign_blocks_to_circles(
            circle_drawings, page_data.blocks, page_bbox
        )

        # Process each circle as a potential bag icon
        for circle in circle_drawings:
            score_details = self._score_circle(circle, bag_number_candidates, page_bbox)
            if score_details is None:
                continue

            combined = score_details.score()
            if combined < config.open_bag.min_score:
                log.debug(
                    "[open_bag] Skipping circle score=%.2f (below %.2f) bbox=%s",
                    combined,
                    config.open_bag.min_score,
                    circle.bbox,
                )
                continue

            # Get blocks assigned to this specific circle
            overlapping_blocks = block_assignments.get(id(circle), [])

            result.add_candidate(
                Candidate(
                    bbox=circle.bbox,
                    label="open_bag",
                    score=combined,
                    score_details=score_details,
                    source_blocks=overlapping_blocks,
                ),
            )

            log.debug(
                "[open_bag] candidate size=%.2f aspect=%.2f position=%.2f "
                "has_number=%s score=%.2f blocks=%d bbox=%s",
                score_details.size_score,
                score_details.aspect_score,
                score_details.position_score,
                score_details.has_bag_number,
                combined,
                len(overlapping_blocks),
                circle.bbox,
            )

    def _assign_blocks_to_circles(
        self,
        circles: list[Drawing],
        blocks: list[Blocks],
        page_bbox: BBox,
    ) -> dict[int, list[Blocks]]:
        """Assign blocks to circles based on spatial containment and draw order.

        When multiple circles overlap, a block is assigned to the circle with
        the closest higher draw_order (i.e., the circle drawn immediately after
        the block). This ensures blocks are claimed by the correct circle.

        Args:
            circles: List of circular drawings (potential bag icons).
            blocks: All blocks on the page.
            page_bbox: The page bounding box for size comparison.

        Returns:
            Dictionary mapping circle id() to list of blocks assigned to it.
        """
        # Only consider drawings and images
        drawing_image_blocks: list[Blocks] = [
            b for b in blocks if isinstance(b, Drawing | Image)
        ]

        # Filter out large blocks (likely backgrounds) - >50% of page size
        small_blocks = filter_by_max_area(
            drawing_image_blocks, max_ratio=0.5, reference_bbox=page_bbox
        )

        # Build a mapping of circle id to its expanded bbox and draw_order
        circle_info: list[tuple[int, BBox, int | None]] = []
        for circle in circles:
            # Expand circle bbox slightly to catch blocks on the edge
            expanded_bbox = circle.bbox.expand(2.0)
            circle_info.append((id(circle), expanded_bbox, circle.draw_order))

        # Initialize result dictionary
        result: dict[int, list[Blocks]] = {id(c): [] for c in circles}

        # Assign each block to the appropriate circle
        for block in small_blocks:
            block_draw_order = block.draw_order

            # Find all circles that contain this block and have higher draw_order
            containing_circles: list[tuple[int, int | None]] = []
            for circle_id, expanded_bbox, circle_draw_order in circle_info:
                # Check if block is spatially contained in circle
                if not expanded_bbox.contains(block.bbox):
                    continue

                # Block must be drawn before the circle (lower draw_order)
                if (
                    block_draw_order is not None
                    and circle_draw_order is not None
                    and block_draw_order > circle_draw_order
                ):
                    continue

                containing_circles.append((circle_id, circle_draw_order))

            if not containing_circles:
                continue

            # If block is contained by multiple circles, assign to the one
            # with the closest (smallest) draw_order that's still >= block's
            # This means the circle that was drawn immediately after the block
            if len(containing_circles) == 1:
                best_circle_id = containing_circles[0][0]
            else:
                # Sort by draw_order (None goes last)
                containing_circles.sort(
                    key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0)
                )
                best_circle_id = containing_circles[0][0]

            result[best_circle_id].append(block)

            if len(containing_circles) > 1:
                log.debug(
                    "[open_bag] Block %s (draw_order=%s) contained by %d circles, "
                    "assigned to circle with draw_order=%s",
                    block.bbox,
                    block_draw_order,
                    len(containing_circles),
                    containing_circles[0][1],
                )

        return result

    def _score_circle(
        self,
        circle: Drawing,
        bag_number_candidates: list[Candidate],
        page_bbox: BBox,
    ) -> _OpenBagScore | None:
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
                "[open_bag] Circle at %s rejected: y=%.1f > max_y=%.1f",
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
            "[open_bag] Circle score details: size=%.2f (avg=%.1f/ideal=%.1f) "
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

        return _OpenBagScore(
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

    def build(self, candidate: Candidate, result: ClassificationResult) -> OpenBag:
        """Construct an OpenBag element from a single candidate.

        Discovers and builds the bag number at build time by finding
        the best-scoring bag number candidate inside the cluster.

        If no bag number is found, looks for a Part candidate instead
        (some bag icons contain a part rather than a bag number).

        For OpenBags containing a Part, also looks for a loose part symbol
        cluster to the right of the circle (from LoosePartSymbolClassifier).
        """
        detail_score = candidate.score_details
        assert isinstance(detail_score, _OpenBagScore)

        # Find and construct bag number at build time
        bag_number = self._find_and_build_bag_number(detail_score.cluster_bbox, result)

        # If no bag number found, look for a part inside the circle
        part: Part | None = None
        loose_part_symbol: LoosePartSymbol | None = None
        if bag_number is None:
            part = self._find_and_build_part(detail_score.cluster_bbox, result)
            # If we have a part, look for a loose part symbol to the right
            if part is not None:
                loose_part_symbol = self._find_and_build_loose_part_symbol(
                    detail_score.cluster_bbox, result
                )

        # Filter out blocks that were consumed by children (bag_number, part, etc.)
        # Parent elements should NOT claim blocks that are owned by their children
        own_blocks = [
            b for b in candidate.source_blocks if not result.is_block_consumed(b)
        ]
        candidate.source_blocks = own_blocks

        # Compute bbox as union of source_blocks + children
        # This ensures the bbox matches source_blocks + children as required
        bbox = BBox.union_all([b.bbox for b in candidate.source_blocks])
        if bag_number:
            bbox = bbox.union(bag_number.bbox)
        if part:
            bbox = bbox.union(part.bbox)
        if loose_part_symbol:
            bbox = bbox.union(loose_part_symbol.bbox)

        return OpenBag(
            bbox=bbox,
            number=bag_number,
            part=part,
            loose_part_symbol=loose_part_symbol,
        )

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
        bag_number_candidates = result.get_scored_candidates("bag_number")
        contained = list(filter_contained(bag_number_candidates, cluster_bbox))
        best_candidate = find_best_scoring(contained)

        log.debug(
            "[open_bag] Build: looking for bag_number in %s, "
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
            "[open_bag] Built bag_number=%s at %s",
            bag_number_elem.value,
            bag_number_elem.bbox,
        )
        return bag_number_elem

    def _find_and_build_part(
        self, cluster_bbox: BBox, result: ClassificationResult
    ) -> Part | None:
        """Find and build a part inside the cluster.

        Some bag icons contain a part instead of a bag number. This method
        looks for part candidates inside the cluster bbox.

        Args:
            cluster_bbox: Bounding box of the cluster.
            result: Classification result for accessing candidates.

        Returns:
            Built Part element, or None if not found.
        """
        part_candidates = result.get_scored_candidates("part")
        contained = list(filter_contained(part_candidates, cluster_bbox))
        best_candidate = find_best_scoring(contained)

        log.debug(
            "[open_bag] Build: looking for part in %s, "
            "found %d candidates, %d contained, best=%s",
            cluster_bbox,
            len(part_candidates),
            len(contained),
            best_candidate.bbox if best_candidate else None,
        )

        if best_candidate is None:
            return None

        part_elem = result.build(best_candidate)
        assert isinstance(part_elem, Part)
        log.debug(
            "[open_bag] Built part at %s",
            part_elem.bbox,
        )
        return part_elem

    def _find_and_build_loose_part_symbol(
        self,
        cluster_bbox: BBox,
        result: ClassificationResult,
    ) -> LoosePartSymbol | None:
        """Find and build a loose part symbol from the LoosePartSymbolClassifier.

        Looks for loose part symbol candidates that are positioned to the right of
        the OpenBag circle.

        Args:
            cluster_bbox: Bounding box of the OpenBag circle.
            result: Classification result for accessing candidates.

        Returns:
            Built LoosePartSymbol element, or None if not found.
        """
        symbol_candidates = result.get_scored_candidates("loose_part_symbol")

        # Define search region: to the right of the circle
        search_x_min = cluster_bbox.x0 + cluster_bbox.width * 0.5
        search_x_max = cluster_bbox.x1 + cluster_bbox.width
        search_y_min = cluster_bbox.y0 - cluster_bbox.height * 0.2
        search_y_max = cluster_bbox.y0 + cluster_bbox.height * 0.8

        # Find symbol candidates in the search region
        matching: list[Candidate] = []
        for candidate in symbol_candidates:
            bbox = candidate.bbox
            # Check if symbol overlaps with search region
            if (
                bbox.x0 >= search_x_min
                and bbox.x1 <= search_x_max
                and bbox.y0 >= search_y_min
                and bbox.y1 <= search_y_max
            ):
                matching.append(candidate)

        log.debug(
            "[open_bag] Build: looking for loose_part_symbol in region "
            "x=[%.1f, %.1f] y=[%.1f, %.1f], found %d candidates, %d matching",
            search_x_min,
            search_x_max,
            search_y_min,
            search_y_max,
            len(symbol_candidates),
            len(matching),
        )

        if not matching:
            return None

        # Take the best-scoring matching symbol
        best_candidate = find_best_scoring(matching)
        if best_candidate is None:
            return None

        symbol_elem = result.build(best_candidate)
        assert isinstance(symbol_elem, LoosePartSymbol)
        log.debug(
            "[open_bag] Built loose_part_symbol at %s",
            symbol_elem.bbox,
        )
        return symbol_elem
