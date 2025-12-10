"""
Arrow classifier.

Purpose
-------
Identify arrows on LEGO instruction pages. Arrows typically:
- Point from a main assembly to a sub-step callout
- Indicate direction of motion or insertion
- Connect related elements visually

Heuristic
---------
1. Find Drawing blocks with triangular shapes (3-4 line items) - the arrowhead
2. Filter to small filled shapes (5-20px, filled color)
3. Calculate the tip (furthest point from centroid)
4. Calculate direction angle from centroid to tip
5. Search for an adjacent thin rectangle (the shaft) that connects to the arrowhead
6. Trace the shaft to find the tail point (far end from arrowhead)

Arrows consist of:
- Arrowhead: A small filled triangular shape (3-4 line items)
- Shaft: A thin filled rectangle adjacent to the arrowhead base

Debugging
---------
Enable with `LOG_LEVEL=DEBUG` for structured logs.
"""

from __future__ import annotations

import logging
import math
from typing import ClassVar

from pydantic import BaseModel

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import LabelClassifier
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.bbox import filter_overlapping
from build_a_long.pdf_extract.extractor.lego_page_elements import Arrow, ArrowHead
from build_a_long.pdf_extract.extractor.page_blocks import Blocks, Drawing

log = logging.getLogger(__name__)


class _ArrowHeadData(BaseModel):
    """Data for a single arrowhead within an arrow."""

    tip: tuple[float, float]
    """The tip point (x, y) - where the arrowhead points TO."""

    direction: float
    """Direction angle in degrees (0=right, 90=down, 180=left, -90=up)."""

    shape_score: float
    """Score based on shape being triangular (0.0-1.0)."""

    size_score: float
    """Score based on size being in expected range (0.0-1.0)."""

    block: Drawing
    """The Drawing block for this arrowhead."""

    shaft_block: Drawing | None = None
    """The shaft Drawing block, if detected."""

    tail: tuple[float, float] | None = None
    """The tail/origin point where the shaft starts. None if no shaft detected."""


class _ArrowScore(Score):
    """Score representation for an arrow (one or more arrowheads + optional shaft)."""

    heads: list[_ArrowHeadData]
    """Data for each arrowhead in this arrow."""

    tail: tuple[float, float] | None = None
    """The tail/origin point where the shaft starts. None if no shaft detected."""

    shaft_block: Drawing | None = None
    """The shaft Drawing block, if detected."""

    # Weights for score calculation
    shape_weight: float = 0.7
    size_weight: float = 0.3

    def score(self) -> Weight:
        """Return the average score of all arrowheads."""
        if not self.heads:
            return 0.0
        total = sum(
            h.shape_score * self.shape_weight + h.size_score * self.size_weight
            for h in self.heads
        )
        return total / len(self.heads)


class ArrowClassifier(LabelClassifier):
    """Classifier for arrow elements (arrowheads)."""

    output: ClassVar[str] = "arrow"
    requires: ClassVar[frozenset[str]] = frozenset()

    def _score(self, result: ClassificationResult) -> None:
        """Score Drawing blocks as potential arrowheads and group by shared tail."""
        page_data = result.page_data
        arrow_config = self.config.arrow

        all_drawings = [
            block for block in page_data.blocks if isinstance(block, Drawing)
        ]

        # Phase 1: Find all valid arrowheads
        arrowheads: list[_ArrowHeadData] = []
        for block in all_drawings:
            head = self._score_arrowhead(block, all_drawings)
            if head is None:
                continue

            head_score = (
                head.shape_score * arrow_config.shape_weight
                + head.size_score * arrow_config.size_weight
            )
            if head_score < arrow_config.min_score:
                log.debug(
                    "[arrow] Rejected at %s: score=%.2f < min_score=%.2f",
                    block.bbox,
                    head_score,
                    arrow_config.min_score,
                )
                continue

            arrowheads.append(head)

        # Phase 2: Group arrowheads by tail (rounded to group nearby tails)
        groups: dict[tuple[float, float] | None, list[_ArrowHeadData]] = {}
        for head in arrowheads:
            tail_key: tuple[float, float] | None = None
            if head.tail is not None:
                tail_key = (round(head.tail[0], 1), round(head.tail[1], 1))
            groups.setdefault(tail_key, []).append(head)

        # Phase 3: Create candidates
        for tail_key, heads in groups.items():
            if tail_key is None:
                # No shaft - each arrowhead is its own arrow
                for head in heads:
                    self._add_arrow_candidate(result, [head])
            else:
                # Shared tail - one arrow with multiple heads
                self._add_arrow_candidate(result, heads)

    def _add_arrow_candidate(
        self, result: ClassificationResult, heads: list[_ArrowHeadData]
    ) -> None:
        """Create and add an arrow candidate from arrowhead data."""
        # Collect source blocks (deduplicated by object identity)
        seen_ids: set[int] = set()
        source_blocks: list[Blocks] = []
        for head in heads:
            if id(head.block) not in seen_ids:
                seen_ids.add(id(head.block))
                source_blocks.append(head.block)
            if head.shaft_block is not None and id(head.shaft_block) not in seen_ids:
                seen_ids.add(id(head.shaft_block))
                source_blocks.append(head.shaft_block)

        # Compute combined bbox
        arrow_bbox = source_blocks[0].bbox
        for block in source_blocks[1:]:
            arrow_bbox = arrow_bbox.union(block.bbox)

        # Get shared tail and shaft (if any)
        tail = next((h.tail for h in heads if h.tail), None)
        shaft_block = next((h.shaft_block for h in heads if h.shaft_block), None)

        arrow_score = _ArrowScore(
            heads=heads,
            tail=tail,
            shaft_block=shaft_block,
            shape_weight=self.config.arrow.shape_weight,
            size_weight=self.config.arrow.size_weight,
        )

        result.add_candidate(
            Candidate(
                bbox=arrow_bbox,
                label="arrow",
                score=arrow_score.score(),
                score_details=arrow_score,
                source_blocks=source_blocks,
            )
        )

        if len(heads) == 1:
            log.debug(
                "[arrow] Candidate at %s: score=%.2f, direction=%.0f°",
                arrow_bbox,
                arrow_score.score(),
                heads[0].direction,
            )
        else:
            log.debug(
                "[arrow] Candidate (multi-head) at %s: score=%.2f, heads=%d",
                arrow_bbox,
                arrow_score.score(),
                len(heads),
            )

    def _score_arrowhead(
        self, block: Drawing, all_drawings: list[Drawing]
    ) -> _ArrowHeadData | None:
        """Score a Drawing block as a potential arrowhead.

        Args:
            block: The Drawing block to score
            all_drawings: All Drawing blocks on the page (for shaft searching)

        Returns:
            ArrowHeadData if this is a valid arrowhead, None otherwise.
            Includes shaft_block and tail if a shaft was found.
        """
        arrow_config = self.config.arrow
        bbox = block.bbox
        items = block.items

        # Must have items
        if not items:
            return None

        # Must be filled (arrowheads are filled shapes)
        if not block.fill_color:
            return None

        # Check size constraints
        if bbox.width < arrow_config.min_size or bbox.width > arrow_config.max_size:
            return None
        if bbox.height < arrow_config.min_size or bbox.height > arrow_config.max_size:
            return None

        # Check aspect ratio (triangles are roughly square-ish to elongated)
        aspect = bbox.width / bbox.height if bbox.height > 0 else 0
        if (
            aspect < arrow_config.min_aspect_ratio
            or aspect > arrow_config.max_aspect_ratio
        ):
            return None

        # Must have 3-5 line items forming the shape
        line_items = [item for item in items if item[0] == "l"]
        if len(line_items) < 3 or len(line_items) > 5:
            return None

        # All items should be lines (no curves, rectangles, etc.)
        if len(line_items) != len(items):
            return None

        # Extract unique points from line items
        points = self._extract_unique_points(line_items)
        if len(points) < 3 or len(points) > 5:
            return None

        # Calculate centroid
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)

        # Find the tip (point furthest from centroid)
        max_dist = 0.0
        tip = points[0]
        for p in points:
            dist = math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2)
            if dist > max_dist:
                max_dist = dist
                tip = p

        # Calculate direction from centroid to tip
        direction = math.degrees(math.atan2(tip[1] - cy, tip[0] - cx))

        # Score the shape (more points closer to triangle = better)
        # Ideal triangle has 3-4 points
        if len(points) == 3:
            shape_score = 1.0
        elif len(points) == 4:
            shape_score = 0.9
        else:
            shape_score = 0.7

        # Score the size (prefer sizes closer to ideal)
        ideal_size = arrow_config.ideal_size
        size_diff = abs(bbox.width - ideal_size) + abs(bbox.height - ideal_size)
        size_score = max(0.0, 1.0 - (size_diff / (ideal_size * 2)))

        # Try to find a shaft for this arrowhead
        shaft_block: Drawing | None = None
        tail: tuple[float, float] | None = None
        shaft_result = self._find_shaft(block, direction, tip, all_drawings)
        if shaft_result is not None:
            shaft_block, tail = shaft_result
            log.debug(
                "[arrow] Found shaft for arrowhead at %s, tail at %s",
                block.bbox,
                tail,
            )

        return _ArrowHeadData(
            tip=tip,
            direction=direction,
            shape_score=shape_score,
            size_score=size_score,
            block=block,
            shaft_block=shaft_block,
            tail=tail,
        )

    def _extract_unique_points(
        self, line_items: list[tuple]
    ) -> list[tuple[float, float]]:
        """Extract unique points from line items.

        Args:
            line_items: List of line items, each ('l', (x1, y1), (x2, y2))
                where points are tuples (converted from PyMuPDF Point objects)

        Returns:
            List of unique (x, y) points, rounded to 1 decimal place
        """
        points: list[tuple[float, float]] = []
        seen: set[tuple[float, float]] = set()

        for item in line_items:
            # item is ('l', (x1, y1), (x2, y2)) - tuples not Point objects
            p1, p2 = item[1], item[2]
            for p in [p1, p2]:
                # p is a tuple (x, y)
                key = (round(p[0], 1), round(p[1], 1))
                if key not in seen:
                    seen.add(key)
                    points.append((p[0], p[1]))

        return points

    def _find_shaft(
        self,
        arrowhead: Drawing,
        direction: float,
        tip: tuple[float, float],
        all_drawings: list[Drawing],
    ) -> tuple[Drawing, tuple[float, float]] | None:
        """Find the shaft connected to an arrowhead.

        The shaft can be:
        1. A thin rectangle (single "re" item) - simple straight shaft
        2. A path with multiple "l" (line) items and/or "re" items - L-shaped shaft

        The shaft must:
        - Be positioned adjacent to the arrowhead base (opposite the tip)
        - Have the same fill color as the arrowhead (within tolerance)

        Args:
            arrowhead: The arrowhead Drawing block
            direction: Direction angle in degrees (0=right, 90=down, etc.)
            tip: The tip point (x, y) of the arrowhead
            all_drawings: All Drawing blocks on the page to search

        Returns:
            Tuple of (shaft Drawing block, tail point) if found, None otherwise
        """
        # Try to find a simple rectangular shaft first
        # Optimization: Filter to drawings near the arrowhead
        # Shaft must connect to arrowhead, so it must overlap a slightly expanded bbox
        search_bbox = arrowhead.bbox.expand(20.0)  # generous margin
        nearby_drawings = filter_overlapping(all_drawings, search_bbox)

        result = self._find_simple_shaft(arrowhead, direction, tip, nearby_drawings)
        if result is not None:
            return result

        # Try to find an L-shaped (cornered) shaft
        return self._find_cornered_shaft(arrowhead, direction, nearby_drawings)

    def _find_simple_shaft(
        self,
        arrowhead: Drawing,
        direction: float,
        tip: tuple[float, float],
        all_drawings: list[Drawing],
    ) -> tuple[Drawing, tuple[float, float]] | None:
        """Find a simple rectangular shaft connected to an arrowhead.

        The shaft is a thin rectangle that:
        - Has a single "re" (rectangle) item
        - Is very thin (typically 1-3 pixels in one dimension)
        - Is positioned adjacent to the arrowhead base (opposite the tip)
        - Has the same fill color as the arrowhead

        Args:
            arrowhead: The arrowhead Drawing block
            direction: Direction angle in degrees (0=right, 90=down, etc.)
            tip: The tip point (x, y) of the arrowhead
            all_drawings: All Drawing blocks on the page to search

        Returns:
            Tuple of (shaft Drawing block, tail point) if found, None otherwise
        """
        direction_rad = math.radians(direction)

        # The arrowhead's base is opposite the tip
        # Calculate expected shaft direction (opposite to tip direction)
        shaft_dx = -math.cos(direction_rad)
        shaft_dy = -math.sin(direction_rad)

        # Maximum distance from arrowhead to search for shaft
        max_shaft_gap = 5.0  # pixels
        # Maximum thickness for a shaft
        max_shaft_thickness = 5.0  # pixels
        # Minimum length for a shaft
        min_shaft_length = 10.0  # pixels

        best_shaft: Drawing | None = None
        best_tail: tuple[float, float] | None = None
        best_distance = float("inf")

        for drawing in all_drawings:
            if drawing is arrowhead:
                continue

            # Must be filled
            if not drawing.fill_color:
                continue

            # Should have same fill color as arrowhead (within tolerance)
            if arrowhead.fill_color and not self._colors_match(
                arrowhead.fill_color, drawing.fill_color
            ):
                continue

            # Must have a single rectangle item
            items = drawing.items
            if not items or len(items) != 1:
                continue
            if items[0][0] != "re":
                continue

            bbox = drawing.bbox

            # Determine if this is a horizontal or vertical shaft
            is_horizontal = bbox.width > bbox.height
            thickness = bbox.height if is_horizontal else bbox.width
            length = bbox.width if is_horizontal else bbox.height

            # Check shaft dimensions
            if thickness > max_shaft_thickness:
                continue
            if length < min_shaft_length:
                continue

            # Check if the shaft is positioned appropriately relative to arrowhead
            # The shaft should be in the direction opposite to the tip
            shaft_center_x = (bbox.x0 + bbox.x1) / 2
            shaft_center_y = (bbox.y0 + bbox.y1) / 2

            # Vector from tip to shaft center
            to_shaft_x = shaft_center_x - tip[0]
            to_shaft_y = shaft_center_y - tip[1]

            # Distance from tip to shaft center
            distance = math.sqrt(to_shaft_x**2 + to_shaft_y**2)

            # Check if shaft is roughly in the expected direction (opposite tip)
            if distance > 0:
                # Normalize the vector
                norm_to_shaft_x = to_shaft_x / distance
                norm_to_shaft_y = to_shaft_y / distance

                # Dot product with expected shaft direction
                # (should be positive if in same direction)
                dot = norm_to_shaft_x * shaft_dx + norm_to_shaft_y * shaft_dy
                if dot < 0.5:  # Not pointing in expected direction
                    continue

            # Check if shaft connects to arrowhead (adjacent or slightly overlapping)
            if is_horizontal:
                # Horizontal shaft - check vertical alignment with arrowhead center
                arrowhead_center_y = (arrowhead.bbox.y0 + arrowhead.bbox.y1) / 2
                if abs(shaft_center_y - arrowhead_center_y) > max_shaft_gap + thickness:
                    continue

                # Check horizontal proximity to arrowhead
                if direction > 90 or direction < -90:
                    # Arrow points left, shaft should be to the right
                    gap = bbox.x0 - arrowhead.bbox.x1
                else:
                    # Arrow points right, shaft should be to the left
                    gap = arrowhead.bbox.x0 - bbox.x1

                if gap > max_shaft_gap or gap < -arrowhead.bbox.width:
                    continue

                # Calculate tail point (far end of shaft from arrowhead)
                if direction > 90 or direction < -90:
                    # Arrow points left, tail is at right end of shaft
                    tail = (bbox.x1, shaft_center_y)
                else:
                    # Arrow points right, tail is at left end of shaft
                    tail = (bbox.x0, shaft_center_y)
            else:
                # Vertical shaft - check horizontal alignment with arrowhead center
                arrowhead_center_x = (arrowhead.bbox.x0 + arrowhead.bbox.x1) / 2
                if abs(shaft_center_x - arrowhead_center_x) > max_shaft_gap + thickness:
                    continue

                # Check vertical proximity to arrowhead
                if direction > 0:
                    # Arrow points down, shaft should be above
                    gap = arrowhead.bbox.y0 - bbox.y1
                else:
                    # Arrow points up, shaft should be below
                    gap = bbox.y0 - arrowhead.bbox.y1

                if gap > max_shaft_gap or gap < -arrowhead.bbox.height:
                    continue

                # Calculate tail point (far end of shaft from arrowhead)
                if direction > 0:
                    # Arrow points down, tail is at top of shaft
                    tail = (shaft_center_x, bbox.y0)
                else:
                    # Arrow points up, tail is at bottom of shaft
                    tail = (shaft_center_x, bbox.y1)

            # Track the closest/best shaft
            if distance < best_distance:
                best_distance = distance
                best_shaft = drawing
                best_tail = tail

        if best_shaft is not None and best_tail is not None:
            return (best_shaft, best_tail)
        return None

    def _find_cornered_shaft(
        self,
        arrowhead: Drawing,
        direction: float,
        all_drawings: list[Drawing],
    ) -> tuple[Drawing, tuple[float, float]] | None:
        """Find an L-shaped (cornered) shaft connected to an arrowhead.

        L-shaped shafts are paths consisting of multiple line segments and/or
        rectangles that form a corner. They are used when arrows need to point
        around obstacles.

        The shaft must:
        - Have line ("l") items and/or rectangle ("re") items
        - Have the same fill color as the arrowhead
        - Connect to the arrowhead base

        Args:
            arrowhead: The arrowhead Drawing block
            direction: Direction angle in degrees (0=right, 90=down, etc.)
            all_drawings: All Drawing blocks on the page to search

        Returns:
            Tuple of (shaft Drawing block, tail point) if found, None otherwise
        """
        # Calculate the arrowhead base center (opposite the tip)
        arrowhead_center_x = (arrowhead.bbox.x0 + arrowhead.bbox.x1) / 2
        arrowhead_center_y = (arrowhead.bbox.y0 + arrowhead.bbox.y1) / 2

        # The base is on the opposite side from the tip
        # For a downward-pointing arrow, base is at the top
        # For an upward-pointing arrow, base is at the bottom
        # etc.
        if -45 <= direction < 45:
            # Points right, base is on the left
            base_x = arrowhead.bbox.x0
            base_y = arrowhead_center_y
        elif 45 <= direction < 135:
            # Points down, base is at the top
            base_x = arrowhead_center_x
            base_y = arrowhead.bbox.y0
        elif direction >= 135 or direction < -135:
            # Points left, base is on the right
            base_x = arrowhead.bbox.x1
            base_y = arrowhead_center_y
        else:
            # Points up, base is at the bottom
            base_x = arrowhead_center_x
            base_y = arrowhead.bbox.y1

        # Maximum distance from arrowhead base to consider a connection
        max_connection_distance = 15.0  # pixels

        best_shaft: Drawing | None = None
        best_tail: tuple[float, float] | None = None
        best_distance = float("inf")

        for drawing in all_drawings:
            if drawing is arrowhead:
                continue

            # Must be filled
            if not drawing.fill_color:
                continue

            # Should have same fill color as arrowhead (within tolerance)
            if arrowhead.fill_color and not self._colors_match(
                arrowhead.fill_color, drawing.fill_color
            ):
                continue

            items = drawing.items
            if not items:
                continue

            # Check if this looks like an L-shaped shaft path
            # It should have multiple line items (forming the L-shape)
            # Single rectangles are handled by _find_simple_shaft
            line_items = [item for item in items if item[0] == "l"]

            # Must have at least 2 line items to form an L-shape
            # Paths with only rectangles are handled by _find_simple_shaft
            if len(line_items) < 2:
                continue

            # Reject if there are curve items - those are typically not shafts
            curve_items = [item for item in items if item[0] == "c"]
            if curve_items:
                continue

            # Extract all points from the path
            all_points = self._extract_path_points(items)
            if len(all_points) < 2:
                continue

            # Find the point closest to the arrowhead base
            closest_dist = float("inf")
            for p in all_points:
                dist = math.sqrt((p[0] - base_x) ** 2 + (p[1] - base_y) ** 2)
                if dist < closest_dist:
                    closest_dist = dist

            # Check if this shaft connects to the arrowhead
            if closest_dist > max_connection_distance:
                continue

            # Find the tail point (furthest point from the arrowhead base)
            tail_point = None
            furthest_dist = 0.0
            for p in all_points:
                dist = math.sqrt((p[0] - base_x) ** 2 + (p[1] - base_y) ** 2)
                if dist > furthest_dist:
                    furthest_dist = dist
                    tail_point = p

            if tail_point is None:
                continue

            # Track the best shaft (closest connection to arrowhead)
            if closest_dist < best_distance:
                best_distance = closest_dist
                best_shaft = drawing
                best_tail = tail_point

        if best_shaft is not None and best_tail is not None:
            return (best_shaft, best_tail)
        return None

    def _extract_path_points(
        self, items: tuple[tuple, ...] | list[tuple]
    ) -> list[tuple[float, float]]:
        """Extract all unique points from path items.

        Args:
            items: List of path items (lines, rectangles, curves)

        Returns:
            List of unique (x, y) points from the path
        """
        points: list[tuple[float, float]] = []
        seen: set[tuple[float, float]] = set()

        for item in items:
            item_type = item[0]

            if item_type == "l":
                # Line: ('l', (x1, y1), (x2, y2))
                p1, p2 = item[1], item[2]
                for p in [p1, p2]:
                    key = (round(p[0], 1), round(p[1], 1))
                    if key not in seen:
                        seen.add(key)
                        points.append((p[0], p[1]))

            elif item_type == "re":
                # Rectangle: ('re', (x0, y0, x1, y1), ...)
                rect = item[1]
                # Add all four corners
                corners = [
                    (rect[0], rect[1]),  # top-left
                    (rect[2], rect[1]),  # top-right
                    (rect[0], rect[3]),  # bottom-left
                    (rect[2], rect[3]),  # bottom-right
                ]
                for p in corners:
                    key = (round(p[0], 1), round(p[1], 1))
                    if key not in seen:
                        seen.add(key)
                        points.append(p)

        return points

    def _colors_match(
        self,
        color1: tuple[float, ...],
        color2: tuple[float, ...],
        tolerance: float = 0.1,
    ) -> bool:
        """Check if two colors match within a tolerance.

        Args:
            color1: First color as RGB tuple (0.0-1.0)
            color2: Second color as RGB tuple (0.0-1.0)
            tolerance: Maximum difference per channel

        Returns:
            True if colors match within tolerance
        """
        if len(color1) != len(color2):
            return False
        return all(
            abs(c1 - c2) <= tolerance for c1, c2 in zip(color1, color2, strict=True)
        )

    def build(self, candidate: Candidate, result: ClassificationResult) -> Arrow:
        """Construct an Arrow element from a candidate."""
        score_details = candidate.score_details
        assert isinstance(score_details, _ArrowScore)

        # Build ArrowHead instances from head data
        heads = [
            ArrowHead(tip=head.tip, direction=head.direction)
            for head in score_details.heads
        ]

        return Arrow(
            bbox=candidate.bbox,
            heads=heads,
            tail=score_details.tail,
        )
