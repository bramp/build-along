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
from build_a_long.pdf_extract.classifier.utils import (
    colors_match,
    extract_unique_points,
)
from build_a_long.pdf_extract.extractor.bbox import BBox, filter_overlapping
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

        # Phase 2: Group arrowheads that share the same shaft or have nearby tails
        # This handles:
        # - Y-shaped arrows: multiple heads with tails close together
        # - L-shaped arrows: multiple heads sharing the same shaft block
        tolerance = arrow_config.tail_grouping_tolerance
        groups = self._group_arrowheads(arrowheads, tolerance)

        # Phase 3: Create candidates
        for heads in groups:
            self._add_arrow_candidate(result, heads)

    def _group_arrowheads(
        self, arrowheads: list[_ArrowHeadData], tail_tolerance: float
    ) -> list[list[_ArrowHeadData]]:
        """Group arrowheads that belong to the same arrow.

        Arrowheads are grouped together if they:
        1. Share the same shaft_block (same object identity), OR
        2. Have tails within tail_tolerance distance of each other

        Args:
            arrowheads: List of arrowhead data to group
            tail_tolerance: Maximum distance between tail coordinates to be grouped

        Returns:
            List of groups, where each group is a list of arrowheads
        """
        if not arrowheads:
            return []

        # Use union-find to group arrowheads
        # Each arrowhead starts in its own group
        parent: dict[int, int] = {i: i for i in range(len(arrowheads))}

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Group by shared shaft_block (same object identity)
        shaft_to_indices: dict[int, list[int]] = {}
        for i, head in enumerate(arrowheads):
            if head.shaft_block is not None:
                shaft_id = id(head.shaft_block)
                if shaft_id in shaft_to_indices:
                    # Union with first arrowhead sharing this shaft
                    union(i, shaft_to_indices[shaft_id][0])
                    shaft_to_indices[shaft_id].append(i)
                else:
                    shaft_to_indices[shaft_id] = [i]

        # Group by tail proximity
        for i, head_i in enumerate(arrowheads):
            if head_i.tail is None:
                continue
            for j, head_j in enumerate(arrowheads):
                if i >= j or head_j.tail is None:
                    continue
                dx = head_i.tail[0] - head_j.tail[0]
                dy = head_i.tail[1] - head_j.tail[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist <= tail_tolerance:
                    union(i, j)

        # Collect groups
        group_map: dict[int, list[_ArrowHeadData]] = {}
        for i, head in enumerate(arrowheads):
            root = find(i)
            group_map.setdefault(root, []).append(head)

        return list(group_map.values())

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
        arrow_bbox = BBox.union_all([b.bbox for b in source_blocks])

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
        points = extract_unique_points(line_items)
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

    def _find_shaft(
        self,
        arrowhead: Drawing,
        direction: float,
        tip: tuple[float, float],
        all_drawings: list[Drawing],
    ) -> tuple[Drawing, tuple[float, float]] | None:
        """Find the shaft connected to an arrowhead.

        The shaft can be:
        1. A thin filled rectangle (single "re" item)
        2. A stroked line (single "l" item with stroke_color)
        3. A path with multiple "l" (line) items - L-shaped shaft

        The shaft must:
        - Be positioned adjacent to the arrowhead base (opposite the tip)
        - Have a color matching the arrowhead's fill_color (either fill or stroke)

        All shaft types are handled uniformly by extracting their endpoints
        and finding the point closest to the tip (connection) and furthest
        from the tip (tail).

        Args:
            arrowhead: The arrowhead Drawing block
            direction: Direction angle in degrees (0=right, 90=down, etc.)
            tip: The tip point (x, y) of the arrowhead
            all_drawings: All Drawing blocks on the page to search

        Returns:
            Tuple of (shaft Drawing block, tail point) if found, None otherwise
        """
        # Optimization: Filter to drawings near the arrowhead
        # Shaft must connect to arrowhead, so it must overlap a slightly expanded bbox
        search_bbox = arrowhead.bbox.expand(20.0)  # generous margin
        nearby_drawings = filter_overlapping(all_drawings, search_bbox)

        # Configuration
        max_connection_distance = 15.0  # pixels - max gap between shaft and arrowhead
        min_shaft_length = 10.0  # pixels - minimum distance from connection to tail
        max_shaft_thickness = 5.0  # pixels - for thin rect shafts

        best_shaft: Drawing | None = None
        best_tail: tuple[float, float] | None = None
        best_distance = float("inf")

        for drawing in nearby_drawings:
            if drawing is arrowhead:
                continue

            # Check color match - shaft color must match arrowhead fill_color
            # Shaft can be either filled (fill_color) or stroked (stroke_color)
            shaft_color = drawing.fill_color or drawing.stroke_color
            if not shaft_color:
                continue

            if arrowhead.fill_color and not colors_match(
                arrowhead.fill_color, shaft_color
            ):
                continue

            items = drawing.items
            if not items:
                continue

            # Reject if there are curve items - those are typically not shafts
            if any(item[0] == "c" for item in items):
                continue

            # For thin rectangles, check thickness constraint
            if len(items) == 1 and items[0][0] == "re":
                bbox = drawing.bbox
                thickness = min(bbox.width, bbox.height)
                if thickness > max_shaft_thickness:
                    continue

            # Extract all points from the drawing
            points = self._extract_path_points(items)
            if len(points) < 2:
                continue

            # Find the point closest to the arrowhead tip (connection point)
            closest_point = None
            closest_dist = float("inf")
            for p in points:
                dist = math.sqrt((p[0] - tip[0]) ** 2 + (p[1] - tip[1]) ** 2)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_point = p

            # Check if this shaft connects to the arrowhead
            if closest_dist > max_connection_distance:
                continue

            # Find the tail point (furthest point from the tip)
            tail_point = None
            furthest_dist = 0.0
            for p in points:
                dist = math.sqrt((p[0] - tip[0]) ** 2 + (p[1] - tip[1]) ** 2)
                if dist > furthest_dist:
                    furthest_dist = dist
                    tail_point = p

            if tail_point is None or closest_point is None:
                continue

            # Check minimum shaft length (distance from connection to tail)
            shaft_length = math.sqrt(
                (tail_point[0] - closest_point[0]) ** 2
                + (tail_point[1] - closest_point[1]) ** 2
            )
            if shaft_length < min_shaft_length:
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
