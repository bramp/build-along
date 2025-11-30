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
1. Find Drawing blocks with triangular shapes (3-4 line items)
2. Filter to small filled shapes (5-20px, filled color)
3. Calculate the tip (furthest point from centroid)
4. Calculate direction angle from centroid to tip

Arrows are detected by their arrowhead - a filled triangular shape.
The arrow shaft (connecting line) is not currently tracked, though
the arrowhead alone provides directional information.

Debugging
---------
Enable with `LOG_LEVEL=DEBUG` for structured logs.
"""

from __future__ import annotations

import logging
import math
from typing import ClassVar

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import LabelClassifier
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.lego_page_elements import Arrow
from build_a_long.pdf_extract.extractor.page_blocks import Drawing

log = logging.getLogger(__name__)


class _ArrowScore(Score):
    """Internal score representation for arrow classification."""

    shape_score: float
    """Score based on shape being triangular (0.0-1.0)."""

    size_score: float
    """Score based on size being in expected range (0.0-1.0)."""

    direction: float
    """Direction angle in degrees (0=right, 90=down, 180=left, -90=up)."""

    tip: tuple[float, float]
    """The tip point (x, y) of the arrowhead."""

    # Store weights for score calculation
    shape_weight: float = 0.7
    size_weight: float = 0.3

    def score(self) -> Weight:
        """Calculate final weighted score from components."""
        return self.shape_score * self.shape_weight + self.size_score * self.size_weight


class ArrowClassifier(LabelClassifier):
    """Classifier for arrow elements (arrowheads)."""

    output: ClassVar[str] = "arrow"
    requires: ClassVar[frozenset[str]] = frozenset()

    def _score(self, result: ClassificationResult) -> None:
        """Score Drawing blocks as potential arrowheads."""
        page_data = result.page_data
        arrow_config = self.config.arrow

        # Process each Drawing block
        for block in page_data.blocks:
            if not isinstance(block, Drawing):
                continue

            score_details = self._score_drawing(block)
            if score_details is None:
                continue

            if score_details.score() < arrow_config.min_score:
                log.debug(
                    "[arrow] Rejected at %s: score=%.2f < min_score=%.2f",
                    block.bbox,
                    score_details.score(),
                    arrow_config.min_score,
                )
                continue

            result.add_candidate(
                Candidate(
                    bbox=block.bbox,
                    label="arrow",
                    score=score_details.score(),
                    score_details=score_details,
                    source_blocks=[block],
                )
            )
            log.debug(
                "[arrow] Candidate at %s: score=%.2f, direction=%.0f°",
                block.bbox,
                score_details.score(),
                score_details.direction,
            )

    def _score_drawing(self, block: Drawing) -> _ArrowScore | None:
        """Score a Drawing block as a potential arrowhead.

        Args:
            block: The Drawing block to analyze

        Returns:
            Score details if this could be an arrowhead, None otherwise
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
        if aspect < arrow_config.min_aspect or aspect > arrow_config.max_aspect:
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

        return _ArrowScore(
            shape_score=shape_score,
            size_score=size_score,
            direction=direction,
            tip=tip,
            shape_weight=arrow_config.shape_weight,
            size_weight=arrow_config.size_weight,
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

    def build(self, candidate: Candidate, result: ClassificationResult) -> Arrow:
        """Construct an Arrow element from a candidate."""
        score_details = candidate.score_details
        assert isinstance(score_details, _ArrowScore)

        return Arrow(
            bbox=candidate.bbox,
            direction=score_details.direction,
            tip=score_details.tip,
        )
