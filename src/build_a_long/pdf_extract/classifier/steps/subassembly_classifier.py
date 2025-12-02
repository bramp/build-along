"""
SubAssembly classifier.

Purpose
-------
Identify sub-assembly callout boxes on LEGO instruction pages. SubAssemblies
typically:
- Are white/light-colored rectangular boxes
- Contain a count label (e.g., "2x") indicating how many to build
- Contain a small diagram/image of the sub-assembly
- Have an arrow pointing from them to the main diagram

Heuristic
---------
1. Find Drawing blocks that form rectangular boxes (potential subassembly containers)
2. Look for step_count candidates inside the boxes
3. Look for diagram candidates inside the boxes
4. Optionally find arrows near the boxes

Debugging
---------
Set environment variables to aid investigation without code changes:

- LOG_LEVEL=DEBUG
    Enables DEBUG-level logging (if not already configured by caller).
"""

from __future__ import annotations

import logging
from typing import ClassVar

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.config import SubAssemblyConfig
from build_a_long.pdf_extract.classifier.label_classifier import LabelClassifier
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Diagram,
    StepCount,
    SubAssembly,
)
from build_a_long.pdf_extract.extractor.page_blocks import Drawing

log = logging.getLogger(__name__)


class _SubAssemblyScore(Score):
    """Internal score representation for subassembly classification."""

    box_score: float
    """Score based on box having white fill / black border (0.0-1.0)."""

    count_score: float
    """Score for having a valid step_count candidate inside (0.0-1.0)."""

    diagram_score: float
    """Score for having a diagram candidate inside (0.0-1.0)."""

    step_count_candidate: Candidate | None
    """The step_count candidate found inside the box."""

    diagram_candidate: Candidate | None
    """The diagram candidate found inside the box."""

    arrow_candidate: Candidate | None
    """Arrow candidate pointing from/near this subassembly."""

    config: SubAssemblyConfig
    """Configuration containing weights for score calculation."""

    def score(self) -> Weight:
        """Calculate final weighted score from components."""
        return (
            self.box_score * self.config.box_shape_weight
            + self.count_score * self.config.count_weight
            + self.diagram_score * self.config.diagram_weight
        )


class SubAssemblyClassifier(LabelClassifier):
    """Classifier for subassembly callout boxes."""

    output: ClassVar[str] = "subassembly"
    requires: ClassVar[frozenset[str]] = frozenset({"arrow", "step_count", "diagram"})

    def _score(self, result: ClassificationResult) -> None:
        """Score Drawing blocks as potential subassembly boxes."""
        page_data = result.page_data
        subassembly_config = self.config.subassembly

        # Get step_count, diagram, and arrow candidates
        step_count_candidates = result.get_scored_candidates(
            "step_count", valid_only=False, exclude_failed=True
        )
        diagram_candidates = result.get_scored_candidates(
            "diagram", valid_only=False, exclude_failed=True
        )
        arrow_candidates = result.get_scored_candidates(
            "arrow", valid_only=False, exclude_failed=True
        )

        # Collect all potential candidates, then deduplicate by bbox
        # Multiple Drawing blocks can have nearly identical bboxes (e.g.,
        # white-filled box and black-bordered box for the same subassembly)
        found_bboxes: list[BBox] = []

        # Find rectangular drawing blocks that could be subassembly boxes
        for block in page_data.blocks:
            if not isinstance(block, Drawing):
                continue

            bbox = block.bbox

            # Skip boxes smaller than minimum subassembly size
            # (subassemblies must be larger than individual parts)
            if (
                bbox.width < subassembly_config.min_subassembly_width
                or bbox.height < subassembly_config.min_subassembly_height
            ):
                continue

            # Skip if we've already found a similar bbox
            if any(self._bboxes_similar(bbox, found, 2.0) for found in found_bboxes):
                log.debug(
                    "[subassembly] Skipping duplicate bbox at %s",
                    bbox,
                )
                continue

            # Score the box colors (white fill, black border)
            box_score = self._score_box_colors(block)
            if box_score < 0.3:
                continue

            # Find step_count candidate inside the box
            step_count_candidate = self._find_candidate_inside(
                bbox, step_count_candidates
            )
            count_score = 1.0 if step_count_candidate else 0.0

            # Find diagram candidate inside/overlapping the box
            diagram_candidate = self._find_diagram_inside(bbox, diagram_candidates)
            diagram_score = 1.0 if diagram_candidate else 0.0

            # Find nearby arrow
            arrow_candidate = self._find_arrow_for_subassembly(bbox, arrow_candidates)

            # We need at least a box - count and diagram are optional
            score_details = _SubAssemblyScore(
                box_score=box_score,
                count_score=count_score,
                diagram_score=diagram_score,
                step_count_candidate=step_count_candidate,
                diagram_candidate=diagram_candidate,
                arrow_candidate=arrow_candidate,
                config=subassembly_config,
            )

            if score_details.score() < subassembly_config.min_score:
                log.debug(
                    "[subassembly] Rejected box at %s: score=%.2f < min_score=%.2f",
                    bbox,
                    score_details.score(),
                    subassembly_config.min_score,
                )
                continue

            # Track this bbox to avoid duplicates
            found_bboxes.append(bbox)

            result.add_candidate(
                Candidate(
                    bbox=bbox,
                    label="subassembly",
                    score=score_details.score(),
                    score_details=score_details,
                    source_blocks=[block],
                )
            )
            log.debug(
                "[subassembly] Candidate at %s: has_count=%s, has_diagram=%s, score=%.2f",
                bbox,
                step_count_candidate is not None,
                diagram_candidate is not None,
                score_details.score(),
            )

    def _score_box_colors(self, block: Drawing) -> float:
        """Score a drawing block based on having white fill.

        SubAssembly boxes typically have a white or light fill color.
        The outer black border boxes can be matched separately later.

        Args:
            block: The Drawing block to analyze

        Returns:
            Score from 0.0 to 1.0 where 1.0 is white fill
        """
        # Check fill color (white or light = good)
        if block.fill_color is not None:
            r, g, b = block.fill_color
            # Check if it's white or very light (all channels > 0.9)
            if r > 0.9 and g > 0.9 and b > 0.9:
                return 1.0
            # Light gray is also acceptable
            if r > 0.7 and g > 0.7 and b > 0.7:
                return 0.6

        return 0.0

    def _bboxes_similar(self, a: BBox, b: BBox, tolerance: float) -> bool:
        """Check if two bboxes are nearly identical within tolerance.

        Args:
            a: First bounding box
            b: Second bounding box
            tolerance: Maximum difference allowed for each coordinate

        Returns:
            True if bboxes are within tolerance of each other
        """
        return (
            abs(a.x0 - b.x0) <= tolerance
            and abs(a.y0 - b.y0) <= tolerance
            and abs(a.x1 - b.x1) <= tolerance
            and abs(a.y1 - b.y1) <= tolerance
        )

    def _find_candidate_inside(
        self, bbox: BBox, candidates: list[Candidate]
    ) -> Candidate | None:
        """Find the best candidate that is fully inside the given box.

        Args:
            bbox: The bounding box of the subassembly container
            candidates: Candidates to search

        Returns:
            The best candidate inside the box, or None
        """
        best_candidate = None
        best_score = 0.0

        for candidate in candidates:
            if bbox.contains(candidate.bbox) and candidate.score > best_score:
                best_candidate = candidate
                best_score = candidate.score

        return best_candidate

    def _find_diagram_inside(
        self, bbox: BBox, diagram_candidates: list[Candidate]
    ) -> Candidate | None:
        """Find the best diagram candidate that overlaps with the box.

        Args:
            bbox: The bounding box of the subassembly container
            diagram_candidates: Diagram candidates to search

        Returns:
            The best diagram candidate overlapping the box, or None
        """
        best_candidate = None
        best_overlap = 0.0

        for candidate in diagram_candidates:
            if bbox.overlaps(candidate.bbox):
                # Calculate overlap area
                # TODO Should this use bbox.intersection_area(candidate.bbox)?
                overlap = bbox.intersect(candidate.bbox)
                overlap_area = overlap.width * overlap.height
                if overlap_area > best_overlap:
                    best_candidate = candidate
                    best_overlap = overlap_area

        return best_candidate

    def _find_arrow_for_subassembly(
        self, bbox: BBox, arrow_candidates: list[Candidate]
    ) -> Candidate | None:
        """Find an arrow that points from/near this subassembly box.

        Looks for arrows that are either:
        - Inside the box
        - Adjacent to the box (within a small margin)

        Args:
            bbox: The bounding box of the subassembly container
            arrow_candidates: All arrow candidates on the page

        Returns:
            The best matching arrow candidate, or None
        """
        margin = 20.0  # Points of margin around the box
        expanded_bbox = BBox(
            x0=bbox.x0 - margin,
            y0=bbox.y0 - margin,
            x1=bbox.x1 + margin,
            y1=bbox.y1 + margin,
        )

        best_arrow = None
        best_score = 0.0

        for arrow_candidate in arrow_candidates:
            if (
                expanded_bbox.overlaps(arrow_candidate.bbox)
                and arrow_candidate.score > best_score
            ):
                best_arrow = arrow_candidate
                best_score = arrow_candidate.score

        return best_arrow

    def build(self, candidate: Candidate, result: ClassificationResult) -> SubAssembly:
        """Construct a SubAssembly element from a candidate."""
        score_details = candidate.score_details
        assert isinstance(score_details, _SubAssemblyScore)

        # Build the step_count element if present
        count = None
        if score_details.step_count_candidate:
            count_elem = result.build(score_details.step_count_candidate)
            assert isinstance(count_elem, StepCount)
            count = count_elem

        # Build diagram if present
        diagram = None
        if score_details.diagram_candidate:
            diagram_elem = result.build(score_details.diagram_candidate)
            assert isinstance(diagram_elem, Diagram)
            diagram = diagram_elem

        return SubAssembly(
            bbox=candidate.bbox,
            diagram=diagram,
            count=count,
        )
