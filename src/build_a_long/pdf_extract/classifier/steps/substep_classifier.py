"""
SubStep classifier.

Purpose
-------
Identify sub-step callout boxes on LEGO instruction pages. SubSteps typically:
- Are white/light-colored rectangular boxes
- Contain a count label (e.g., "2x") indicating how many to build
- Contain a small diagram/image of the sub-assembly
- Have an arrow pointing from them to the main diagram

Heuristic
---------
1. Find Drawing blocks that form rectangular boxes (potential substep containers)
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
from build_a_long.pdf_extract.classifier.config import SubStepConfig
from build_a_long.pdf_extract.classifier.label_classifier import LabelClassifier
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Arrow,
    Diagram,
    StepCount,
    SubStep,
)
from build_a_long.pdf_extract.extractor.page_blocks import Drawing

log = logging.getLogger(__name__)


class _SubStepScore(Score):
    """Internal score representation for substep classification."""

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
    """Arrow candidate pointing from/near this substep."""

    config: SubStepConfig
    """Configuration containing weights for score calculation."""

    def score(self) -> Weight:
        """Calculate final weighted score from components."""
        return (
            self.box_score * self.config.box_shape_weight
            + self.count_score * self.config.count_weight
            + self.diagram_score * self.config.diagram_weight
        )


class SubStepClassifier(LabelClassifier):
    """Classifier for substep callout boxes."""

    output: ClassVar[str] = "substep"
    requires: ClassVar[frozenset[str]] = frozenset({"arrow", "step_count", "diagram"})

    def _score(self, result: ClassificationResult) -> None:
        """Score Drawing blocks as potential substep boxes."""
        page_data = result.page_data
        substep_config = self.config.substep

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

        # Find rectangular drawing blocks that could be substep boxes
        for block in page_data.blocks:
            if not isinstance(block, Drawing):
                continue

            bbox = block.bbox

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
            arrow_candidate = self._find_arrow_for_substep(bbox, arrow_candidates)

            # We need at least a box - count and diagram are optional
            score_details = _SubStepScore(
                box_score=box_score,
                count_score=count_score,
                diagram_score=diagram_score,
                step_count_candidate=step_count_candidate,
                diagram_candidate=diagram_candidate,
                arrow_candidate=arrow_candidate,
                config=substep_config,
            )

            if score_details.score() < substep_config.min_score:
                log.debug(
                    "[substep] Rejected box at %s: score=%.2f < min_score=%.2f",
                    bbox,
                    score_details.score(),
                    substep_config.min_score,
                )
                continue

            result.add_candidate(
                Candidate(
                    bbox=bbox,
                    label="substep",
                    score=score_details.score(),
                    score_details=score_details,
                    source_blocks=[block],
                )
            )
            log.debug(
                "[substep] Candidate at %s: has_count=%s, has_diagram=%s, score=%.2f",
                bbox,
                step_count_candidate is not None,
                diagram_candidate is not None,
                score_details.score(),
            )

    def _score_box_colors(self, block: Drawing) -> float:
        """Score a drawing block based on having white fill and/or black border.

        Substep boxes typically have:
        - White or light fill color
        - Black or dark stroke/border color

        Args:
            block: The Drawing block to analyze

        Returns:
            Score from 0.0 to 1.0 where 1.0 is white fill with black border
        """
        score = 0.0

        # Check fill color (white or light = good)
        if block.fill_color is not None:
            r, g, b = block.fill_color
            # Check if it's white or very light (all channels > 0.9)
            if r > 0.9 and g > 0.9 and b > 0.9:
                score += 0.5
            # Light gray is also acceptable
            elif r > 0.7 and g > 0.7 and b > 0.7:
                score += 0.3

        # Check stroke color (black or dark = good)
        if block.stroke_color is not None:
            r, g, b = block.stroke_color
            # Check if it's black or very dark (all channels < 0.2)
            if r < 0.2 and g < 0.2 and b < 0.2:
                score += 0.5
            # Dark gray is also acceptable
            elif r < 0.4 and g < 0.4 and b < 0.4:
                score += 0.3

        # If no fill but has black stroke, still decent
        if block.fill_color is None and block.stroke_color is not None:
            r, g, b = block.stroke_color
            if r < 0.2 and g < 0.2 and b < 0.2:
                score = 0.4

        return min(score, 1.0)

    def _find_candidate_inside(
        self, bbox: BBox, candidates: list[Candidate]
    ) -> Candidate | None:
        """Find the best candidate that is fully inside the given box.

        Args:
            bbox: The bounding box of the substep container
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
            bbox: The bounding box of the substep container
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

    def _find_arrow_for_substep(
        self, bbox: BBox, arrow_candidates: list[Candidate]
    ) -> Candidate | None:
        """Find an arrow that points from/near this substep box.

        Looks for arrows that are either:
        - Inside the box
        - Adjacent to the box (within a small margin)

        Args:
            bbox: The bounding box of the substep container
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

    def build(self, candidate: Candidate, result: ClassificationResult) -> SubStep:
        """Construct a SubStep element from a candidate."""
        score_details = candidate.score_details
        assert isinstance(score_details, _SubStepScore)

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

        # Build arrow if present
        arrow = None
        if score_details.arrow_candidate:
            arrow_elem = result.build(score_details.arrow_candidate)
            assert isinstance(arrow_elem, Arrow)
            arrow = arrow_elem

        return SubStep(
            bbox=candidate.bbox,
            diagram=diagram,
            count=count,
            arrow=arrow,
        )
