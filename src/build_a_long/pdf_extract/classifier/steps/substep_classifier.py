"""
SubStep classifier.

Purpose
-------
Identify SubStep elements by finding substep numbers paired with diagrams.

SubSteps are mini-steps that appear either:
1. Inside SubAssembly callout boxes (numbered 1, 2, 3 within the box)
2. As "naked" substeps on the page (small numbers 1, 2, 3, 4 alongside a main step)

This classifier pairs substep_number with diagram candidates based on position:
- Step number should be to the left or above the diagram
- Step number and diagram should be relatively close together

Architecture
------------
This classifier independently finds SubStep candidates during scoring:
- Gets substep_number candidates (from SubStepNumberClassifier - smaller font)
- Gets diagram candidates
- Creates SubStep candidates by pairing substep numbers with nearby diagrams

These candidates are then used by:
- SubAssemblyClassifier: claims SubSteps that are inside callout boxes
- StepClassifier: uses remaining SubSteps as "naked" substeps

The key insight is that SubStep step numbers have a SMALLER FONT SIZE than
main step numbers. SubStepNumberClassifier detects these smaller font step numbers
and this classifier pairs them with diagrams.
"""

import logging
from typing import ClassVar

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import LabelClassifier
from build_a_long.pdf_extract.classifier.rule_based_classifier import StepNumberScore
from build_a_long.pdf_extract.classifier.score import Score
from build_a_long.pdf_extract.classifier.steps.pairing import (
    DEFAULT_MAX_PAIRING_DISTANCE,
    PairingConfig,
    find_optimal_pairings,
)
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Diagram,
    StepNumber,
    SubStep,
)

log = logging.getLogger(__name__)


class _SubStepScore(Score):
    """Score details for SubStep candidates.

    Scoring is based on:
    - Position: step number should be to the left/top of the diagram
    - Distance: step number should be close to the diagram
    """

    step_value: int
    """The parsed step number value (e.g., 1, 2, 3)."""

    substep_number_candidate: Candidate
    """The substep_number candidate for this substep."""

    diagram_candidate: Candidate
    """The diagram candidate paired with this step number."""

    position_score: float
    """Score based on step number being to left/top of diagram (0.0-1.0)."""

    distance_score: float
    """Score based on distance between step number and diagram (0.0-1.0)."""

    def score(self) -> float:
        """Return the weighted score value."""
        return self.position_score * 0.5 + self.distance_score * 0.5


class SubStepClassifier(LabelClassifier):
    """Classifier for SubStep elements.

    This classifier finds step numbers and pairs them with nearby diagrams based
    on position. The pairing creates SubStep candidates which are then:
    - Claimed by SubAssemblyClassifier if inside a callout box
    - Used by StepClassifier as naked substeps if not inside a box

    Scoring phase:
    - Gets all step_number candidates
    - Gets all diagram candidates
    - Creates SubStep candidates by pairing step numbers with diagrams
      where the step number is to the left/above the diagram

    Build phase:
    - Builds the step_number from its candidate (substep_number -> StepNumber)
    - Builds the diagram from its candidate
    - Creates the SubStep with both elements
    """

    output: ClassVar[str] = "substep"
    requires: ClassVar[frozenset[str]] = frozenset({"substep_number", "diagram"})

    def _score(self, result: ClassificationResult) -> None:
        """Score substep number + diagram pairings to create SubStep candidates."""
        # Get substep number candidates (small font step numbers)
        # During scoring, candidates are not yet constructed, so is_valid=False
        substep_number_candidates = result.get_scored_candidates(
            "substep_number", valid_only=False, exclude_failed=True
        )

        if not substep_number_candidates:
            log.debug("[substep] No substep_number candidates found")
            return

        # Get diagram candidates (not yet built - use valid_only=False)
        diagram_candidates = result.get_scored_candidates(
            "diagram", valid_only=False, exclude_failed=True
        )

        if not diagram_candidates:
            log.debug("[substep] No diagram candidates found")
            return

        log.debug(
            "[substep] Found %d substep numbers, %d diagrams",
            len(substep_number_candidates),
            len(diagram_candidates),
        )

        # Use Hungarian algorithm to find optimal pairing
        self._create_candidates_with_hungarian(
            substep_number_candidates,
            diagram_candidates,
            result,
        )

    def _get_step_value(self, candidate: Candidate) -> int:
        """Get the step number value from a candidate's score."""
        score_details = candidate.score_details
        if isinstance(score_details, StepNumberScore):
            return score_details.step_value
        return 0

    def _create_candidates_with_hungarian(
        self,
        step_candidates: list[Candidate],
        diagram_candidates: list[Candidate],
        result: ClassificationResult,
    ) -> None:
        """Use Hungarian algorithm to optimally pair step numbers with diagrams.

        Uses the shared pairing module to find optimal step number to diagram
        pairings based on position and distance.

        Args:
            step_candidates: Step number candidates
            diagram_candidates: Diagram candidates
            result: Classification result to add candidates to
        """
        if not step_candidates or not diagram_candidates:
            return

        # Get dividers for obstruction checking
        divider_candidates = result.get_scored_candidates("divider", valid_only=True)
        divider_bboxes = [
            c.constructed.bbox for c in divider_candidates if c.constructed is not None
        ]

        # Extract bboxes for pairing
        step_bboxes = [c.bbox for c in step_candidates]
        diagram_bboxes = [c.bbox for c in diagram_candidates]

        # Configure pairing: substeps use smaller max distance
        config = PairingConfig(
            max_distance=DEFAULT_MAX_PAIRING_DISTANCE,
            position_weight=0.5,
            distance_weight=0.5,
            check_dividers=True,
            top_left_tolerance=100.0,
        )

        # Find optimal pairings using shared logic
        pairings = find_optimal_pairings(
            step_bboxes, diagram_bboxes, config, divider_bboxes
        )

        # Create candidates from pairings
        for pairing in pairings:
            substep_cand = step_candidates[pairing.step_index]
            diag_cand = diagram_candidates[pairing.diagram_index]

            step_value = self._get_step_value(substep_cand)
            score_details = _SubStepScore(
                step_value=step_value,
                substep_number_candidate=substep_cand,
                diagram_candidate=diag_cand,
                position_score=pairing.position_score,
                distance_score=pairing.distance_score,
            )

            # Combined bbox
            combined_bbox = substep_cand.bbox.union(diag_cand.bbox)

            candidate = Candidate(
                bbox=combined_bbox,
                label="substep",
                score=score_details.score(),
                score_details=score_details,
                source_blocks=[],  # Blocks claimed via nested candidates
            )

            result.add_candidate(candidate)
            log.debug(
                "[substep] Created candidate: step=%s, score=%.2f",
                score_details.step_value,
                score_details.score(),
            )

    def build(
        self,
        candidate: Candidate,
        result: ClassificationResult,
        constraint_bbox: BBox | None = None,
    ) -> SubStep:
        """Construct a SubStep from a candidate.

        Args:
            candidate: The candidate to construct, with score_details containing
                substep_number_candidate and diagram_candidate.
            result: The classification result context.
            constraint_bbox: Optional bounding box to constrain diagram clustering.
                When building inside a SubAssembly, this should be the
                SubAssembly's bbox to prevent diagrams from clustering beyond
                the SubAssembly boundaries.

        Returns:
            The constructed SubStep element.
        """
        score = candidate.score_details
        assert isinstance(score, _SubStepScore)

        # Build the step number from the substep_number candidate
        # (SubStepNumberClassifier.build() returns a StepNumber element)
        step_num_elem = result.build(score.substep_number_candidate)
        assert isinstance(step_num_elem, StepNumber)

        # Build the diagram from its candidate, passing constraint_bbox if provided
        # to prevent the diagram from clustering beyond the SubAssembly boundaries
        diagram_elem = result.build(
            score.diagram_candidate, constraint_bbox=constraint_bbox
        )
        assert isinstance(diagram_elem, Diagram)

        # Compute bbox including both step number and diagram
        substep_bbox = step_num_elem.bbox.union(diagram_elem.bbox)

        log.debug(
            "[substep] Built SubStep %d",
            step_num_elem.value,
        )

        return SubStep(
            bbox=substep_bbox,
            step_number=step_num_elem,
            diagram=diagram_elem,
        )
