"""
Step classifier.

Purpose
-------
Identify complete Step structures by combining step_number, parts_list, and diagram
elements. A Step represents a single building instruction comprising:
- A StepNumber label
- An optional PartsList (the parts needed for this step)
- A Diagram (the main instruction graphic showing what to build)

We look for step_numbers and attempt to pair them with nearby parts_lists and
identify the appropriate diagram region for each step.

Debugging
---------
Set environment variables to aid investigation without code changes:

- LOG_LEVEL=DEBUG
    Enables DEBUG-level logging (if not already configured by caller).
"""

import logging
from dataclasses import dataclass

from build_a_long.pdf_extract.classifier.classification_result import (
    Candidate,
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Diagram,
    PartsList,
    Step,
    StepNumber,
)

log = logging.getLogger(__name__)


@dataclass
class _StepScore:
    """Internal score representation for step classification."""

    step_number: StepNumber
    """The step number this step is associated with."""

    has_parts_list: bool
    """Whether this step has an associated parts list."""

    step_proximity_score: float
    """Score based on proximity to the PartsList above (0.0-1.0).
    1.0 for closest proximity, 0.0 if very far. 0.0 if no parts list."""

    step_alignment_score: float
    """Score based on left-edge alignment with PartsList above (0.0-1.0).
    1.0 is perfect alignment, 0.0 is very misaligned. 0.0 if no parts list."""

    diagram_area: float
    """Area of the diagram region."""

    def pairing_score(self) -> float:
        """Calculate pairing quality score (average of proximity and alignment)."""
        if not self.has_parts_list:
            return 0.0
        return (self.step_proximity_score + self.step_alignment_score) / 2.0

    def sort_key(self) -> tuple[float, int]:
        """Return a tuple for sorting candidates.

        We prefer:
        1. Higher pairing scores (better StepNumber-PartsList match)
        2. Lower step number values (to break ties and maintain order)
        """
        return (-self.pairing_score(), self.step_number.value)


@dataclass(frozen=True)
class StepClassifier(LabelClassifier):
    """Classifier for complete Step structures."""

    outputs = frozenset({"step"})
    requires = frozenset({"step_number", "parts_list"})

    def evaluate(self, result: ClassificationResult) -> None:
        """Evaluate elements and create Step candidates for all possible pairings.

        Creates Step candidates for each StepNumber, scoring all possible pairings
        with PartsLists. Candidates are stored in ClassificationResult, and the
        best ones will be selected in classify().
        """
        page_data = result.page_data

        # Get step_number candidates
        step_candidates = result.get_candidates("step_number")
        steps: list[StepNumber] = []

        for candidate in step_candidates:
            if (
                candidate.is_winner
                and candidate.constructed is not None
                and isinstance(candidate.constructed, StepNumber)
            ):
                steps.append(candidate.constructed)

        if not steps:
            return

        # Get parts_list candidates (winners only)
        parts_list_candidates = result.get_candidates("parts_list")
        parts_lists: list[PartsList] = []

        for candidate in parts_list_candidates:
            if (
                candidate.is_winner
                and candidate.constructed is not None
                and isinstance(candidate.constructed, PartsList)
            ):
                parts_lists.append(candidate.constructed)

        log.debug(
            "[step] page=%s steps=%d parts_lists=%d",
            page_data.page_number,
            len(steps),
            len(parts_lists),
        )

        # Create Step candidates for all possible pairings
        for step_num in steps:
            # Create candidates for this StepNumber paired with each PartsList
            for parts_list in parts_lists:
                self._create_step_candidate(step_num, parts_list, result)

            # Also create a candidate with no PartsList (fallback)
            self._create_step_candidate(step_num, None, result)

        log.debug(
            "[step] Created %d step candidates",
            len(result.get_candidates("step")),
        )

    def _create_step_candidate(
        self,
        step_num: StepNumber,
        parts_list: PartsList | None,
        result: ClassificationResult,
    ) -> None:
        """Create a Step candidate for a StepNumber paired with a PartsList (or None).

        Args:
            step_num: The StepNumber for this candidate
            parts_list: The PartsList to pair with (or None for no pairing)
            result: Classification result to add the candidate to
        """
        ABOVE_EPS = 2.0  # Small epsilon for "above" check
        ALIGNMENT_THRESHOLD_MULTIPLIER = 1.0  # Max horizontal offset
        DISTANCE_THRESHOLD_MULTIPLIER = 1.0  # Max vertical distance

        # Calculate pairing scores if there's a parts_list above the step
        proximity_score = 0.0
        alignment_score = 0.0

        if (
            parts_list is not None
            and parts_list.bbox.y1 <= step_num.bbox.y0 + ABOVE_EPS
        ):
            # Calculate distance (how far apart vertically)
            distance = step_num.bbox.y0 - parts_list.bbox.y1

            # Calculate proximity score
            max_distance = step_num.bbox.height * DISTANCE_THRESHOLD_MULTIPLIER
            if max_distance > 0:
                proximity_score = max(0.0, 1.0 - (distance / max_distance))

            # Calculate alignment score (how well left edges align)
            max_alignment_diff = step_num.bbox.width * ALIGNMENT_THRESHOLD_MULTIPLIER
            left_diff = abs(parts_list.bbox.x0 - step_num.bbox.x0)
            if max_alignment_diff > 0:
                alignment_score = max(0.0, 1.0 - (left_diff / max_alignment_diff))

        # Identify diagram region
        diagram_bbox = self._identify_diagram_region(step_num, parts_list, result)

        # Build Step
        diagram = Diagram(bbox=diagram_bbox)
        constructed = Step(
            bbox=self._compute_step_bbox(step_num, parts_list, diagram),
            step_number=step_num,
            parts_list=parts_list or PartsList(bbox=step_num.bbox, parts=[]),
            diagram=diagram,
        )

        # Create score
        score = _StepScore(
            step_number=step_num,
            has_parts_list=parts_list is not None,
            step_proximity_score=proximity_score,
            step_alignment_score=alignment_score,
            diagram_area=diagram_bbox.area,
        )

        # Add candidate (not yet a winner)
        step_candidate = Candidate(
            bbox=constructed.bbox,
            label="step",
            score=score.pairing_score(),
            score_details=score,
            constructed=constructed,
            source_block=None,
            failure_reason=None,
            is_winner=False,
        )

        result.add_candidate("step", step_candidate)

    def _identify_diagram_region(
        self,
        step_num: StepNumber,
        parts_list: PartsList | None,
        result: ClassificationResult,
    ) -> BBox:
        """Identify the diagram region for a step.

        The diagram is typically the large area below the step number and parts list.
        For now, we create a simple heuristic-based region.

        Args:
            step_num: The step number
            parts_list: The associated parts list (if any)
            result: Classification result containing page_data

        Returns:
            BBox representing the diagram region
        """
        page_data = result.page_data
        # Simple heuristic: use the step number's bbox as a starting point
        # In the future, we should look for actual drawing elements below the step

        # Start with step number position
        x0 = step_num.bbox.x0
        y0 = step_num.bbox.y1  # Below the step number

        # If there's a parts list, the diagram should be below it
        if parts_list:
            y0 = max(y0, parts_list.bbox.y1)

        # Extend to a reasonable area (placeholder logic)
        # TODO: Find actual drawing elements and use their bounds
        page_bbox = page_data.bbox
        assert page_bbox is not None

        # Use the rest of the page width and height as a simple approximation
        x1 = page_bbox.x1
        y1 = page_bbox.y1

        # Create a bbox for the diagram region
        return BBox(x0=x0, y0=y0, x1=x1, y1=y1)

    def _compute_step_bbox(
        self,
        step_num: StepNumber,
        parts_list: PartsList | None,
        diagram: Diagram,
    ) -> BBox:
        """Compute the overall bounding box for the Step.

        This encompasses the step number, parts list (if any), and diagram.

        Args:
            step_num: The step number element
            parts_list: The parts list (if any)
            diagram: The diagram element

        Returns:
            Combined bounding box
        """
        bboxes = [step_num.bbox, diagram.bbox]
        if parts_list:
            bboxes.append(parts_list.bbox)

        return BBox.union_all(bboxes)

    def classify(self, result: ClassificationResult) -> None:
        """Greedily select the best Step candidates.

        Uses the candidates created in evaluate() to select the best pairings.
        Ensures each StepNumber value and each PartsList is used at most once.
        """
        # Get all Step candidates
        candidate_list = result.get_candidates("step")

        # Sort candidates by score (highest first)
        sorted_candidates = sorted(
            candidate_list,
            key=lambda c: c.score_details.sort_key(),
        )

        # Track which StepNumber values and PartsLists have been used
        used_step_values: set[int] = set()
        used_parts_list_ids: set[int] = set()

        # Greedily select winners
        for candidate in sorted_candidates:
            if candidate.constructed is None:
                continue

            assert isinstance(candidate.constructed, Step)
            step = candidate.constructed
            step_value = step.step_number.value
            parts_list = step.parts_list if len(step.parts_list.parts) > 0 else None

            # Skip if this step number value is already used
            if step_value in used_step_values:
                log.debug(
                    "[step] Skipping candidate for step %d - value already used",
                    step_value,
                )
                continue

            # Skip if this parts_list is already used (if it has parts)
            if parts_list is not None:
                parts_list_id = id(parts_list)
                if parts_list_id in used_parts_list_ids:
                    log.debug(
                        "[step] Skipping candidate for step %d - "
                        "PartsList already used",
                        step_value,
                    )
                    continue
                # Claim this parts_list
                used_parts_list_ids.add(parts_list_id)

            # Mark this candidate as winner
            result.mark_winner(candidate, step)
            used_step_values.add(step_value)

            log.debug(
                "[step] Marking step %d as winner (parts_list=%s, pairing_score=%.2f)",
                step_value,
                "yes" if parts_list is not None else "no",
                candidate.score_details.pairing_score(),
            )
