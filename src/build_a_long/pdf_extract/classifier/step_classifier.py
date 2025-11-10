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
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Diagram,
    PartsList,
    Step,
    StepNumber,
)

log = logging.getLogger(__name__)


@dataclass
class _StepPartsListPairingScore:
    """Score for pairing a StepNumber with a PartsList."""

    step_proximity_score: float
    """Score based on proximity to the PartsList above (0.0-1.0).
    1.0 for closest proximity, 0.0 if very far."""

    step_alignment_score: float
    """Score based on left-edge alignment with PartsList above (0.0-1.0).
    1.0 is perfect alignment, 0.0 is very misaligned."""

    parts_list: PartsList
    """The PartsList being paired."""

    step_number: StepNumber
    """The StepNumber being paired."""

    def combined_score(self) -> float:
        """Calculate combined pairing score (average of proximity and alignment)."""
        return (self.step_proximity_score + self.step_alignment_score) / 2.0

    def sort_key(self) -> float:
        """Return sort key for matching (higher is better)."""
        return self.combined_score()


@dataclass
class _StepScore:
    """Internal score representation for step classification."""

    step_number: StepNumber
    """The step number this step is associated with."""

    has_parts_list: bool
    """Whether this step has an associated parts list."""

    diagram_area: float
    """Area of the diagram region."""

    def sort_key(self) -> tuple[int, int, float]:
        """Return a tuple for sorting candidates.

        We prefer:
        1. Lower step number values (to maintain order)
        2. Steps with parts lists over those without
        3. Larger diagram areas (more content)
        """
        return (self.step_number.value, -int(self.has_parts_list), -self.diagram_area)


class StepClassifier(LabelClassifier):
    """Classifier for complete Step structures."""

    outputs = {"step"}
    requires = {"step_number", "parts_list"}

    def __init__(self, config, classifier):
        super().__init__(config, classifier)
        self._pairing_scores: list[_StepPartsListPairingScore] = []

    def evaluate(self, page_data: PageData, result: ClassificationResult) -> None:
        """Evaluate elements and score all possible StepNumber-PartsList pairings.

        Creates pairing scores for proximity and alignment between each StepNumber
        and each PartsList. These scores will be used in classify() to greedily
        select the best pairings.
        """

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

        # Score all possible pairings
        self._pairing_scores = self._score_all_pairings(steps, parts_lists)

        log.debug("[step] Created %d pairing scores", len(self._pairing_scores))

    def _score_all_pairings(
        self,
        steps: list[StepNumber],
        parts_lists: list[PartsList],
    ) -> list[_StepPartsListPairingScore]:
        """Score all possible (StepNumber, PartsList) pairings.

        Returns a list of pairing scores, one for each viable pairing.
        """
        ABOVE_EPS = 2.0  # Small epsilon for "above" check
        ALIGNMENT_THRESHOLD_MULTIPLIER = 1.0  # Max horizontal offset
        DISTANCE_THRESHOLD_MULTIPLIER = 1.0  # Max vertical distance

        pairings: list[_StepPartsListPairingScore] = []

        for step_num in steps:
            for parts_list in parts_lists:
                # Check if parts_list is above the step number
                if parts_list.bbox.y1 > step_num.bbox.y0 + ABOVE_EPS:
                    # PartsList is not above this step, skip
                    continue

                # Calculate distance (how far apart vertically)
                distance = step_num.bbox.y0 - parts_list.bbox.y1

                # Calculate proximity score
                max_distance = step_num.bbox.height * DISTANCE_THRESHOLD_MULTIPLIER
                if max_distance > 0:
                    proximity = max(0.0, 1.0 - (distance / max_distance))
                else:
                    proximity = 0.0

                # Calculate alignment score (how well left edges align)
                max_alignment_diff = (
                    step_num.bbox.width * ALIGNMENT_THRESHOLD_MULTIPLIER
                )
                left_diff = abs(parts_list.bbox.x0 - step_num.bbox.x0)
                if max_alignment_diff > 0:
                    alignment = max(0.0, 1.0 - (left_diff / max_alignment_diff))
                else:
                    alignment = 0.0

                # Only create pairing if scores are reasonable
                if proximity > 0.0 or alignment > 0.0:
                    pairings.append(
                        _StepPartsListPairingScore(
                            step_proximity_score=proximity,
                            step_alignment_score=alignment,
                            parts_list=parts_list,
                            step_number=step_num,
                        )
                    )

        return pairings

    def _identify_diagram_region(
        self,
        step_num: StepNumber,
        parts_list: PartsList | None,
        page_data: PageData,
    ) -> BBox:
        """Identify the diagram region for a step.

        The diagram is typically the large area below the step number and parts list.
        For now, we create a simple heuristic-based region.

        Args:
            step_num: The step number
            parts_list: The associated parts list (if any)
            page_data: The page data

        Returns:
            BBox representing the diagram region
        """
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

    # TODO This seems a useful union function for the bbox element.
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

        x0 = min(b.x0 for b in bboxes)
        y0 = min(b.y0 for b in bboxes)
        x1 = max(b.x1 for b in bboxes)
        y1 = max(b.y1 for b in bboxes)

        return BBox(x0=x0, y0=y0, x1=x1, y1=y1)

    def classify(self, page_data: PageData, result: ClassificationResult) -> None:
        """Greedily select the best StepNumber-PartsList pairings and create Steps.

        Uses the pairing scores created in evaluate() to select the best matches.
        Ensures each StepNumber and each PartsList is used at most once.
        """
        # Sort pairing scores by quality (highest first)
        sorted_pairings = sorted(
            self._pairing_scores,
            key=lambda p: p.sort_key(),
            reverse=True,
        )

        # Track which StepNumbers and PartsLists have been used
        used_step_ids: set[int] = set()
        used_parts_list_ids: set[int] = set()
        used_step_values: set[int] = set()  # Track by step VALUE for uniqueness

        # Greedily select pairings
        selected_pairings: dict[int, PartsList | None] = {}  # step_id -> PartsList

        for pairing in sorted_pairings:
            step_id = id(pairing.step_number)
            parts_list_id = id(pairing.parts_list)
            step_value = pairing.step_number.value

            # Skip if this step number value is already used
            if step_value in used_step_values:
                continue

            # Skip if this step or parts_list is already used
            if step_id in used_step_ids or parts_list_id in used_parts_list_ids:
                continue

            # Use this pairing
            selected_pairings[step_id] = pairing.parts_list
            used_step_ids.add(step_id)
            used_parts_list_ids.add(parts_list_id)
            used_step_values.add(step_value)

        # Get all StepNumbers (including those without pairings)
        step_candidates = result.get_candidates("step_number")
        all_steps: list[StepNumber] = []

        for candidate in step_candidates:
            if (
                candidate.is_winner
                and candidate.constructed is not None
                and isinstance(candidate.constructed, StepNumber)
            ):
                all_steps.append(candidate.constructed)

        # Create Step candidates for each StepNumber
        for step_num in all_steps:
            step_id = id(step_num)
            step_value = step_num.value

            # Skip if this step value was already used
            if step_value in used_step_values and step_id not in selected_pairings:
                log.debug(
                    "[step] Skipping step %d - value already used",
                    step_value,
                )
                continue

            # Get the paired PartsList (if any)
            parts_list = selected_pairings.get(step_id)

            # Identify diagram region
            diagram_bbox = self._identify_diagram_region(
                step_num, parts_list, page_data
            )

            # Build Step
            diagram = Diagram(bbox=diagram_bbox)
            constructed = Step(
                bbox=self._compute_step_bbox(step_num, parts_list, diagram),
                step_number=step_num,
                parts_list=parts_list or PartsList(bbox=step_num.bbox, parts=[]),
                diagram=diagram,
            )

            # Create and mark Step candidate as winner
            score = _StepScore(
                step_number=step_num,
                has_parts_list=parts_list is not None,
                diagram_area=diagram_bbox.area,
            )

            step_candidate = Candidate(
                bbox=constructed.bbox,
                label="step",
                score=1.0,
                score_details=score,
                constructed=constructed,
                source_block=None,
                failure_reason=None,
                is_winner=True,
            )

            result.add_candidate("step", step_candidate)
            result.mark_winner(step_candidate, constructed)
