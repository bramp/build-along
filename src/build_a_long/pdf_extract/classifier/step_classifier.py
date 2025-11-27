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
from build_a_long.pdf_extract.classifier.text_extractors import (
    extract_step_number_value,
)
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Diagram,
    LegoPageElements,
    PartsList,
    Step,
    StepNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import Text

log = logging.getLogger(__name__)


@dataclass
class _StepScore:
    """Internal score representation for step classification."""

    step_number_candidate: Candidate
    """The step number candidate this step is associated with."""

    parts_list_candidate: Candidate | None
    """The parts list candidate paired with this step (if any)."""

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
        # Extract step number value from candidate's source block
        step_num_candidate = self.step_number_candidate

        # Assume single source block for step number
        if step_num_candidate.source_blocks and isinstance(
            step_num_candidate.source_blocks[0], Text
        ):
            text_block = step_num_candidate.source_blocks[0]
            step_value = extract_step_number_value(text_block.text)
            if step_value is not None:
                return (-self.pairing_score(), step_value)

        return (-self.pairing_score(), 0)  # Fallback if value cannot be extracted


@dataclass(frozen=True)
class StepClassifier(LabelClassifier):
    """Classifier for complete Step structures."""

    outputs = frozenset({"step"})
    requires = frozenset({"step_number", "parts_list"})

    def _score(self, result: ClassificationResult) -> None:
        """Score step pairings and create candidates WITHOUT construction."""
        page_data = result.page_data

        # Get step number and parts list candidates (not constructed elements)
        step_candidates = result.get_scored_candidates(
            "step_number", valid_only=False, exclude_failed=True
        )

        if not step_candidates:
            return

        # Get parts_list candidates
        parts_list_candidates = result.get_scored_candidates(
            "parts_list",
            valid_only=False,
            exclude_failed=True,
        )

        log.debug(
            "[step] page=%s step_candidates=%d parts_list_candidates=%d",
            page_data.page_number,
            len(step_candidates),
            len(parts_list_candidates),
        )

        # Create all possible Step candidates for pairings
        all_candidates: list[Candidate] = []
        for step_candidate in step_candidates:
            # Create candidates for this StepNumber paired with each PartsList
            for parts_list_candidate in parts_list_candidates:
                candidate = self._create_step_candidate(
                    step_candidate, parts_list_candidate, result
                )
                if candidate:
                    all_candidates.append(candidate)

            # Also create a candidate with no PartsList (fallback)
            candidate = self._create_step_candidate(step_candidate, None, result)
            if candidate:
                all_candidates.append(candidate)

        # Greedily select the best candidates (deduplication)
        deduplicated_candidates = self._deduplicate_candidates(all_candidates)

        # Add the deduplicated candidates to the result
        for candidate in deduplicated_candidates:
            result.add_candidate("step", candidate)

        log.debug(
            "[step] Created %d deduplicated step candidates (from %d possibilities)",
            len(deduplicated_candidates),
            len(all_candidates),
        )

    def construct(self, result: ClassificationResult) -> None:
        """Construct Step elements from candidates."""
        candidates = result.get_candidates("step")
        for candidate in candidates:
            try:
                elem = self.construct_candidate(candidate, result)
                candidate.constructed = elem
            except Exception as e:
                candidate.failure_reason = str(e)

    def construct_candidate(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        """Construct a Step element from a single candidate."""
        score = candidate.score_details
        assert isinstance(score, _StepScore)

        # Validate and extract step number from parent candidate
        step_num_candidate = score.step_number_candidate

        step_num_elem = result.construct_candidate(step_num_candidate)
        assert isinstance(step_num_elem, StepNumber)
        step_num = step_num_elem

        # Validate and extract parts list from parent candidate (if present)
        parts_list = None
        if score.parts_list_candidate:
            parts_list_candidate = score.parts_list_candidate
            parts_list_elem = result.construct_candidate(parts_list_candidate)
            assert isinstance(parts_list_elem, PartsList)
            parts_list = parts_list_elem

        # Identify diagram region
        diagram_bbox = self._identify_diagram_region(
            step_num.bbox, parts_list.bbox if parts_list else None, result
        )

        # Build Step
        diagram = Diagram(bbox=diagram_bbox)
        return Step(
            bbox=self._compute_step_bbox(step_num, parts_list, diagram),
            step_number=step_num,
            parts_list=parts_list or PartsList(bbox=step_num.bbox, parts=[]),
            diagram=diagram,
        )

    def _create_step_candidate(
        self,
        step_candidate: Candidate,
        parts_list_candidate: Candidate | None,
        result: ClassificationResult,
    ) -> Candidate | None:
        """Create a Step candidate WITHOUT construction.

        Args:
            step_candidate: The StepNumber candidate for this step
            parts_list_candidate: The PartsList candidate to pair with (or None)
            result: Classification result

        Returns:
            The created Candidate with score but no construction
        """
        ABOVE_EPS = 2.0  # Small epsilon for "above" check
        ALIGNMENT_THRESHOLD_MULTIPLIER = 1.0  # Max horizontal offset
        DISTANCE_THRESHOLD_MULTIPLIER = 1.0  # Max vertical distance

        step_bbox = step_candidate.bbox
        parts_list_bbox = parts_list_candidate.bbox if parts_list_candidate else None

        # Calculate pairing scores if there's a parts_list above the step
        proximity_score = 0.0
        alignment_score = 0.0

        if (
            parts_list_bbox is not None
            and parts_list_bbox.y1 <= step_bbox.y0 + ABOVE_EPS
        ):
            # Calculate distance (how far apart vertically)
            distance = step_bbox.y0 - parts_list_bbox.y1

            # Calculate proximity score
            max_distance = step_bbox.height * DISTANCE_THRESHOLD_MULTIPLIER
            if max_distance > 0:
                proximity_score = max(0.0, 1.0 - (distance / max_distance))

            # Calculate alignment score (how well left edges align)
            max_alignment_diff = step_bbox.width * ALIGNMENT_THRESHOLD_MULTIPLIER
            left_diff = abs(parts_list_bbox.x0 - step_bbox.x0)
            if max_alignment_diff > 0:
                alignment_score = max(0.0, 1.0 - (left_diff / max_alignment_diff))

        # Estimate diagram bbox for scoring purposes
        diagram_bbox = self._identify_diagram_region(step_bbox, parts_list_bbox, result)

        # Create score object with candidate references
        score = _StepScore(
            step_number_candidate=step_candidate,
            parts_list_candidate=parts_list_candidate,
            has_parts_list=parts_list_candidate is not None,
            step_proximity_score=proximity_score,
            step_alignment_score=alignment_score,
            diagram_area=diagram_bbox.area,
        )

        # Calculate combined bbox for the candidate
        bboxes = [step_bbox, diagram_bbox]
        if parts_list_bbox:
            bboxes.append(parts_list_bbox)
        combined_bbox = BBox.union_all(bboxes)

        # Create candidate WITHOUT construction
        return Candidate(
            bbox=combined_bbox,
            label="step",
            score=score.pairing_score(),
            score_details=score,
            constructed=None,
            source_blocks=[],
            failure_reason=None,
        )

    def _identify_diagram_region(
        self,
        step_bbox: BBox,
        parts_list_bbox: BBox | None,
        result: ClassificationResult,
    ) -> BBox:
        """Identify the diagram region for a step.

        The diagram is typically the large area below the step number and parts list.
        For now, we create a simple heuristic-based region.

        Args:
            step_bbox: The step number bbox
            parts_list_bbox: The associated parts list bbox (if any)
            result: Classification result containing page_data

        Returns:
            BBox representing the diagram region
        """
        page_data = result.page_data
        # Simple heuristic: use the step number's bbox as a starting point
        # In the future, we should look for actual drawing elements below the step

        # Start with step number position
        x0 = step_bbox.x0
        y0 = step_bbox.y1  # Below the step number

        # If there's a parts list, the diagram should be below it
        if parts_list_bbox:
            y0 = max(y0, parts_list_bbox.y1)

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

    def _deduplicate_candidates(self, candidates: list[Candidate]) -> list[Candidate]:
        """Greedily select the best Step candidates.

        Ensures each StepNumber value and each PartsList is used at most once.

        Args:
            candidates: All possible Step candidates

        Returns:
            Deduplicated list of Step candidates
        """
        # Sort candidates by score (highest first)
        sorted_candidates = sorted(
            candidates,
            key=lambda c: c.score_details.sort_key(),
        )

        # Track which StepNumber values and PartsLists have been used
        used_step_values: set[int] = set()
        used_parts_list_ids: set[int] = set()
        selected: list[Candidate] = []

        # Greedily select winners
        for candidate in sorted_candidates:
            # Get step info from score_details (candidates not yet constructed)
            assert isinstance(candidate.score_details, _StepScore)
            score = candidate.score_details

            # Extract step number value from parent candidate source block
            step_num_candidate = score.step_number_candidate

            # Extract step value from text block
            if not step_num_candidate.source_blocks:
                continue
            text_block = step_num_candidate.source_blocks[0]
            if not isinstance(text_block, Text):
                continue

            step_value = extract_step_number_value(text_block.text)
            if step_value is None:
                continue

            # Extract parts list from parent candidate (if present)
            parts_list_candidate = score.parts_list_candidate

            # Skip if this step number value is already used
            if step_value in used_step_values:
                log.debug(
                    "[step] Skipping candidate for step %d - value already used",
                    step_value,
                )
                continue

            # Skip if this parts_list is already used (if it has parts)
            if parts_list_candidate is not None:
                # Check if parts list has parts (look at its score details)
                has_parts = False
                if hasattr(parts_list_candidate.score_details, "part_candidates"):
                    has_parts = (
                        len(parts_list_candidate.score_details.part_candidates) > 0
                    )

                if has_parts:
                    parts_list_id = id(parts_list_candidate)
                    if parts_list_id in used_parts_list_ids:
                        log.debug(
                            "[step] Skipping candidate for step %d - "
                            "PartsList candidate already used",
                            step_value,
                        )
                        continue
                    # Claim this parts_list
                    used_parts_list_ids.add(parts_list_id)

            # Select this candidate
            selected.append(candidate)
            used_step_values.add(step_value)

            log.debug(
                "[step] Selected step %d (parts_list=%s, pairing_score=%.2f)",
                step_value,
                "yes" if parts_list_candidate is not None else "no",
                score.pairing_score(),
            )

        return selected
