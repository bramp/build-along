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

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.parts.parts_list_classifier import (
    _PartsListScore,
)
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.classifier.text import (
    extract_step_number_value,
)
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Diagram,
    PartsList,
    RotationSymbol,
    Step,
    StepNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import Text

log = logging.getLogger(__name__)


class _StepScore(Score):
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

    diagram_candidate: Candidate | None
    """The diagram candidate paired with this step (if any)."""

    diagram_score: float
    """Score for the diagram pairing (0.0-1.0). 0.0 if no diagram."""

    def score(self) -> Weight:
        """Return the overall pairing score."""
        return self.overall_score()

    def overall_score(self) -> float:
        """Calculate overall quality score combining parts list and diagram."""
        parts_list_score = 0.0
        if self.has_parts_list:
            parts_list_score = (
                self.step_proximity_score + self.step_alignment_score
            ) / 2.0

        # Combine parts list score (weight 0.4) and diagram score (weight 0.6)
        # Diagram is more important for step identification
        return 0.4 * parts_list_score + 0.6 * self.diagram_score

    def sort_key(self) -> tuple[float, int]:
        """Return a tuple for sorting candidates.

        We prefer:
        1. Higher overall scores (better StepNumber-PartsList-Diagram match)
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
                return (-self.overall_score(), step_value)

        return (-self.overall_score(), 0)  # Fallback if value cannot be extracted


class StepClassifier(LabelClassifier):
    """Classifier for complete Step structures."""

    output = "step"
    requires = frozenset({"step_number", "parts_list", "diagram", "rotation_symbol"})

    def _score(self, result: ClassificationResult) -> None:
        """Score step pairings and create candidates."""
        page_data = result.page_data

        # Get step number and parts list candidates (not constructed elements)
        step_number_candidates = result.get_scored_candidates(
            "step_number", valid_only=False, exclude_failed=True
        )

        if not step_number_candidates:
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
            len(step_number_candidates),
            len(parts_list_candidates),
        )

        # Create all possible Step candidates for pairings (without diagrams initially)
        all_candidates: list[Candidate] = []
        for step_candidate in step_number_candidates:
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
        # This will assign diagrams as part of the selection process
        deduplicated_candidates = self._deduplicate_and_assign_diagrams(
            all_candidates, result
        )

        # Add the deduplicated candidates to the result
        for candidate in deduplicated_candidates:
            result.add_candidate(candidate)

        log.debug(
            "[step] Created %d deduplicated step candidates (from %d possibilities)",
            len(deduplicated_candidates),
            len(all_candidates),
        )

    def build(self, candidate: Candidate, result: ClassificationResult) -> Step:
        """Construct a Step element from a single candidate."""
        score = candidate.score_details
        assert isinstance(score, _StepScore)

        # Validate and extract step number from parent candidate
        step_num_candidate = score.step_number_candidate

        step_num_elem = result.build(step_num_candidate)
        assert isinstance(step_num_elem, StepNumber)
        step_num = step_num_elem

        # Validate and extract parts list from parent candidate (if present)
        parts_list = None
        if score.parts_list_candidate:
            parts_list_candidate = score.parts_list_candidate
            parts_list_elem = result.build(parts_list_candidate)
            assert isinstance(parts_list_elem, PartsList)
            parts_list = parts_list_elem

        # Get the diagram from the diagram candidate
        diagram = None
        if score.diagram_candidate:
            diagram_elem = result.build(score.diagram_candidate)
            assert isinstance(diagram_elem, Diagram)
            diagram = diagram_elem

        # Get rotation symbols near this step (if any)
        rotation_symbol = self._get_rotation_symbol_for_step(step_num, diagram, result)

        # Build Step
        return Step(
            bbox=self._compute_step_bbox(step_num, parts_list, diagram),
            step_number=step_num,
            parts_list=parts_list,
            diagram=diagram,
            rotation_symbol=rotation_symbol,
        )

    def _create_step_candidate(
        self,
        step_candidate: Candidate,
        parts_list_candidate: Candidate | None,
        result: ClassificationResult,
    ) -> Candidate | None:
        """Create a Step candidate (without diagram assignment).

        Diagrams will be assigned later during greedy selection to ensure
        each diagram is matched with the best step.

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

        # Diagram will be assigned later during greedy selection
        # Create score object with candidate references (no diagram yet)
        score = _StepScore(
            step_number_candidate=step_candidate,
            parts_list_candidate=parts_list_candidate,
            has_parts_list=parts_list_candidate is not None,
            step_proximity_score=proximity_score,
            step_alignment_score=alignment_score,
            diagram_candidate=None,
            diagram_score=0.0,
        )

        # Calculate combined bbox for the candidate (without diagram for now)
        combined_bbox = step_bbox
        if parts_list_bbox:
            combined_bbox = BBox.union(combined_bbox, parts_list_bbox)

        # Create candidate
        return Candidate(
            bbox=combined_bbox,
            label="step",
            score=score.overall_score(),
            score_details=score,
            source_blocks=[],
        )

    def _find_best_diagram(
        self,
        step_bbox: BBox,
        parts_list_bbox: BBox | None,
        result: ClassificationResult,
        used_diagrams: set[int],
    ) -> tuple[Candidate | None, float]:
        """Find the best diagram candidate for this step.

        Scores all diagram candidates based on:
        - Position relative to step (prefers right and below, allows slight left offset)
        - Distance from step (prefers closer)
        - Avoids diagrams already used by other steps

        Args:
            step_bbox: The step number bbox
            parts_list_bbox: The associated parts list bbox (if any)
            result: Classification result containing diagram candidates
            used_diagrams: Set of diagram candidate IDs already assigned to steps

        Returns:
            Tuple of (best matching diagram candidate, score) or (None, 0.0)
        """
        # Get all diagram candidates
        diagram_candidates = result.get_scored_candidates(
            "diagram", valid_only=False, exclude_failed=True
        )

        if not diagram_candidates:
            log.debug("[step] No diagram candidates available")
            return None, 0.0

        # Determine the reference point (bottom-right of step header)
        ref_y = step_bbox.y1
        ref_x = step_bbox.x1
        if parts_list_bbox:
            ref_y = max(ref_y, parts_list_bbox.y1)
            ref_x = max(ref_x, parts_list_bbox.x1)

        log.debug(
            "[step] Finding diagram for step at (%s,%s) ref=(%s,%s)",
            step_bbox.x0,
            step_bbox.y0,
            ref_x,
            ref_y,
        )

        best_candidate = None
        best_score = 0.0

        # TODO There are a lot of magic numbers here that could be tuned / removed.

        for candidate in diagram_candidates:
            # Skip diagrams already used by other steps
            if id(candidate) in used_diagrams:
                continue

            # Calculate position scores
            diagram_bbox = candidate.bbox

            # Horizontal position score
            # Prefer diagrams to the right, but allow diagrams to the left as well
            # Allow up to 200 points to the left (about half a page)
            left_tolerance = 200.0
            x_offset = diagram_bbox.x0 - ref_x

            if x_offset >= -left_tolerance:
                # Diagram starts to the right or within tolerance to the left
                # Score decreases as we go further left
                if x_offset >= 0:
                    x_score = 1.0  # Perfect: to the right
                else:
                    # Linearly decrease from 1.0 to 0.5 as we go left
                    x_score = 1.0 + (x_offset / left_tolerance) * 0.5
            else:
                # Too far to the left
                x_score = 0.0

            # Vertical position score
            # Prefer diagrams below the step header
            y_offset = diagram_bbox.y0 - ref_y

            # Allow diagrams that start slightly above (overlapping header)
            if y_offset >= -50.0:
                # Below or slightly overlapping
                # Score decreases with distance
                # Penalize being above less than being far below
                distance = abs(y_offset) if y_offset >= 0 else abs(y_offset) * 0.5
                max_distance = 200.0  # Maximum reasonable distance
                y_score = max(0.0, 1.0 - (distance / max_distance))
            else:
                # Diagram is too far above the step header - very bad
                y_score = 0.0

            # Combined score (both must be reasonable)
            if x_score > 0.0 and y_score > 0.0:
                # Weight vertical position more heavily (0.6) than horizontal (0.4)
                score = 0.4 * x_score + 0.6 * y_score

                log.debug(
                    "[step]   Diagram at (%s,%s) x_offset=%s y_offset=%s "
                    "x_score=%.2f y_score=%.2f score=%.2f",
                    diagram_bbox.x0,
                    diagram_bbox.y0,
                    x_offset,
                    y_offset,
                    x_score,
                    y_score,
                    score,
                )

                if score > best_score:
                    best_score = score
                    best_candidate = candidate

        if best_candidate:
            log.debug(
                "[step] Best diagram at %s with score %.2f",
                best_candidate.bbox,
                best_score,
            )
        else:
            log.debug("[step] No suitable diagram found")

        return best_candidate, best_score

    def _get_rotation_symbol_for_step(
        self,
        step_num: StepNumber,
        diagram: Diagram | None,
        result: ClassificationResult,
    ) -> RotationSymbol | None:
        """Find rotation symbol associated with this step.

        Looks for rotation symbol candidates that are positioned near the
        step's diagram or step number. Returns the highest-scored candidate
        if multiple are found.

        Args:
            step_num: The step number element
            diagram: The diagram element (if any)
            result: Classification result containing rotation symbol candidates

        Returns:
            Single RotationSymbol element for this step, or None if not found
        """
        rotation_symbol_candidates = result.get_scored_candidates(
            "rotation_symbol", valid_only=False, exclude_failed=True
        )

        log.debug(
            "[step] Looking for rotation symbols for step %d, found %d candidates",
            step_num.value,
            len(rotation_symbol_candidates),
        )

        if not rotation_symbol_candidates:
            return None

        # Determine search region: prefer diagram area, fallback to step area
        search_bbox = diagram.bbox if diagram else step_num.bbox

        # Expand search region to catch nearby symbols
        search_region = BBox(
            x0=search_bbox.x0 - 50,
            y0=search_bbox.y0 - 50,
            x1=search_bbox.x1 + 50,
            y1=search_bbox.y1 + 50,
        )

        log.debug(
            "[step] Search region for step %d: %s",
            step_num.value,
            search_region,
        )

        # Find rotation symbols within or overlapping the search region
        # Keep track of best candidate by score
        best_candidate = None
        best_score = 0.0
        for candidate in rotation_symbol_candidates:
            overlaps = candidate.bbox.overlaps(search_region)
            log.debug(
                "[step]   Candidate at %s, overlaps=%s, score=%.2f",
                candidate.bbox,
                overlaps,
                candidate.score,
            )
            if overlaps and candidate.score > best_score:
                best_candidate = candidate
                best_score = candidate.score

        if best_candidate:
            rotation_symbol = result.build(best_candidate)
            assert isinstance(rotation_symbol, RotationSymbol)
            log.debug(
                "[step] Found rotation symbol for step %d (score=%.2f)",
                step_num.value,
                best_score,
            )
            return rotation_symbol

        log.debug("[step] No rotation symbol found for step %d", step_num.value)
        return None

    def _compute_step_bbox(
        self,
        step_num: StepNumber,
        parts_list: PartsList | None,
        diagram: Diagram | None,
    ) -> BBox:
        """Compute the overall bounding box for the Step.

        This encompasses the step number, parts list (if any), and diagram (if any).

        Args:
            step_num: The step number element
            parts_list: The parts list (if any)
            diagram: The diagram element (if any)

        Returns:
            Combined bounding box
        """
        bboxes = [step_num.bbox]
        if parts_list:
            bboxes.append(parts_list.bbox)
        if diagram:
            bboxes.append(diagram.bbox)

        return BBox.union_all(bboxes)

    def _deduplicate_and_assign_diagrams(
        self, candidates: list[Candidate], result: ClassificationResult
    ) -> list[Candidate]:
        """Greedily select the best Step candidates and assign diagrams.

        For each selected step, finds the best available diagram and updates
        the candidate's score. Ensures each StepNumber value, PartsList, and
        Diagram is used at most once.

        Args:
            candidates: All possible Step candidates (without diagrams)
            result: Classification result containing diagram candidates

        Returns:
            Deduplicated list of Step candidates with diagrams assigned
        """
        # Sort candidates by parts_list score initially
        # (we'll resort after assigning diagrams)
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (
                c.score_details.sort_key()
                if isinstance(c.score_details, _StepScore)
                else (0.0, 0)
            ),
        )

        # Track which elements have been used
        used_step_values: set[int] = set()
        used_parts_list_ids: set[int] = set()
        used_diagrams: set[int] = set()
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
                if isinstance(parts_list_candidate.score_details, _PartsListScore):
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

            # Find and assign the best available diagram for this step
            step_bbox = step_num_candidate.bbox
            parts_list_bbox = (
                parts_list_candidate.bbox if parts_list_candidate else None
            )
            diagram_candidate, diagram_score = self._find_best_diagram(
                step_bbox, parts_list_bbox, result, used_diagrams
            )

            # Update the score with the diagram assignment
            updated_score = _StepScore(
                step_number_candidate=score.step_number_candidate,
                parts_list_candidate=score.parts_list_candidate,
                has_parts_list=score.has_parts_list,
                step_proximity_score=score.step_proximity_score,
                step_alignment_score=score.step_alignment_score,
                diagram_candidate=diagram_candidate,
                diagram_score=diagram_score,
            )

            # Update the candidate's bbox to include the diagram
            updated_bbox = candidate.bbox
            if diagram_candidate:
                updated_bbox = BBox.union(updated_bbox, diagram_candidate.bbox)

            # Create updated candidate with diagram
            updated_candidate = Candidate(
                bbox=updated_bbox,
                label=candidate.label,
                score=updated_score.overall_score(),
                score_details=updated_score,
                source_blocks=candidate.source_blocks,
            )

            # Select this candidate
            selected.append(updated_candidate)
            used_step_values.add(step_value)

            # Mark the diagram as used
            if diagram_candidate:
                used_diagrams.add(id(diagram_candidate))

            log.debug(
                "[step] Selected step %d (parts_list=%s, diagram=%s, score=%.2f)",
                step_value,
                "yes" if parts_list_candidate is not None else "no",
                "yes" if diagram_candidate is not None else "no",
                updated_score.overall_score(),
            )

        return selected
