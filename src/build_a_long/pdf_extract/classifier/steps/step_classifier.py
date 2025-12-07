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

import numpy as np
from scipy.optimize import linear_sum_assignment

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    CandidateFailedError,
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.parts.parts_list_classifier import (
    _PartsListScore,
)
from build_a_long.pdf_extract.classifier.score import (
    Score,
    Weight,
    find_best_scoring,
)
from build_a_long.pdf_extract.classifier.text import (
    extract_step_number_value,
)
from build_a_long.pdf_extract.extractor.bbox import BBox, filter_overlapping
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Arrow,
    Diagram,
    PartsList,
    RotationSymbol,
    Step,
    StepNumber,
    SubAssembly,
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

    def score(self) -> Weight:
        """Return the overall pairing score."""
        return self.overall_score()

    def overall_score(self) -> float:
        """Calculate overall quality score based on parts list pairing.

        Steps with a parts_list are given a bonus to prefer them over
        steps without parts_list. Diagrams are found at build time, not
        during scoring, to allow rotation symbols to claim small images first.
        """
        if self.has_parts_list:
            # Base score for having parts_list + proximity/alignment bonus
            parts_list_bonus = 0.5
            pairing_score = (
                self.step_proximity_score + self.step_alignment_score
            ) / 2.0
            return parts_list_bonus + 0.5 * pairing_score
        return 0.3  # Lower base score for steps without parts list

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
    requires = frozenset(
        {"step_number", "parts_list", "diagram", "rotation_symbol", "subassembly"}
    )

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

        # Find and build the diagram for this step first
        diagram = self._find_and_build_diagram_for_step(step_num, parts_list, result)

        # Compute the step's bbox (step_num + parts_list + diagram)
        page_bbox = result.page_data.bbox
        step_bbox = self._compute_step_bbox(step_num, parts_list, diagram, page_bbox)

        # Get arrows for this step (from subassemblies and other sources)
        arrows = self._get_arrows_for_step(step_num, diagram, result)

        # Get subassemblies for this step
        subassemblies = self._get_subassemblies_for_step(step_num, diagram, result)

        # Build Step (rotation_symbol will be assigned later)
        return Step(
            bbox=step_bbox,
            step_number=step_num,
            parts_list=parts_list,
            diagram=diagram,
            rotation_symbol=None,
            arrows=arrows,
            subassemblies=subassemblies,
        )

    def _find_and_build_diagram_for_step(
        self,
        step_num: StepNumber,
        parts_list: PartsList | None,
        result: ClassificationResult,
    ) -> Diagram | None:
        """Find and build the best diagram for this step.

        This is called at build time, after rotation symbols have been built,
        so they've already claimed any small images they need. This ensures
        the diagram doesn't incorrectly cluster rotation symbol images.

        Args:
            step_num: The built step number element
            parts_list: The built parts list element (if any)
            result: Classification result containing diagram candidates

        Returns:
            The built Diagram element, or None if no suitable diagram found
        """
        # Get all non-failed, non-constructed diagram candidates
        diagram_candidates = result.get_scored_candidates(
            "diagram", valid_only=False, exclude_failed=True
        )

        # Filter to only candidates that haven't been built yet
        available_candidates = [c for c in diagram_candidates if c.constructed is None]

        if not available_candidates:
            log.debug(
                "[step] No diagram candidates available for step %d",
                step_num.value,
            )
            return None

        # Score each candidate based on position relative to step
        step_bbox = step_num.bbox
        best_candidate = None
        best_score = -float("inf")

        for candidate in available_candidates:
            score = self._score_step_diagram_pair(step_bbox, candidate.bbox)

            log.debug(
                "[step] Diagram candidate at %s for step %d: score=%.2f",
                candidate.bbox,
                step_num.value,
                score,
            )

            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None or best_score < 0.2:
            log.debug(
                "[step] No suitable diagram found for step %d (best_score=%.2f)",
                step_num.value,
                best_score,
            )
            return None

        # Build the diagram
        try:
            diagram_elem = result.build(best_candidate)
            assert isinstance(diagram_elem, Diagram)
            log.debug(
                "[step] Built diagram at %s for step %d (score=%.2f)",
                diagram_elem.bbox,
                step_num.value,
                best_score,
            )
            return diagram_elem
        except CandidateFailedError as e:
            log.debug(
                "[step] Failed to build diagram for step %d: %s",
                step_num.value,
                e,
            )
            return None

    def _create_step_candidate(
        self,
        step_candidate: Candidate,
        parts_list_candidate: Candidate | None,
        result: ClassificationResult,
    ) -> Candidate | None:
        """Create a Step candidate (without diagram assignment).

        Diagrams are found at build time, not during scoring, to allow
        rotation symbols to claim small images first.

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

        # Create score object with candidate references
        # Diagrams are found at build time, not during scoring
        score = _StepScore(
            step_number_candidate=step_candidate,
            parts_list_candidate=parts_list_candidate,
            has_parts_list=parts_list_candidate is not None,
            step_proximity_score=proximity_score,
            step_alignment_score=alignment_score,
        )

        # Calculate combined bbox for the candidate (without diagram)
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

    def _score_step_diagram_pair(
        self,
        step_bbox: BBox,
        diagram_bbox: BBox,
    ) -> float:
        """Score how well a diagram matches a step.

        Diagrams are typically positioned to the right of and/or below the step
        number. This method scores based on:
        - Horizontal position: prefer diagrams to the right, penalize left
        - Vertical position: prefer diagrams below the step header
        - Distance: closer is better

        Args:
            step_bbox: The step number bounding box
            diagram_bbox: The diagram bounding box to score

        Returns:
            Score between 0.0 and 1.0 (higher is better match)
        """
        # Reference point: bottom-right of step number
        ref_x = step_bbox.x1
        ref_y = step_bbox.y1

        # TODO Move all these constants into config, or make them adaptive?

        # Horizontal score
        # Diagrams to the right are preferred, but allow some overlap
        x_offset = diagram_bbox.x0 - ref_x

        if x_offset >= -50:
            # Diagram starts to the right or slightly overlapping - good
            # Score decreases slightly with distance to the right
            x_score = max(0.5, 1.0 - abs(x_offset) / 400.0)
        elif x_offset >= -200:
            # Diagram is moderately to the left - acceptable
            x_score = 0.3 + 0.2 * (1.0 + x_offset / 200.0)
        else:
            # Diagram is far to the left - poor match
            x_score = max(0.1, 0.3 + x_offset / 400.0)

        # Vertical score
        # Diagrams below the step header are preferred
        y_offset = diagram_bbox.y0 - ref_y

        if y_offset >= -30:
            # Diagram starts below or slightly overlapping - good
            # Score decreases with vertical distance
            y_score = max(0.3, 1.0 - abs(y_offset) / 300.0)
        elif y_offset >= -100:
            # Diagram is moderately above - less good but acceptable
            y_score = 0.2 + 0.3 * (1.0 + y_offset / 100.0)
        else:
            # Diagram is far above - poor match
            y_score = max(0.05, 0.2 + y_offset / 300.0)

        # Combined score - weight both dimensions equally
        score = 0.5 * x_score + 0.5 * y_score

        return score

    def _get_rotation_symbol_for_step(
        self,
        step_bbox: BBox,
        result: ClassificationResult,
    ) -> RotationSymbol | None:
        """Find an already-built rotation symbol within this step's area.

        Rotation symbols are built by PageClassifier before steps are processed.
        This method finds the already-built rotation symbol that falls within
        the step's bounding box.

        Args:
            step_bbox: The step's bounding box (including step_num, parts_list,
                and diagram)
            result: Classification result containing rotation symbol candidates

        Returns:
            Single RotationSymbol element for this step, or None if not found
        """
        # Get rotation_symbol candidates - they should already be built
        # by PageClassifier. Use valid_only=True to only get successfully
        # constructed rotation symbols.
        rotation_symbol_candidates = result.get_scored_candidates(
            "rotation_symbol", valid_only=True
        )

        log.debug(
            "[step] Looking for rotation symbols in step bbox %s, "
            "found %d built candidates",
            step_bbox,
            len(rotation_symbol_candidates),
        )

        if not rotation_symbol_candidates:
            return None

        # Find best-scoring rotation symbol overlapping the step's bbox
        overlapping_candidates = filter_overlapping(
            rotation_symbol_candidates, step_bbox
        )
        best_candidate = find_best_scoring(overlapping_candidates)

        if best_candidate and best_candidate.constructed is not None:
            rotation_symbol = best_candidate.constructed
            assert isinstance(rotation_symbol, RotationSymbol)
            log.debug(
                "[step] Found rotation symbol at %s (score=%.2f)",
                rotation_symbol.bbox,
                best_candidate.score,
            )
            return rotation_symbol

        log.debug("[step] No rotation symbol found in step bbox %s", step_bbox)
        return None

    def _get_arrows_for_step(
        self,
        step_num: StepNumber,
        diagram: Diagram | None,
        result: ClassificationResult,
    ) -> list[Arrow]:
        """Find arrows associated with this step.

        Looks for arrow candidates that are positioned near the step's diagram
        or step number. Typically these are arrows pointing from subassembly
        callout boxes to the main diagram.

        Args:
            step_num: The step number element
            diagram: The diagram element (if any)
            result: Classification result containing arrow candidates

        Returns:
            List of Arrow elements for this step
        """
        arrow_candidates = result.get_scored_candidates(
            "arrow", valid_only=False, exclude_failed=True
        )

        log.debug(
            "[step] Looking for arrows for step %d, found %d candidates",
            step_num.value,
            len(arrow_candidates),
        )

        if not arrow_candidates:
            return []

        # Determine search region: prefer diagram area, fallback to step area
        search_bbox = diagram.bbox if diagram else step_num.bbox

        # Expand search region to catch arrows near the diagram
        # Use a larger margin than rotation symbols since arrows can extend further
        search_region = search_bbox.expand(100.0)

        log.debug(
            "[step] Arrow search region for step %d: %s",
            step_num.value,
            search_region,
        )

        # Find arrows within or overlapping the search region
        arrows: list[Arrow] = []
        overlapping_candidates = filter_overlapping(arrow_candidates, search_region)

        for candidate in overlapping_candidates:
            log.debug(
                "[step]   Arrow candidate at %s, overlaps=True, score=%.2f",
                candidate.bbox,
                candidate.score,
            )
            try:
                arrow = result.build(candidate)
                assert isinstance(arrow, Arrow)
                arrows.append(arrow)
            except CandidateFailedError:
                # Arrow lost conflict to another arrow (they share source blocks)
                # This is expected when multiple arrows overlap - skip it
                log.debug(
                    "[step]   Arrow candidate at %s failed (conflict), skipping",
                    candidate.bbox,
                )
                continue

        log.debug(
            "[step] Found %d arrows for step %d",
            len(arrows),
            step_num.value,
        )
        return arrows

    def _get_subassemblies_for_step(
        self,
        step_num: StepNumber,
        diagram: Diagram | None,
        result: ClassificationResult,
    ) -> list[SubAssembly]:
        """Find subassemblies associated with this step.

        Looks for subassembly candidates that are positioned near the step's
        diagram or step number. SubAssemblies are callout boxes showing
        sub-assemblies.

        Args:
            step_num: The step number element
            diagram: The diagram element (if any)
            result: Classification result containing subassembly candidates

        Returns:
            List of SubAssembly elements for this step
        """
        subassembly_candidates = result.get_scored_candidates(
            "subassembly", valid_only=False, exclude_failed=True
        )

        log.debug(
            "[step] Looking for subassemblies for step %d, found %d candidates",
            step_num.value,
            len(subassembly_candidates),
        )

        if not subassembly_candidates:
            return []

        # Determine search region: prefer diagram area, fallback to step area
        search_bbox = diagram.bbox if diagram else step_num.bbox

        # Expand search region to catch subassemblies near the diagram
        # Use a larger margin since subassemblies can be positioned further from
        # the main diagram
        search_region = search_bbox.expand(150.0)

        log.debug(
            "[step] SubAssembly search region for step %d: %s",
            step_num.value,
            search_region,
        )

        # Find subassemblies within or overlapping the search region
        subassemblies: list[SubAssembly] = []
        overlapping_candidates = filter_overlapping(
            subassembly_candidates, search_region
        )

        for candidate in overlapping_candidates:
            log.debug(
                "[step]   SubAssembly candidate at %s, overlaps=True, score=%.2f",
                candidate.bbox,
                candidate.score,
            )
            try:
                subassembly = result.build(candidate)
                assert isinstance(subassembly, SubAssembly)
                subassemblies.append(subassembly)
            except Exception as e:
                log.debug(
                    "[step]   Failed to build subassembly at %s: %s",
                    candidate.bbox,
                    e,
                )

        log.debug(
            "[step] Found %d subassemblies for step %d",
            len(subassemblies),
            step_num.value,
        )
        return subassemblies

    def _compute_step_bbox(
        self,
        step_num: StepNumber,
        parts_list: PartsList | None,
        diagram: Diagram | None,
        page_bbox: BBox,
    ) -> BBox:
        """Compute the overall bounding box for the Step.

        This encompasses the step number, parts list (if any), and diagram (if any).
        The result is clipped to the page bounds to handle elements that extend
        slightly off-page (e.g., arrows in diagrams).

        Args:
            step_num: The step number element
            parts_list: The parts list (if any)
            diagram: The diagram element (if any)
            page_bbox: The page bounding box to clip to

        Returns:
            Combined bounding box, clipped to page bounds
        """
        bboxes = [step_num.bbox]
        if parts_list:
            bboxes.append(parts_list.bbox)
        if diagram:
            bboxes.append(diagram.bbox)

        return BBox.union_all(bboxes).clip_to(page_bbox)

    def _deduplicate_and_assign_diagrams(
        self, candidates: list[Candidate], result: ClassificationResult
    ) -> list[Candidate]:
        """Select the best Step candidates, ensuring each step number is unique.

        Diagrams are found at build time, not during scoring, to allow
        rotation symbols to claim small images first.

        Args:
            candidates: All possible Step candidates
            result: Classification result (unused, kept for API compatibility)

        Returns:
            Deduplicated list of Step candidates (one per step number value)
        """
        # First, deduplicate candidates by step number value
        # Pick the best candidate for each unique step number
        best_by_step_value: dict[int, Candidate] = {}

        for candidate in candidates:
            assert isinstance(candidate.score_details, _StepScore)
            score = candidate.score_details

            # Extract step number value
            step_num_candidate = score.step_number_candidate
            if not step_num_candidate.source_blocks:
                continue
            text_block = step_num_candidate.source_blocks[0]
            if not isinstance(text_block, Text):
                continue

            step_value = extract_step_number_value(text_block.text)
            if step_value is None:
                continue

            # Keep the best candidate for each step value
            if step_value not in best_by_step_value:
                best_by_step_value[step_value] = candidate
            else:
                existing = best_by_step_value[step_value]
                if candidate.score > existing.score:
                    best_by_step_value[step_value] = candidate

        # Get unique step candidates
        unique_step_candidates = list(best_by_step_value.values())

        if not unique_step_candidates:
            return []

        # Build final candidates ensuring parts list uniqueness
        selected: list[Candidate] = []
        used_parts_list_ids: set[int] = set()

        for candidate in unique_step_candidates:
            assert isinstance(candidate.score_details, _StepScore)
            score = candidate.score_details

            # Check parts list uniqueness
            parts_list_candidate = score.parts_list_candidate
            if parts_list_candidate is not None:
                has_parts = False
                if isinstance(parts_list_candidate.score_details, _PartsListScore):
                    has_parts = (
                        len(parts_list_candidate.score_details.part_candidates) > 0
                    )

                if has_parts:
                    parts_list_id = id(parts_list_candidate)
                    if parts_list_id in used_parts_list_ids:
                        # Use None for parts list if already used
                        parts_list_candidate = None
                    else:
                        used_parts_list_ids.add(parts_list_id)

            # Create updated score if parts_list changed
            if parts_list_candidate != score.parts_list_candidate:
                updated_score = _StepScore(
                    step_number_candidate=score.step_number_candidate,
                    parts_list_candidate=parts_list_candidate,
                    has_parts_list=parts_list_candidate is not None,
                    step_proximity_score=score.step_proximity_score,
                    step_alignment_score=score.step_alignment_score,
                )
                candidate = Candidate(
                    bbox=candidate.bbox,
                    label=candidate.label,
                    score=updated_score.overall_score(),
                    score_details=updated_score,
                    source_blocks=candidate.source_blocks,
                )

            selected.append(candidate)

            # Log selection
            text_block = score.step_number_candidate.source_blocks[0]
            assert isinstance(text_block, Text)
            step_value = extract_step_number_value(text_block.text)
            log.debug(
                "[step] Selected step %d (parts_list=%s, score=%.2f)",
                step_value or 0,
                "yes" if parts_list_candidate is not None else "no",
                candidate.score,
            )

        return selected


def assign_rotation_symbols_to_steps(
    steps: list[Step],
    rotation_symbols: list[RotationSymbol],
    max_distance: float = 300.0,
) -> None:
    """Assign rotation symbols to steps using Hungarian algorithm.

    Uses optimal bipartite matching to pair rotation symbols with steps based on
    minimum distance from the rotation symbol to the step's diagram (or step bbox
    center if no diagram).

    This function mutates the Step objects in place, setting their rotation_symbol
    attribute.

    Args:
        steps: List of Step objects to assign rotation symbols to
        rotation_symbols: List of RotationSymbol objects to assign
        max_distance: Maximum distance for a valid assignment. Pairs with distance
            greater than this will not be matched.
    """
    if not steps or not rotation_symbols:
        log.debug(
            "[step] No rotation symbol assignment needed: "
            "steps=%d, rotation_symbols=%d",
            len(steps),
            len(rotation_symbols),
        )
        return

    n_steps = len(steps)
    n_symbols = len(rotation_symbols)

    log.debug(
        "[step] Running Hungarian matching: %d steps, %d rotation symbols",
        n_steps,
        n_symbols,
    )

    # Build cost matrix: rows = rotation symbols, cols = steps
    # Cost = distance from rotation symbol center to step's diagram center
    # (or step bbox center if no diagram)
    cost_matrix = np.zeros((n_symbols, n_steps))

    for i, rs in enumerate(rotation_symbols):
        rs_center = rs.bbox.center
        for j, step in enumerate(steps):
            # Use diagram center if available, otherwise step bbox center
            if step.diagram:
                target_center = step.diagram.bbox.center
            else:
                target_center = step.bbox.center

            # Euclidean distance
            distance = (
                (rs_center[0] - target_center[0]) ** 2
                + (rs_center[1] - target_center[1]) ** 2
            ) ** 0.5

            cost_matrix[i, j] = distance
            log.debug(
                "[step]   Distance from rotation_symbol[%d] at %s to step[%d]: %.1f",
                i,
                rs.bbox,
                step.step_number.value,
                distance,
            )

    # Apply max_distance threshold by setting costs above threshold to a high value
    # This prevents assignments that are too far apart
    high_cost = max_distance * 10  # Arbitrary large value
    cost_matrix_thresholded = np.where(
        cost_matrix > max_distance, high_cost, cost_matrix
    )

    # Run Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix_thresholded)

    # Assign rotation symbols to steps based on the matching
    for row_idx, col_idx in zip(row_indices, col_indices, strict=True):
        # Check if this assignment is within the max_distance threshold
        if cost_matrix[row_idx, col_idx] <= max_distance:
            rs = rotation_symbols[row_idx]
            step = steps[col_idx]
            step.rotation_symbol = rs
            # Expand step bbox to include the rotation symbol
            step.bbox = step.bbox.union(rs.bbox)
            log.debug(
                "[step] Assigned rotation symbol at %s to step %d "
                "(distance=%.1f, new step bbox=%s)",
                rs.bbox,
                step.step_number.value,
                cost_matrix[row_idx, col_idx],
                step.bbox,
            )
        else:
            log.debug(
                "[step] Skipped assignment: rotation_symbol[%d] to step[%d] "
                "distance=%.1f > max_distance=%.1f",
                row_idx,
                col_indices[row_idx] if row_idx < len(col_indices) else -1,
                cost_matrix[row_idx, col_idx],
                max_distance,
            )
