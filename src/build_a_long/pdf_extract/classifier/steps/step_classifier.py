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
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.classifier.text import (
    extract_step_number_value,
)
from build_a_long.pdf_extract.extractor.bbox import BBox
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

        # Get the diagram from the diagram candidate
        # If the original diagram candidate is failed, try to find a replacement
        diagram = None
        diagram_candidate = score.diagram_candidate
        if diagram_candidate:
            if diagram_candidate.failure_reason:
                # Original diagram candidate was replaced (e.g., due to conflict)
                # Try to find a suitable replacement
                replacement = self._find_replacement_diagram(diagram_candidate, result)
                if replacement:
                    log.debug(
                        "[step] Using replacement diagram for step %d: %s -> %s",
                        step_num.value,
                        diagram_candidate.bbox,
                        replacement.bbox,
                    )
                    diagram_candidate = replacement
                else:
                    log.debug(
                        "[step] No replacement diagram found for step %d",
                        step_num.value,
                    )
                    diagram_candidate = None

            if diagram_candidate:
                diagram_elem = result.build(diagram_candidate)
                assert isinstance(diagram_elem, Diagram)
                diagram = diagram_elem

        # Get rotation symbols near this step (if any)
        rotation_symbol = self._get_rotation_symbol_for_step(step_num, diagram, result)

        # Get arrows for this step (from subassemblies and other sources)
        arrows = self._get_arrows_for_step(step_num, diagram, result)

        # Get subassemblies for this step
        subassemblies = self._get_subassemblies_for_step(step_num, diagram, result)

        # Build Step - clip bbox to page bounds
        page_bbox = result.page_data.bbox
        return Step(
            bbox=self._compute_step_bbox(step_num, parts_list, diagram, page_bbox),
            step_number=step_num,
            parts_list=parts_list,
            diagram=diagram,
            rotation_symbol=rotation_symbol,
            arrows=arrows,
            subassemblies=subassemblies,
        )

    def _find_replacement_diagram(
        self,
        original: Candidate,
        result: ClassificationResult,
    ) -> Candidate | None:
        """Find a replacement for a failed diagram candidate.

        When a diagram candidate fails due to conflict (e.g., blocks consumed
        by arrows), a reduced replacement candidate may have been created.
        This method finds the best replacement that overlaps with the original.

        Args:
            original: The original (failed) diagram candidate
            result: Classification result with all candidates

        Returns:
            A suitable replacement candidate, or None if none found
        """
        # Get all non-failed diagram candidates
        diagram_candidates = result.get_scored_candidates(
            "diagram", valid_only=False, exclude_failed=True
        )

        # Find candidates that significantly overlap with the original
        # (reduced candidates should have similar bbox)
        OVERLAP_THRESHOLD = 0.8  # Require 80% overlap

        best_candidate = None
        best_overlap = 0.0

        for candidate in diagram_candidates:
            # Check overlap ratio using intersection_area
            intersection_area = original.bbox.intersection_area(candidate.bbox)
            if intersection_area > 0:
                # Calculate overlap as percentage of the smaller bbox
                original_area = original.bbox.area
                candidate_area = candidate.bbox.area
                min_area = min(original_area, candidate_area)

                if min_area > 0:
                    overlap_ratio = intersection_area / min_area
                    if (
                        overlap_ratio >= OVERLAP_THRESHOLD
                        and overlap_ratio > best_overlap
                    ):
                        best_overlap = overlap_ratio
                        best_candidate = candidate

        return best_candidate

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

    def _assign_diagrams_hungarian(
        self,
        step_candidates: list[Candidate],
        diagram_candidates: list[Candidate],
    ) -> dict[int, tuple[Candidate, float]]:
        """Optimally assign diagrams to steps using the Hungarian algorithm.

        This finds the assignment that maximizes the total score across all
        step-diagram pairs, ensuring each step gets at most one diagram and
        each diagram is assigned to at most one step.

        Args:
            step_candidates: List of step candidates (with step_number info)
            diagram_candidates: List of diagram candidates

        Returns:
            Dict mapping step candidate index to (diagram_candidate, score)
        """
        if not step_candidates or not diagram_candidates:
            return {}

        n_diagrams = len(diagram_candidates)

        # Build cost matrix (we'll use negative scores since Hungarian minimizes)
        # Rows = steps, Columns = diagrams
        cost_matrix: list[list[float]] = []

        for step_cand in step_candidates:
            assert isinstance(step_cand.score_details, _StepScore)
            step_score = step_cand.score_details
            step_bbox = step_score.step_number_candidate.bbox

            row: list[float] = []
            for diag_cand in diagram_candidates:
                score = self._score_step_diagram_pair(step_bbox, diag_cand.bbox)
                # Convert to cost (negative score) for minimization
                row.append(-score)
            cost_matrix.append(row)

        # Run Hungarian algorithm
        row_assignments, col_assignments = self._hungarian_algorithm(cost_matrix)

        # Build result mapping
        result: dict[int, tuple[Candidate, float]] = {}
        for step_idx, diag_idx in zip(row_assignments, col_assignments, strict=True):
            if diag_idx < n_diagrams:  # Valid assignment (not a dummy)
                score = -cost_matrix[step_idx][diag_idx]
                result[step_idx] = (diagram_candidates[diag_idx], score)

        return result

    def _hungarian_algorithm(
        self, cost_matrix: list[list[float]]
    ) -> tuple[list[int], list[int]]:
        """Use scipy's Hungarian algorithm implementation.

        Args:
            cost_matrix: 2D cost matrix (rows=steps, cols=diagrams)

        Returns:
            Tuple of (row_indices, col_indices) for optimal assignment
        """
        if not cost_matrix or not cost_matrix[0]:
            return [], []

        # Convert to numpy array for scipy
        cost_array = np.array(cost_matrix)

        # scipy's linear_sum_assignment finds the optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_array)

        return list(row_indices), list(col_indices)

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
        search_region = BBox(
            x0=search_bbox.x0 - 100,
            y0=search_bbox.y0 - 100,
            x1=search_bbox.x1 + 100,
            y1=search_bbox.y1 + 100,
        )

        log.debug(
            "[step] Arrow search region for step %d: %s",
            step_num.value,
            search_region,
        )

        # Find arrows within or overlapping the search region
        arrows: list[Arrow] = []
        for candidate in arrow_candidates:
            overlaps = candidate.bbox.overlaps(search_region)
            log.debug(
                "[step]   Arrow candidate at %s, overlaps=%s, score=%.2f",
                candidate.bbox,
                overlaps,
                candidate.score,
            )
            if overlaps:
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
        search_region = BBox(
            x0=search_bbox.x0 - 150,
            y0=search_bbox.y0 - 150,
            x1=search_bbox.x1 + 150,
            y1=search_bbox.y1 + 150,
        )

        log.debug(
            "[step] SubAssembly search region for step %d: %s",
            step_num.value,
            search_region,
        )

        # Find subassemblies within or overlapping the search region
        subassemblies: list[SubAssembly] = []

        for candidate in subassembly_candidates:
            overlaps = candidate.bbox.overlaps(search_region)
            log.debug(
                "[step]   SubAssembly candidate at %s, overlaps=%s, score=%.2f",
                candidate.bbox,
                overlaps,
                candidate.score,
            )
            if overlaps:
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
        """Select the best Step candidates and optimally assign diagrams.

        Uses the Hungarian algorithm to find the optimal assignment between
        steps and diagrams that maximizes total matching score. Ensures each
        StepNumber value, PartsList, and Diagram is used at most once.

        Args:
            candidates: All possible Step candidates (without diagrams)
            result: Classification result containing diagram candidates

        Returns:
            Deduplicated list of Step candidates with diagrams assigned
        """
        # Get subassembly bboxes to exclude step numbers inside subassemblies
        # from diagram assignment (they have their own embedded diagrams)
        subassembly_candidates = result.get_scored_candidates(
            "subassembly", valid_only=False, exclude_failed=True
        )
        subassembly_bboxes = [c.bbox for c in subassembly_candidates]

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

        # Separate steps into main steps and subassembly steps
        # Steps inside subassemblies shouldn't compete for page-level diagrams
        main_step_candidates: list[Candidate] = []
        subassembly_step_indices: set[int] = set()

        for i, candidate in enumerate(unique_step_candidates):
            assert isinstance(candidate.score_details, _StepScore)
            score = candidate.score_details
            step_bbox = score.step_number_candidate.bbox

            # Check if this step is inside any subassembly
            # Use the step number's center point for containment check
            step_center_x, step_center_y = step_bbox.center
            is_inside_subassembly = any(
                sub_bbox.x0 <= step_center_x <= sub_bbox.x1
                and sub_bbox.y0 <= step_center_y <= sub_bbox.y1
                for sub_bbox in subassembly_bboxes
            )

            if is_inside_subassembly:
                subassembly_step_indices.add(i)
                log.debug(
                    "[step] Step at (%s,%s) is inside a subassembly, "
                    "excluding from diagram assignment",
                    step_bbox.x0,
                    step_bbox.y0,
                )
            else:
                main_step_candidates.append(candidate)

        # Get all diagram candidates
        diagram_candidates = result.get_scored_candidates(
            "diagram", valid_only=False, exclude_failed=True
        )

        log.debug(
            "[step] Assigning %d diagrams to %d main steps "
            "(excluding %d subassembly steps) using Hungarian algorithm",
            len(diagram_candidates),
            len(main_step_candidates),
            len(subassembly_step_indices),
        )

        # Use Hungarian algorithm for optimal assignment (main steps only)
        diagram_assignments = self._assign_diagrams_hungarian(
            main_step_candidates, diagram_candidates
        )

        # Create a mapping from main_step_candidates index to diagram assignment
        # We need to map this back to unique_step_candidates indices
        main_to_unique_idx: dict[int, int] = {}
        main_idx = 0
        for i, candidate in enumerate(unique_step_candidates):
            if i not in subassembly_step_indices:
                main_to_unique_idx[main_idx] = i
                main_idx += 1

        # Remap diagram assignments to unique_step_candidates indices
        unique_diagram_assignments: dict[int, tuple[Candidate, float]] = {}
        for main_idx, (diag_cand, diag_score) in diagram_assignments.items():
            unique_idx = main_to_unique_idx[main_idx]
            unique_diagram_assignments[unique_idx] = (diag_cand, diag_score)

        # Log the assignment matrix for debugging
        for i, step_cand in enumerate(unique_step_candidates):
            assert isinstance(step_cand.score_details, _StepScore)
            step_score = step_cand.score_details
            step_bbox = step_score.step_number_candidate.bbox

            # Get step value for logging
            text_block = step_score.step_number_candidate.source_blocks[0]
            assert isinstance(text_block, Text)
            step_value = extract_step_number_value(text_block.text)

            if i in subassembly_step_indices:
                log.debug(
                    "[step] Step %s at (%s,%s) -> (inside subassembly, no diagram)",
                    step_value,
                    step_bbox.x0,
                    step_bbox.y0,
                )
            elif i in unique_diagram_assignments:
                diag_cand, diag_score = unique_diagram_assignments[i]
                log.debug(
                    "[step] Step %s at (%s,%s) -> Diagram at (%s,%s) score=%.2f",
                    step_value,
                    step_bbox.x0,
                    step_bbox.y0,
                    diag_cand.bbox.x0,
                    diag_cand.bbox.y0,
                    diag_score,
                )
            else:
                log.debug(
                    "[step] Step %s at (%s,%s) -> No diagram assigned",
                    step_value,
                    step_bbox.x0,
                    step_bbox.y0,
                )

        # Build final candidates with diagram assignments
        selected: list[Candidate] = []
        used_parts_list_ids: set[int] = set()

        for i, candidate in enumerate(unique_step_candidates):
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

            # Get diagram assignment (only for main steps, not subassembly steps)
            diagram_candidate = None
            diagram_score = 0.0
            if i in unique_diagram_assignments:
                diagram_candidate, diagram_score = unique_diagram_assignments[i]

            # Update the score with the diagram assignment
            updated_score = _StepScore(
                step_number_candidate=score.step_number_candidate,
                parts_list_candidate=parts_list_candidate,
                has_parts_list=parts_list_candidate is not None,
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

            selected.append(updated_candidate)

            # Log selection
            text_block = score.step_number_candidate.source_blocks[0]
            assert isinstance(text_block, Text)
            step_value = extract_step_number_value(text_block.text)
            log.debug(
                "[step] Selected step %d (parts_list=%s, diagram=%s, score=%.2f)",
                step_value or 0,
                "yes" if parts_list_candidate is not None else "no",
                "yes" if diagram_candidate is not None else "no",
                updated_score.overall_score(),
            )

        return selected
