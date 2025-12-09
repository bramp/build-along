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
    Divider,
    LegoPageElements,
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

    step_value: int
    """The parsed step number value (e.g., 1, 2, 3)."""

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
        return (-self.overall_score(), self.step_value)


class StepClassifier(LabelClassifier):
    """Classifier for complete Step structures."""

    output = "step"
    requires = frozenset(
        {
            "step_number",
            "parts_list",
            "diagram",
            "rotation_symbol",
            "subassembly",
            "preview",
        }
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
        # Deduplication happens at build time, not scoring time, so that
        # diagram assignment can be re-evaluated as blocks get claimed.
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

        # Add all candidates to result (deduplication happens at build time)
        for candidate in all_candidates:
            result.add_candidate(candidate)

        log.debug(
            "[step] Created %d step candidates",
            len(all_candidates),
        )

    def build_all(self, result: ClassificationResult) -> list[LegoPageElements]:
        """Build all Step elements with coordinated rotation symbol assignment.

        This method:
        1. Builds all rotation symbols first (so they claim their Drawing blocks)
        2. Builds all parts lists (so they claim their blocks)
        3. Builds Step candidates, deduplicating by step value at build time
        4. Uses Hungarian matching to optimally assign rotation symbols to steps

        Deduplication happens at build time (not scoring time) so that diagram
        assignment can be re-evaluated as blocks get claimed by other elements.

        This coordination ensures:
        - Rotation symbols are built before diagrams, preventing incorrect clustering
        - Each rotation symbol is assigned to at most one step
        - Assignment is globally optimal based on distance to step diagrams

        Returns:
            List of successfully constructed Step elements, sorted by step number
        """
        # Phase 1: Build all rotation symbols BEFORE steps.
        # This allows rotation symbols to claim their Drawing blocks first,
        # preventing them from being incorrectly clustered into diagrams.
        for rs_candidate in result.get_scored_candidates(
            "rotation_symbol", valid_only=False, exclude_failed=True
        ):
            try:
                result.build(rs_candidate)
                log.debug(
                    "[step] Built rotation symbol at %s (score=%.2f)",
                    rs_candidate.bbox,
                    rs_candidate.score,
                )
            except Exception as e:
                log.debug(
                    "[step] Failed to construct rotation_symbol candidate at %s: %s",
                    rs_candidate.bbox,
                    e,
                )

        # Phase 2: Build all parts lists BEFORE steps.
        # This allows parts lists to claim their Drawing blocks first,
        # preventing them from being claimed by subassemblies.
        for pl_candidate in result.get_scored_candidates(
            "parts_list", valid_only=False, exclude_failed=True
        ):
            try:
                result.build(pl_candidate)
                log.debug(
                    "[step] Built parts_list at %s (score=%.2f)",
                    pl_candidate.bbox,
                    pl_candidate.score,
                )
            except Exception as e:
                log.debug(
                    "[step] Failed to construct parts_list candidate at %s: %s",
                    pl_candidate.bbox,
                    e,
                )

        # Phase 3: Build subassemblies and previews BEFORE steps.
        # Both subassemblies and previews are white boxes with diagrams inside.
        # We combine them and build in score order so the higher-scoring
        # candidate claims the white box first. When a candidate is built,
        # its source_blocks are marked as consumed, causing any competing
        # candidate using the same blocks to fail.
        #
        # This allows subassemblies (which have step_count labels like "2x")
        # to be distinguished from previews (which appear before steps).
        subassembly_candidates = result.get_scored_candidates(
            "subassembly", valid_only=False, exclude_failed=True
        )
        preview_candidates = result.get_scored_candidates(
            "preview", valid_only=False, exclude_failed=True
        )

        # Combine and sort by score (highest first)
        combined_candidates = list(subassembly_candidates) + list(preview_candidates)
        combined_candidates.sort(key=lambda c: c.score, reverse=True)

        for candidate in combined_candidates:
            try:
                result.build(candidate)
                log.debug(
                    "[step] Built %s at %s (score=%.2f)",
                    candidate.label,
                    candidate.bbox,
                    candidate.score,
                )
            except Exception as e:
                log.debug(
                    "[step] Failed to construct %s candidate at %s: %s",
                    candidate.label,
                    candidate.bbox,
                    e,
                )

        # Phase 4: Build Step candidates with deduplication by step value
        # Filter out subassembly steps, then build in score order, skipping
        # step values that have already been built.
        # Steps are built as "partial" - just step_number + parts_list.
        # Diagrams and subassemblies are assigned later via Hungarian matching.
        all_step_candidates = result.get_scored_candidates(
            "step", valid_only=False, exclude_failed=True
        )
        page_level_step_candidates = self._filter_page_level_step_candidates(
            all_step_candidates
        )

        # Sort by score (highest first) so best candidates get built first
        sorted_candidates = sorted(
            page_level_step_candidates,
            key=lambda c: c.score,
            reverse=True,
        )

        steps: list[Step] = []
        built_step_values: set[int] = set()

        for step_candidate in sorted_candidates:
            # Get step value from score
            score = step_candidate.score_details
            if not isinstance(score, _StepScore):
                continue

            # Skip if we've already built a step with this value
            if score.step_value in built_step_values:
                log.debug(
                    "[step] Skipping duplicate step %d candidate (score=%.2f)",
                    score.step_value,
                    step_candidate.score,
                )
                continue

            try:
                elem = result.build(step_candidate)
                assert isinstance(elem, Step)
                steps.append(elem)
                built_step_values.add(score.step_value)
                log.debug(
                    "[step] Built partial step %d (parts_list=%s, score=%.2f)",
                    score.step_value,
                    "yes" if score.parts_list_candidate is not None else "no",
                    step_candidate.score,
                )
            except Exception as e:
                log.debug(
                    "[step] Failed to construct step %d candidate: %s",
                    score.step_value,
                    e,
                )

        # Sort steps by their step_number value
        steps.sort(key=lambda step: step.step_number.value)

        # Phase 5: Assign diagrams to steps using Hungarian matching
        # Collect available diagram candidates (not yet claimed by subassemblies)
        available_diagrams: list[Diagram] = []
        for diag_candidate in result.get_scored_candidates(
            "diagram", valid_only=False, exclude_failed=True
        ):
            if diag_candidate.constructed is None:
                # Build the diagram
                try:
                    diag_elem = result.build(diag_candidate)
                    assert isinstance(diag_elem, Diagram)
                    available_diagrams.append(diag_elem)
                except Exception as e:
                    log.debug(
                        "[step] Failed to build diagram at %s: %s",
                        diag_candidate.bbox,
                        e,
                    )
            elif isinstance(diag_candidate.constructed, Diagram):
                # Already built (claimed by subassembly) - skip
                pass

        log.debug(
            "[step] Available diagrams for step assignment: %d",
            len(available_diagrams),
        )

        # Assign diagrams to steps using Hungarian matching
        assign_diagrams_to_steps(steps, available_diagrams)

        # Phase 6: Assign subassemblies to steps using Hungarian matching
        # Collect built subassemblies
        subassemblies: list[SubAssembly] = []
        for sa_candidate in result.get_scored_candidates(
            "subassembly", valid_only=True
        ):
            assert sa_candidate.constructed is not None
            assert isinstance(sa_candidate.constructed, SubAssembly)
            subassemblies.append(sa_candidate.constructed)

        # Collect built dividers for obstruction checking
        dividers: list[Divider] = []
        for div_candidate in result.get_scored_candidates("divider", valid_only=True):
            assert div_candidate.constructed is not None
            assert isinstance(div_candidate.constructed, Divider)
            dividers.append(div_candidate.constructed)

        log.debug(
            "[step] Assigning %d subassemblies to %d steps (%d dividers for "
            "obstruction checking)",
            len(subassemblies),
            len(steps),
            len(dividers),
        )

        # Assign subassemblies to steps using Hungarian matching
        assign_subassemblies_to_steps(steps, subassemblies, dividers)

        # Phase 7: Finalize steps - compute arrows and final bboxes
        page_bbox = result.page_data.bbox
        for step in steps:
            # Get arrows for this step
            arrows = self._get_arrows_for_step(step.step_number, step.diagram, result)

            # Compute final bbox including all components
            final_bbox = self._compute_step_bbox(
                step.step_number,
                step.parts_list,
                step.diagram,
                step.subassemblies,
                page_bbox,
            )

            # Update the step (Step is a Pydantic model, so we need to reassign)
            # We mutate in place by directly setting attributes
            object.__setattr__(step, "arrows", arrows)
            object.__setattr__(step, "bbox", final_bbox)

        # Phase 8: Assign rotation symbols to steps using Hungarian matching
        # Collect built rotation symbols
        rotation_symbols: list[RotationSymbol] = []
        for rs_candidate in result.get_scored_candidates(
            "rotation_symbol", valid_only=True
        ):
            assert rs_candidate.constructed is not None
            assert isinstance(rs_candidate.constructed, RotationSymbol)
            rotation_symbols.append(rs_candidate.constructed)

        assign_rotation_symbols_to_steps(steps, rotation_symbols)

        log.debug(
            "[step] build_all complete: %d steps, %d rotation symbols assigned",
            len(steps),
            sum(1 for s in steps if s.rotation_symbol is not None),
        )

        # Cast for type compatibility with base class return type
        return list[LegoPageElements](steps)

    def _filter_page_level_step_candidates(
        self, candidates: list[Candidate]
    ) -> list[Candidate]:
        """Filter step candidates to exclude likely subassembly steps.

        Extracts step number values from candidates and uses the generic
        filter_subassembly_values function to filter out subassembly steps.

        Args:
            candidates: All step candidates to filter

        Returns:
            Filtered list of candidates likely to be page-level steps
        """
        # Extract step number values from candidates
        items_with_values: list[tuple[int, Candidate]] = []
        for candidate in candidates:
            score = candidate.score_details
            if not isinstance(score, _StepScore):
                continue
            items_with_values.append((score.step_value, candidate))

        # Use generic filtering function
        filtered_items = filter_subassembly_values(items_with_values)

        # If filtering occurred, return only the filtered candidates
        if len(filtered_items) != len(items_with_values):
            filtered_values = [v for v, _ in filtered_items]
            log.debug(
                "[step] Filtered out likely subassembly steps, keeping values: %s",
                sorted(filtered_values),
            )
            return [item for _, item in filtered_items]

        # No filtering occurred, return original candidates
        # (includes any candidates that couldn't have their value extracted)
        return candidates

    def build(self, candidate: Candidate, result: ClassificationResult) -> Step:
        """Construct a partial Step element from a single candidate.

        This creates a Step with just step_number and parts_list. The diagram,
        subassemblies, and arrows are assigned later in build_all() using
        Hungarian matching for optimal global assignment.
        """
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

        # Create partial step - diagram, subassemblies, arrows assigned later
        initial_bbox = step_num.bbox
        if parts_list:
            initial_bbox = initial_bbox.union(parts_list.bbox)

        return Step(
            bbox=initial_bbox,
            step_number=step_num,
            parts_list=parts_list,
            diagram=None,
            rotation_symbol=None,
            arrows=[],
            subassemblies=[],
        )

    def _find_and_build_diagram_for_step(
        self,
        step_num: StepNumber,
        parts_list: PartsList | None,
        result: ClassificationResult,
    ) -> Diagram | None:
        """Find and build the best diagram for this step.

        This is called at build time, after rotation symbols and subassemblies
        have been built (in build_all Phases 1 and 3), so they've already
        claimed any images/diagrams they need. We filter to only consider
        diagram candidates that haven't been constructed yet.

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
        # (subassemblies and rotation symbols built earlier may have claimed some)
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
            The created Candidate with score but no construction, or None if
            the step number value cannot be extracted.
        """
        ABOVE_EPS = 2.0  # Small epsilon for "above" check
        ALIGNMENT_THRESHOLD_MULTIPLIER = 1.0  # Max horizontal offset
        DISTANCE_THRESHOLD_MULTIPLIER = 1.0  # Max vertical distance

        # Extract step number value from the candidate
        if not step_candidate.source_blocks:
            return None
        source_block = step_candidate.source_blocks[0]
        if not isinstance(source_block, Text):
            return None
        step_value = extract_step_number_value(source_block.text)
        if step_value is None:
            return None

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
            step_value=step_value,
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
        """Get already-built subassemblies that belong to this step.

        Subassemblies are built in build_all() before steps. This method finds
        subassemblies that are positioned near this step's diagram and haven't
        been claimed by another step yet.

        Args:
            step_num: The step number element
            diagram: The diagram element (if any)
            result: Classification result containing subassembly candidates

        Returns:
            List of SubAssembly elements for this step
        """
        # Get all built subassemblies
        all_subassemblies: list[SubAssembly] = []
        for sa_candidate in result.get_scored_candidates(
            "subassembly", valid_only=True
        ):
            assert sa_candidate.constructed is not None
            assert isinstance(sa_candidate.constructed, SubAssembly)
            all_subassemblies.append(sa_candidate.constructed)

        log.debug(
            "[step] Looking for subassemblies for step %d, found %d built",
            step_num.value,
            len(all_subassemblies),
        )

        if not all_subassemblies:
            return []

        # Determine search region based on step_number and diagram
        reference_bbox = diagram.bbox.union(step_num.bbox) if diagram else step_num.bbox

        page_bbox = result.page_data.bbox

        # Expand search region around the reference area
        # Horizontally: search the full page width
        # Vertically: search a region around the reference bbox
        vertical_margin = reference_bbox.height
        search_region = BBox(
            x0=page_bbox.x0,
            y0=max(page_bbox.y0, reference_bbox.y0 - vertical_margin),
            x1=page_bbox.x1,
            y1=min(page_bbox.y1, reference_bbox.y1 + vertical_margin),
        )

        log.debug(
            "[step] SubAssembly search region for step %d: %s",
            step_num.value,
            search_region,
        )

        # Find subassemblies that overlap the search region
        subassemblies: list[SubAssembly] = []
        for subassembly in all_subassemblies:
            if subassembly.bbox.overlaps(search_region):
                log.debug(
                    "[step]   SubAssembly at %s overlaps search region",
                    subassembly.bbox,
                )
                subassemblies.append(subassembly)

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
        subassemblies: list[SubAssembly],
        page_bbox: BBox,
    ) -> BBox:
        """Compute the overall bounding box for the Step.

        This encompasses the step number, parts list (if any), diagram (if any),
        and all subassemblies.
        The result is clipped to the page bounds to handle elements that extend
        slightly off-page (e.g., arrows in diagrams).

        Args:
            step_num: The step number element
            parts_list: The parts list (if any)
            diagram: The diagram element (if any)
            subassemblies: List of subassemblies for this step
            page_bbox: The page bounding box to clip to

        Returns:
            Combined bounding box, clipped to page bounds
        """
        bboxes = [step_num.bbox]
        if parts_list:
            bboxes.append(parts_list.bbox)
        if diagram:
            bboxes.append(diagram.bbox)
        for sa in subassemblies:
            bboxes.append(sa.bbox)

        return BBox.union_all(bboxes).clip_to(page_bbox)


def assign_diagrams_to_steps(
    steps: list[Step],
    diagrams: list[Diagram],
    max_distance: float = 500.0,
) -> None:
    """Assign diagrams to steps using Hungarian algorithm.

    Uses optimal bipartite matching to pair diagrams with steps based on
    a scoring function that considers:
    - Distance from step_number to diagram (closer is better)
    - Relative position (diagram typically below/right of step_number)

    This function mutates the Step objects in place, setting their diagram
    attribute.

    Args:
        steps: List of Step objects to assign diagrams to
        diagrams: List of Diagram objects to assign
        max_distance: Maximum distance for a valid assignment. Pairs with distance
            greater than this will not be matched.
    """
    if not steps or not diagrams:
        log.debug(
            "[step] No diagram assignment needed: steps=%d, diagrams=%d",
            len(steps),
            len(diagrams),
        )
        return

    n_steps = len(steps)
    n_diagrams = len(diagrams)

    log.debug(
        "[step] Running Hungarian matching for diagrams: %d steps, %d diagrams",
        n_steps,
        n_diagrams,
    )

    # Build cost matrix: rows = diagrams, cols = steps
    # Lower cost = better match
    cost_matrix = np.zeros((n_diagrams, n_steps))

    for i, diag in enumerate(diagrams):
        diag_center = diag.bbox.center
        for j, step in enumerate(steps):
            step_num_center = step.step_number.bbox.center

            # Base cost is distance from step_number to diagram center
            distance = (
                (step_num_center[0] - diag_center[0]) ** 2
                + (step_num_center[1] - diag_center[1]) ** 2
            ) ** 0.5

            # Apply position penalty: prefer diagrams that are below or to the
            # right of the step_number (typical LEGO instruction layout)
            # If diagram is above the step_number, add penalty
            if diag_center[1] < step_num_center[1] - 50:  # Diagram above step
                distance *= 1.5  # Penalty for being above

            cost_matrix[i, j] = distance
            log.debug(
                "[step]   Cost diagram[%d] at %s to step[%d]: %.1f",
                i,
                diag.bbox,
                step.step_number.value,
                distance,
            )

    # Apply max_distance threshold
    high_cost = max_distance * 10
    cost_matrix_thresholded = np.where(
        cost_matrix > max_distance, high_cost, cost_matrix
    )

    # Run Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix_thresholded)

    # Assign diagrams to steps based on the matching
    for row_idx, col_idx in zip(row_indices, col_indices, strict=True):
        if cost_matrix[row_idx, col_idx] <= max_distance:
            diag = diagrams[row_idx]
            step = steps[col_idx]
            # Use object.__setattr__ because Step is a frozen Pydantic model
            object.__setattr__(step, "diagram", diag)
            log.debug(
                "[step] Assigned diagram at %s to step %d (cost=%.1f)",
                diag.bbox,
                step.step_number.value,
                cost_matrix[row_idx, col_idx],
            )
        else:
            log.debug(
                "[step] Skipped diagram assignment: diagram[%d] to step[%d] "
                "cost=%.1f > max_distance=%.1f",
                row_idx,
                col_idx,
                cost_matrix[row_idx, col_idx],
                max_distance,
            )


def _has_divider_between(
    bbox1: BBox,
    bbox2: BBox,
    dividers: list[Divider],
) -> bool:
    """Check if there is a divider line between two bboxes.

    A divider is considered "between" if it crosses the line connecting
    the centers of the two bboxes.

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
        dividers: List of dividers to check

    Returns:
        True if a divider is between the two bboxes
    """
    center1 = bbox1.center
    center2 = bbox2.center

    for divider in dividers:
        div_bbox = divider.bbox

        # Check if divider is vertical (separates left/right)
        if div_bbox.width < div_bbox.height * 0.2:  # Vertical line
            # Check if divider x is between the two centers
            min_x = min(center1[0], center2[0])
            max_x = max(center1[0], center2[0])
            div_x = div_bbox.center[0]
            if min_x < div_x < max_x:
                # Check if divider spans the y range
                min_y = min(center1[1], center2[1])
                max_y = max(center1[1], center2[1])
                if div_bbox.y0 <= max_y and div_bbox.y1 >= min_y:
                    return True

        # Check if divider is horizontal (separates top/bottom)
        elif div_bbox.height < div_bbox.width * 0.2:  # Horizontal line
            # Check if divider y is between the two centers
            min_y = min(center1[1], center2[1])
            max_y = max(center1[1], center2[1])
            div_y = div_bbox.center[1]
            if min_y < div_y < max_y:
                # Check if divider spans the x range
                min_x = min(center1[0], center2[0])
                max_x = max(center1[0], center2[0])
                if div_bbox.x0 <= max_x and div_bbox.x1 >= min_x:
                    return True

    return False


def assign_subassemblies_to_steps(
    steps: list[Step],
    subassemblies: list[SubAssembly],
    dividers: list[Divider],
    max_distance: float = 400.0,
) -> None:
    """Assign subassemblies to steps using Hungarian algorithm.

    Uses optimal bipartite matching to pair subassemblies with steps based on:
    - Distance from step's diagram to subassembly (closer is better)
    - No divider between the subassembly and the step's diagram

    This function mutates the Step objects in place, adding to their
    subassemblies list.

    Args:
        steps: List of Step objects to assign subassemblies to
        subassemblies: List of SubAssembly objects to assign
        dividers: List of Divider objects for obstruction checking
        max_distance: Maximum distance for a valid assignment
    """
    if not steps or not subassemblies:
        log.debug(
            "[step] No subassembly assignment needed: steps=%d, subassemblies=%d",
            len(steps),
            len(subassemblies),
        )
        return

    n_steps = len(steps)
    n_subassemblies = len(subassemblies)

    log.debug(
        "[step] Running Hungarian matching for subassemblies: "
        "%d steps, %d subassemblies",
        n_steps,
        n_subassemblies,
    )

    # Build cost matrix: rows = subassemblies, cols = steps
    cost_matrix = np.zeros((n_subassemblies, n_steps))
    high_cost = max_distance * 10

    for i, sa in enumerate(subassemblies):
        sa_center = sa.bbox.center
        for j, step in enumerate(steps):
            # Use diagram center if available, otherwise step bbox center
            if step.diagram:
                target_bbox = step.diagram.bbox
                target_center = target_bbox.center
            else:
                target_bbox = step.bbox
                target_center = step.step_number.bbox.center

            # Base cost is distance from diagram/step to subassembly center
            distance = (
                (target_center[0] - sa_center[0]) ** 2
                + (target_center[1] - sa_center[1]) ** 2
            ) ** 0.5

            # Check for divider between subassembly and step's diagram
            if _has_divider_between(sa.bbox, target_bbox, dividers):
                # High cost if there's a divider between
                distance = high_cost
                log.debug(
                    "[step]   Divider between subassembly[%d] at %s and step[%d]",
                    i,
                    sa.bbox,
                    step.step_number.value,
                )

            cost_matrix[i, j] = distance
            log.debug(
                "[step]   Cost subassembly[%d] at %s to step[%d]: %.1f",
                i,
                sa.bbox,
                step.step_number.value,
                distance,
            )

    # Apply max_distance threshold
    cost_matrix_thresholded = np.where(
        cost_matrix > max_distance, high_cost, cost_matrix
    )

    # Run Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix_thresholded)

    # Assign subassemblies to steps based on the matching
    # Note: Unlike diagrams, multiple subassemblies can be assigned to one step
    # The Hungarian algorithm gives us the best 1:1 matching, but we may want
    # to assign additional nearby subassemblies too

    # First, do the 1:1 optimal assignment
    assigned_subassemblies: set[int] = set()
    for row_idx, col_idx in zip(row_indices, col_indices, strict=True):
        if cost_matrix[row_idx, col_idx] <= max_distance:
            sa = subassemblies[row_idx]
            step = steps[col_idx]
            # Add to step's subassemblies list
            new_subassemblies = list(step.subassemblies) + [sa]
            object.__setattr__(step, "subassemblies", new_subassemblies)
            assigned_subassemblies.add(row_idx)
            log.debug(
                "[step] Assigned subassembly at %s to step %d (cost=%.1f)",
                sa.bbox,
                step.step_number.value,
                cost_matrix[row_idx, col_idx],
            )

    # Then, try to assign remaining unassigned subassemblies to their nearest step
    # (if within max_distance and no divider between)
    for i, sa in enumerate(subassemblies):
        if i in assigned_subassemblies:
            continue

        # Find the step with lowest cost for this subassembly
        best_step_idx = None
        best_cost = high_cost
        for j in range(len(steps)):
            if cost_matrix[i, j] < best_cost:
                best_cost = cost_matrix[i, j]
                best_step_idx = j

        if best_step_idx is not None and best_cost <= max_distance:
            step = steps[best_step_idx]
            new_subassemblies = list(step.subassemblies) + [sa]
            object.__setattr__(step, "subassemblies", new_subassemblies)
            log.debug(
                "[step] Assigned extra subassembly at %s to step %d (cost=%.1f)",
                sa.bbox,
                step.step_number.value,
                best_cost,
            )


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


def filter_subassembly_values[T](
    items: list[tuple[int, T]],
    *,
    min_gap: int = 3,
    max_subassembly_start: int = 3,
) -> list[tuple[int, T]]:
    """Filter items to exclude likely subassembly values.

    Subassembly steps (e.g., step 1, 2 inside a subassembly box) often appear
    alongside higher-numbered page-level steps (e.g., 15, 16). This creates
    sequences like [1, 2, 15, 16] which cannot all be page-level steps.

    This function identifies such cases by detecting a significant gap in values
    and filtering out the lower-numbered items that are likely subassembly steps.

    Args:
        items: List of (value, item) tuples where value is the step number
            and item is any associated data (e.g., a Candidate).
        min_gap: Minimum gap size to trigger filtering (exclusive).
            Default is 3, so gaps > 3 trigger filtering.
        max_subassembly_start: Maximum starting value for subassembly detection.
            If min value is <= this, filtering is applied. Default is 3.

    Returns:
        Filtered list of (value, item) tuples, keeping only the higher values
        if a subassembly pattern is detected. Returns original list if no
        filtering is needed.

    Examples:
        >>> filter_subassembly_values([(1, "a"), (2, "b"), (15, "c"), (16, "d")])
        [(15, "c"), (16, "d")]

        >>> filter_subassembly_values([(5, "a"), (6, "b"), (15, "c")])
        [(5, "a"), (6, "b"), (15, "c")]  # min=5 > 3, no filtering
    """
    if len(items) <= 1:
        return items

    # Sort by value
    sorted_items = sorted(items, key=lambda x: x[0])
    values = [v for v, _ in sorted_items]

    # Find the largest gap between consecutive values
    max_gap_size = 0
    max_gap_index = -1
    for i in range(len(values) - 1):
        gap = values[i + 1] - values[i]
        if gap > max_gap_size:
            max_gap_size = gap
            max_gap_index = i

    # Check if filtering should occur:
    # 1. Gap must be larger than min_gap
    # 2. The minimum value must be <= max_subassembly_start
    if max_gap_size > min_gap and max_gap_index >= 0:
        min_value = values[0]
        if min_value <= max_subassembly_start:
            threshold = values[max_gap_index + 1]
            return [(v, item) for v, item in sorted_items if v >= threshold]

    return items
