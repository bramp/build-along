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

- CLASSIFIER_DEBUG=step (or "all")
    Enables more verbose, structured logs in this classifier, including
    candidate enumeration and rejection reasons.
"""

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

from build_a_long.pdf_extract.classifier.classification_result import (
    Candidate,
    ClassificationHints,
    ClassificationResult,
    ClassifierConfig,
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
from build_a_long.pdf_extract.extractor.page_elements import (
    Drawing,
)

log = logging.getLogger(__name__)


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

    def __init__(self, config: ClassifierConfig, classifier):
        super().__init__(config, classifier)
        self._debug_enabled = os.getenv("CLASSIFIER_DEBUG", "").lower() in (
            "step",
            "all",
        )

    def evaluate(
        self,
        page_data: PageData,
        result: ClassificationResult,
    ) -> None:
        """Evaluate elements and create candidates for complete Step structures.

        Combines step_number and parts_list elements, identifies diagram regions,
        and creates Step candidates.
        """

        # Get step_number candidates and their constructed StepNumber elements
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

        # Get parts_list candidates and their constructed PartsList elements
        parts_list_candidates = result.get_candidates("parts_list")
        parts_lists: list[PartsList] = []

        for candidate in parts_list_candidates:
            if (
                candidate.is_winner
                and candidate.constructed is not None
                and isinstance(candidate.constructed, PartsList)
            ):
                parts_lists.append(candidate.constructed)

        drawings: list[Drawing] = [
            e for e in page_data.elements if isinstance(e, Drawing)
        ]

        if self._debug_enabled:
            log.debug(
                "[step] page=%s elements=%d steps=%d parts_lists=%d drawings=%d",
                page_data.page_number,
                len(page_data.elements),
                len(steps),
                len(parts_lists),
                len(drawings),
            )

        # Build a Step for each step_number
        for step_num in steps:
            # Find the parts_list associated with this step (if any)
            # Parts lists are typically above the step number
            associated_parts_list = self._find_associated_parts_list(
                step_num, parts_lists
            )

            # Identify the diagram region for this step
            diagram_bbox = self._identify_diagram_region(
                step_num, associated_parts_list, page_data
            )

            # Create the Step score
            score = _StepScore(
                step_number=step_num,
                has_parts_list=associated_parts_list is not None,
                diagram_area=diagram_bbox.area,
            )

            # Build the Diagram element
            diagram = Diagram(
                bbox=diagram_bbox,
            )

            # Build the Step
            constructed = Step(
                bbox=self._compute_step_bbox(step_num, associated_parts_list, diagram),
                step_number=step_num,
                parts_list=associated_parts_list
                or PartsList(bbox=step_num.bbox, parts=[]),
                diagram=diagram,
            )
            # Add candidate - Note: Step is a synthetic element combining
            # step_number, parts_list, and diagram, so source_element is None
            result.add_candidate(
                "step",
                Candidate(
                    bbox=constructed.bbox,
                    label="step",
                    score=1.0,  # Step uses ranking rather than scores
                    score_details=score,
                    constructed=constructed,
                    source_element=None,  # Synthetic element has no single source
                    failure_reason=None,
                    is_winner=False,  # Will be set by classify()
                ),
            )

    def _find_associated_parts_list(
        self, step_num: StepNumber, parts_lists: Sequence[PartsList]
    ) -> PartsList | None:
        """Find the parts list associated with a step number.

        The parts list is typically above the step number. We look for parts lists
        that are above the step and choose the closest one.

        Args:
            step_num: The step number to find a parts list for
            parts_lists: List of all parts lists on the page

        Returns:
            The associated PartsList or None if no suitable parts list is found
        """
        candidates = []

        for parts_list in parts_lists:
            # Check if parts list is above the step number
            if self._is_parts_list_above_step(parts_list, step_num):
                # Calculate distance
                distance = step_num.bbox.y0 - parts_list.bbox.y1
                candidates.append((distance, parts_list))

        if not candidates:
            return None

        # Return the closest parts list
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _is_parts_list_above_step(
        self, parts_list: PartsList, step_num: StepNumber
    ) -> bool:
        """Check if a parts list is spatially above a step number.

        Args:
            parts_list: The parts list element
            step_num: The step number element

        Returns:
            True if the parts list is above the step number
        """
        ABOVE_EPS = 2.0
        return parts_list.bbox.y1 <= step_num.bbox.y0 + ABOVE_EPS

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

    def classify(
        self,
        page_data: PageData,
        result: ClassificationResult,
        hints: ClassificationHints | None,
    ) -> None:
        """Classify Step candidates and mark winners.

        Args:
            page_data: The page data
            result: Classification result to update
            hints: Optional hints (unused)
        """
        # Get pre-built candidates
        candidate_list = result.get_candidates("step")

        # Sort the candidates based on our scoring criteria
        sorted_candidates = sorted(
            candidate_list,
            key=lambda c: c.score_details.sort_key(),
        )

        # Mark winners (all successfully constructed candidates)
        for candidate in sorted_candidates:
            if candidate.constructed is None:
                # Already has failure_reason from evaluate
                continue

            assert isinstance(candidate.constructed, Step)

            # Step is synthetic and has no source_element, so no removal check needed
            # (there's no underlying element that could be removed by other classifiers)

            # This is a winner!
            result.mark_winner(
                candidate, candidate.source_element, candidate.constructed
            )

            # No need to remove overlapping elements since Step is synthetic
            # and doesn't consume any underlying elements
