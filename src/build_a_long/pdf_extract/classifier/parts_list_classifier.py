"""
Parts list classifier.

Purpose
-------
Identify the drawing region(s) that represent the page's parts list.
We look for drawings above a detected step number which contain one or
more part-count texts (e.g., "2x", "5×"). Among candidates, we prefer
closest-by vertical proximity to the step.

Debugging
---------
Set environment variables to aid investigation without code changes:

- LOG_LEVEL=DEBUG
    Enables DEBUG-level logging (if not already configured by caller).

- CLASSIFIER_DEBUG=parts_list (or "all")
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
from build_a_long.pdf_extract.classifier.text_extractors import (
    extract_part_count_value,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Part,
    PartCount,
    PartsList,
    StepNumber,
)
from build_a_long.pdf_extract.extractor.page_elements import (
    Drawing,
    Element,
    Text,
)

log = logging.getLogger(__name__)


@dataclass
class _PartsListScore:
    """Internal score representation for parts list classification."""

    step: StepNumber
    """The step number this parts list is associated with."""

    part_count_count: int
    """Number of part count texts contained in the bounding box."""

    step_proximity: float
    """Vertical proximity to the step number (lower is better)."""

    area: float
    """Area of the bounding box."""

    def sort_key(self) -> tuple[int, float, float]:
        """Return a tuple for sorting candidates.

        We prefer:
        1. More part counts (higher is better)
        2. Closer proximity to step (lower is better)
        3. Smaller area (lower is better)
        """
        return (self.part_count_count, self.step_proximity, self.area)


class PartsListClassifier(LabelClassifier):
    """Classifier for parts lists."""

    outputs = {"parts_list"}
    requires = {"step_number", "part_count"}

    def __init__(self, config: ClassifierConfig, classifier):
        super().__init__(config, classifier)
        self._debug_enabled = os.getenv("CLASSIFIER_DEBUG", "").lower() in (
            "parts_list",
            "all",
        )

    def evaluate(
        self,
        page_data: PageData,
        result: ClassificationResult,
    ) -> None:
        """Evaluate elements and create candidates for potential parts list drawings.

        Scores drawings based on their proximity to step numbers and the number
        of part count texts they contain. Creates candidates for all viable
        parts list drawings.
        """

        # Get step_number candidates and their constructed StepNumber elements
        step_candidates = result.get_candidates("step_number")
        # Use constructed StepNumber elements instead of source elements
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

        # Get part_count candidates and their constructed PartCount elements
        part_count_candidates = result.get_candidates("part_count")
        # Use constructed PartCount elements instead of source elements
        part_counts: list[PartCount] = []
        for candidate in part_count_candidates:
            if (
                candidate.is_winner
                and candidate.constructed is not None
                and isinstance(candidate.constructed, PartCount)
            ):
                part_counts.append(candidate.constructed)
        if not part_counts:
            return

        drawings: list[Drawing] = [
            e for e in page_data.elements if isinstance(e, Drawing)
        ]
        if not drawings:
            return

        if self._debug_enabled:
            log.debug(
                "[parts_list] page=%s elements=%d steps=%d drawings=%d",
                page_data.page_number,
                len(page_data.elements),
                len(steps),
                len(drawings),
            )

        # Score each drawing relative to all steps
        # For each drawing, find the closest step to determine its score
        for drawing in drawings:
            # Find the closest step and score this drawing
            step_score = self._score_distance_to_step_number(drawing, steps)
            if step_score is None:
                # Drawing is not above any step
                continue

            closest_step, step_score = step_score

            # Find all part counts contained in this drawing
            contained = self._score_containing_parts(drawing, part_counts)
            if not contained:
                # Drawing contains no part counts, skip it
                continue

            # Create score object with associated step
            score = _PartsListScore(
                step=closest_step,
                part_count_count=len(contained),
                step_proximity=step_score,
                area=drawing.bbox.area,
            )

            constructed = PartsList(
                bbox=drawing.bbox,
                # TODO Extract Part and do the matching
                parts=[
                    Part(
                        bbox=pc.bbox,  # TODO
                        count=pc,
                    )
                    for pc in contained
                ],
            )

            # Add candidate
            result.add_candidate(
                "parts_list",
                Candidate(
                    bbox=drawing.bbox,
                    label="parts_list",
                    score=1.0,  # Parts list uses ranking rather than scores
                    score_details=score,
                    constructed=constructed,
                    source_element=drawing,
                    failure_reason=None,
                    is_winner=False,  # Will be set by classify()
                ),
            )

    def _score_containing_parts(
        self, drawing: Drawing, part_counts: Sequence[PartCount]
    ) -> list[PartCount]:
        """Find all part counts that are contained within a drawing.

        Args:
            drawing: The drawing element to check
            part_counts: List of all part count elements on the page

        Returns:
            List of PartCount elements whose bboxes are fully inside the drawing
        """
        contained = [
            part_count
            for part_count in part_counts
            if part_count.bbox.fully_inside(drawing.bbox)
        ]

        return contained

    def _score_distance_to_step_number(
        self, drawing: Drawing, steps: Sequence[StepNumber]
    ) -> tuple[StepNumber, float] | None:
        """Find the closest step that a drawing is above.

        Args:
            drawing: The drawing element to score
            steps: List of step numbers on the page

        Returns:
            Tuple of (closest_step, distance_to_step) if drawing is above at least one step,
            None if drawing is not above any step.
        """
        # Find all steps that this drawing is above
        relevant_steps = []
        for step in steps:
            if self._is_drawing_above_step(drawing, step):
                relevant_steps.append(step)

        # If drawing is not above any step, return None
        if not relevant_steps:
            return None

        # Find the closest step (minimum distance)
        closest_step = min(relevant_steps, key=lambda s: s.bbox.y0 - drawing.bbox.y1)

        return (closest_step, closest_step.bbox.y0 - drawing.bbox.y1)

    def _is_drawing_above_step(self, drawing: Drawing, step: StepNumber) -> bool:
        """Check if a drawing is spatially above a step number.

        Args:
            drawing: The drawing element to check
            step: The step number element

        Returns:
            True if the drawing is above the step (within epsilon)
        """
        ABOVE_EPS = 2.0
        return drawing.bbox.y1 <= step.bbox.y0 + ABOVE_EPS

    def _build_parts_list_for_winner(
        self,
        parts_list_drawing: Drawing,
        page_data: PageData,
        result: ClassificationResult,
    ) -> PartsList | None:
        """Build a PartsList from a winning parts_list drawing.

        This method pairs part_count texts with part_image Images within the
        parts_list drawing using the part_image_pairs stored in the result.

        Args:
            parts_list_drawing: The Drawing element representing the parts list
            page_data: The page data containing all elements
            result: Classification result containing part_image_pairs

        Returns:
            A PartsList object containing all Part entries, or None if construction fails
        """
        parts: list[Part] = []

        # Get all part_image_pairs that are inside this parts_list drawing
        for part_count_elem, image_elem in result.part_image_pairs:
            # Check if both elements are inside the parts_list drawing
            if not part_count_elem.bbox.fully_inside(parts_list_drawing.bbox):
                continue
            if not image_elem.bbox.fully_inside(parts_list_drawing.bbox):
                continue

            # Build a Part from this pair
            part = self._build_part_from_pair(part_count_elem, image_elem)
            if part:
                parts.append(part)

        # Return the PartsList
        return PartsList(
            bbox=parts_list_drawing.bbox,
            parts=parts,
        )

    def _build_part_from_pair(
        self, part_count_elem: Element, image_elem: Element
    ) -> Part | None:
        """Build a Part from a part_count and image pair.

        Args:
            part_count_elem: The element labeled as part_count
            image_elem: The element labeled as part_image

        Returns:
            A Part object or None if it couldn't be built
        """
        # Extract count value from part_count element
        if not isinstance(part_count_elem, Text):
            if self._debug_enabled:
                log.debug(
                    "[parts_list] part_count element is not Text: %s",
                    type(part_count_elem).__name__,
                )
            return None

        count_value = extract_part_count_value(part_count_elem.text)
        if count_value is None:
            if self._debug_enabled:
                log.debug(
                    "[parts_list] Could not parse part count from text: '%s'",
                    part_count_elem.text,
                )
            return None

        # Create PartCount object
        part_count = PartCount(
            bbox=part_count_elem.bbox,
            count=count_value,
        )

        # Combine bboxes of part_count and image to get Part bbox
        combined_bbox = BBox(
            x0=min(part_count_elem.bbox.x0, image_elem.bbox.x0),
            y0=min(part_count_elem.bbox.y0, image_elem.bbox.y0),
            x1=max(part_count_elem.bbox.x1, image_elem.bbox.x1),
            y1=max(part_count_elem.bbox.y1, image_elem.bbox.y1),
        )

        # Create Part object
        # TODO: Extract part name and number from nearby text elements
        return Part(
            bbox=combined_bbox,
            name=None,
            number=None,
            count=part_count,
        )

    def classify(
        self,
        page_data: PageData,
        result: ClassificationResult,
        hints: ClassificationHints | None,
    ) -> None:
        # Get pre-built candidates
        candidate_list = result.get_candidates("parts_list")

        # Sort the candidates based on our scoring criteria
        sorted_candidates = sorted(
            candidate_list,
            key=lambda c: (c.score_details.sort_key()),
        )

        # Mark winners (all successfully constructed candidates)
        for candidate in sorted_candidates:
            if candidate.constructed is None:
                # Already has failure_reason from calculate_scores
                continue

            assert isinstance(candidate.constructed, PartsList)

            # Check if this candidate has been removed due to overlap with a
            # previous winner (skip synthetic candidates without source_element)
            if candidate.source_element is not None and result.is_removed(
                candidate.source_element
            ):
                continue

            # This is a winner!
            result.mark_winner(
                candidate, candidate.source_element, candidate.constructed
            )
            if candidate.source_element is not None:
                self.classifier._remove_child_bboxes(
                    page_data, candidate.source_element, result
                )
                self.classifier._remove_similar_bboxes(
                    page_data, candidate.source_element, result
                )
