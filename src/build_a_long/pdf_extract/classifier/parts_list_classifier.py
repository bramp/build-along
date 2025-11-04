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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set

if TYPE_CHECKING:
    from build_a_long.pdf_extract.classifier.types import (
        Candidate,
        ClassificationHints,
    )
    from build_a_long.pdf_extract.extractor.lego_page_elements import LegoPageElement

from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.types import (
    Candidate,
    ClassifierConfig,
    RemovalReason,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_elements import (
    Drawing,
    Element,
    Image,
    Text,
)

log = logging.getLogger(__name__)


# TODO None of these are scores, change to scores.
@dataclass
class _PartsListScore:
    """Internal score representation for parts list classification."""

    part_count_count: int
    """Number of part count texts contained in the bounding box."""

    proximity: float
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
        return (self.part_count_count, self.proximity, self.area)


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
        labeled_elements: Dict[Element, str],
        candidates: "Dict[str, List[Candidate]]",
    ) -> None:
        """Evaluate elements and create candidates for potential parts list drawings.

        Scores drawings based on their proximity to step numbers and the number
        of part count texts they contain. Creates candidates for all viable
        parts list drawings.
        """

        # Get elements with specific labels
        steps: list[Text] = []
        for element, label in labeled_elements.items():
            if label == "step_number" and isinstance(element, Text):
                steps.append(element)
        if not steps:
            return

        part_counts: list[Element] = []
        for element, label in labeled_elements.items():
            if label == "part_count":
                part_counts.append(element)
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

        candidate_list: "List[Candidate]" = []

        # Score each drawing relative to each step
        for step in steps:
            sb = step.bbox
            candidate_drawings = self._get_candidate_drawings_for_scoring(
                step, drawings
            )

            for drawing in candidate_drawings:
                score_obj = self._score_candidate(
                    drawing, part_counts, labeled_elements, sb
                )

                failure_reason = None
                if score_obj is None:
                    # Drawing was rejected (doesn't contain part counts or other reason)
                    failure_reason = (
                        "Drawing contains no part_count texts or is not above step"
                    )

                # Create candidate (even if rejected, for debugging)
                candidate = Candidate(
                    source_element=drawing,
                    label="parts_list",
                    score=1.0
                    if score_obj
                    else 0.0,  # Parts list uses ranking rather than scores
                    score_details=score_obj,
                    constructed=None,  # PartsList construction requires part pairing, done in builder
                    failure_reason=failure_reason,
                    is_winner=False,  # Will be set by classify()
                )
                candidate_list.append(candidate)

        # Store all candidates
        candidates["parts_list"] = candidate_list

    def _get_candidate_drawings_for_scoring(
        self,
        step: Text,
        drawings: list[Drawing],
    ) -> list[Drawing]:
        """Get candidate drawings above the given step number (for scoring phase)."""
        ABOVE_EPS = 2.0
        sb = step.bbox
        candidates = []
        for d in drawings:
            db = d.bbox
            if db.y1 > sb.y0 + ABOVE_EPS:
                if self._debug_enabled:
                    log.debug(
                        "[parts_list] reject d=%s: not above step sb=%s (db.y1=%s, sb.y0=%s, eps=%s)",
                        db,
                        sb,
                        db.y1,
                        sb.y0,
                        ABOVE_EPS,
                    )
                continue
            candidates.append(d)
        return candidates

    def _score_candidate(
        self,
        drawing: Drawing,
        part_counts: list[Element],
        labeled_elements: Dict[Element, str],
        sb: BBox,
    ) -> Optional[_PartsListScore]:
        """Score a candidate drawing for parts list classification.

        Returns None if the drawing should be rejected, otherwise returns a score object.
        """
        contained = [
            part_count
            for part_count in part_counts
            if part_count.bbox.fully_inside(drawing.bbox)
        ]
        if not contained:
            if self._debug_enabled:
                log.debug(
                    "[parts_list] reject d=%s: contains no part_count texts",
                    drawing.bbox,
                )
            return None

        # TODO Change this to a score.
        count = len(contained)
        # TODO Change this to a score (normalised by page height)
        proximity = max(0.0, sb.y0 - drawing.bbox.y1)

        score = _PartsListScore(
            part_count_count=count,
            proximity=proximity,
            area=drawing.bbox.area,
        )

        return score

    def classify(
        self,
        page_data: PageData,
        labeled_elements: Dict[Element, str],
        removal_reasons: Dict[int, RemovalReason],
        hints: Optional["ClassificationHints"],
        constructed_elements: "Dict[Element, LegoPageElement]",
        candidates: "Dict[str, List[Candidate]]",
    ) -> None:
        """Select winning parts list drawings from pre-built candidates."""
        # Get elements with step_number label
        steps: list[Text] = []
        for element, label in labeled_elements.items():
            if label == "step_number" and isinstance(element, Text):
                steps.append(element)
        if not steps:
            return

        drawings: list[Drawing] = [
            e for e in page_data.elements if isinstance(e, Drawing)
        ]
        if not drawings:
            return

        used_drawings: set[int] = set()

        # Get pre-built candidates
        candidate_list = candidates.get("parts_list", [])

        # Group candidates by step (based on scoring)
        # For each step, find the best parts list drawing
        for step in steps:
            # Get candidates for this step that haven't been used
            step_candidates = []
            for candidate in candidate_list:
                if (
                    id(candidate.source_element) not in used_drawings
                    and id(candidate.source_element) not in removal_reasons
                    and candidate.score_details is not None  # Has valid score
                ):
                    # Check if this candidate is above this step
                    drawing = candidate.source_element
                    if isinstance(drawing, Drawing):
                        db = drawing.bbox
                        sb = step.bbox
                        ABOVE_EPS = 2.0
                        if db.y1 <= sb.y0 + ABOVE_EPS:
                            step_candidates.append(candidate)

            if not step_candidates:
                continue

            # Sort by score details (part_count_count, proximity, area)
            step_candidates.sort(key=lambda c: c.score_details.sort_key())

            # Select the best candidate
            winner = step_candidates[0]
            winner.is_winner = True
            labeled_elements[winner.source_element] = "parts_list"
            used_drawings.add(id(winner.source_element))

            if self._debug_enabled:
                log.debug(
                    "[parts_list] choose d=%s for step sb=%s (candidates=%d)",
                    winner.source_element.bbox,
                    step.bbox,
                    len(step_candidates),
                )

            # Keep child elements that are already labeled or are images
            keep_ids: Set[int] = set()
            chosen_bbox = winner.source_element.bbox
            for ele in page_data.elements:
                if (
                    ele.bbox.fully_inside(chosen_bbox)
                    and labeled_elements.get(ele) is not None
                ):
                    keep_ids.add(id(ele))

            for ele in page_data.elements:
                if isinstance(ele, Image) and ele.bbox.fully_inside(chosen_bbox):
                    keep_ids.add(id(ele))

            self.classifier._remove_child_bboxes(
                page_data, winner.source_element, removal_reasons, keep_ids=keep_ids
            )
            self.classifier._remove_similar_bboxes(
                page_data, winner.source_element, removal_reasons, keep_ids=keep_ids
            )
