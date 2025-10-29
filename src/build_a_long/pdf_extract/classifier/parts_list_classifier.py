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
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.types import (
    ClassifierConfig,
    RemovalReason,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.page_elements import (
    Drawing,
    Element,
    Image,
    Text,
)

if TYPE_CHECKING:
    from build_a_long.pdf_extract.classifier.classifier import Classifier

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

    def __init__(self, config: ClassifierConfig, classifier: "Classifier"):
        super().__init__(config, classifier)
        # Store detailed scores for internal use
        self._detail_scores: Dict[Any, _PartsListScore] = {}
        self._debug_enabled = os.getenv("CLASSIFIER_DEBUG", "").lower() in (
            "parts_list",
            "all",
        )

    def calculate_scores(
        self,
        page_data: PageData,
        scores: Dict[Any, Dict[str, float]],
        labeled_elements: Dict[str, Any],
    ) -> None:
        """Calculate scores for potential parts list drawings.

        Scores drawings based on their proximity to step numbers and the number
        of part count texts they contain.
        """
        # Clear previous detail scores for this page
        self._detail_scores.clear()

        steps = labeled_elements.get("step_number", [])
        if not steps:
            return

        part_counts = labeled_elements.get("part_count", [])
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

        # Score each drawing relative to each step
        for step in steps:
            sb = step.bbox
            candidate_drawings = self._get_candidate_drawings(step, drawings, set(), {})

            for drawing in candidate_drawings:
                score_obj = self._score_candidate(
                    drawing, part_counts, labeled_elements, sb
                )
                if score_obj is not None:
                    # We don't have a single numeric score for parts_list in the traditional sense.
                    # The score object contains multiple factors for sorting.
                    # Store a placeholder score in the scores dict for consistency.
                    if drawing not in scores:
                        scores[drawing] = {}
                    # Use the negative of part_count_count as a simple numeric score
                    # (negative because more part counts = better = higher "confidence")
                    scores[drawing]["parts_list"] = float(score_obj.part_count_count)

    def _find_best_parts_list(
        self, candidates: list[tuple[Drawing, _PartsListScore]]
    ) -> Optional[Drawing]:
        """Select the best parts list drawing from candidates."""
        if not candidates:
            return None
        # Sort by the score's sort_key (part_count_count, proximity, area)
        candidates.sort(key=lambda x: x[1].sort_key())
        return candidates[0][0]

    def _score_candidate(
        self,
        drawing: Drawing,
        part_counts: list[Element],
        labeled_elements: Dict[str, Any],
        sb,
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
        area = drawing.bbox.area()

        score = _PartsListScore(
            part_count_count=count,
            proximity=proximity,
            area=area,
        )

        # Store detailed score
        self._detail_scores[drawing] = score

        return score

    def _get_candidate_drawings(
        self,
        step: Text,
        drawings: list[Drawing],
        used_drawings: set[int],
        to_remove: Dict[int, RemovalReason],
    ) -> list[Drawing]:
        """Get candidate drawings above the given step number."""
        ABOVE_EPS = 2.0
        sb = step.bbox
        candidates = []
        for d in drawings:
            if id(d) in used_drawings or id(d) in to_remove:
                continue
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

    def classify(
        self,
        page_data: PageData,
        scores: Dict[Any, Dict[str, float]],
        labeled_elements: Dict[str, Any],
        to_remove: Dict[int, RemovalReason],
    ) -> None:
        steps = labeled_elements.get("step_number", [])
        if not steps:
            return

        drawings: list[Drawing] = [
            e for e in page_data.elements if isinstance(e, Drawing)
        ]
        if not drawings:
            return

        used_drawings: set[int] = set()
        if "parts_list" not in labeled_elements:
            labeled_elements["parts_list"] = []

        for step in steps:
            candidate_drawings = self._get_candidate_drawings(
                step, drawings, used_drawings, to_remove
            )

            # Get scored candidates from _detail_scores (populated in calculate_scores)
            scored_candidates = []
            for d in candidate_drawings:
                if d in self._detail_scores:
                    scored_candidates.append((d, self._detail_scores[d]))

            chosen = self._find_best_parts_list(scored_candidates)

            if chosen:
                labeled_elements["parts_list"].append(chosen)
                used_drawings.add(id(chosen))

                if self._debug_enabled:
                    log.debug(
                        "[parts_list] choose d=%s for step sb=%s (candidates=%d)",
                        chosen.bbox,
                        step.bbox,
                        len(scored_candidates),
                    )

                keep_ids: Set[int] = set()
                chosen_bbox = chosen.bbox
                for ele in page_data.elements:
                    if ele.bbox.fully_inside(chosen_bbox) and any(
                        ele in v
                        for v in labeled_elements.values()
                        if isinstance(v, list)
                    ):
                        keep_ids.add(id(ele))

                for ele in page_data.elements:
                    if isinstance(ele, Image) and ele.bbox.fully_inside(chosen_bbox):
                        keep_ids.add(id(ele))

                self.classifier._remove_child_bboxes(
                    page_data, chosen, to_remove, keep_ids=keep_ids
                )
                self.classifier._remove_similar_bboxes(
                    page_data, chosen, to_remove, keep_ids=keep_ids
                )
