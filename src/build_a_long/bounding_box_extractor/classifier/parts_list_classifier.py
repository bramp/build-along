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
from typing import TYPE_CHECKING, Any, Dict, Set

from build_a_long.bounding_box_extractor.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.bounding_box_extractor.classifier.part_count_classifier import (
    PartCountClassifier,
)
from build_a_long.bounding_box_extractor.classifier.types import ClassifierConfig
from build_a_long.bounding_box_extractor.extractor import PageData
from build_a_long.bounding_box_extractor.extractor.page_elements import Drawing, Text

if TYPE_CHECKING:
    from build_a_long.bounding_box_extractor.classifier.classifier import Classifier


class PartsListClassifier(LabelClassifier):
    """Classifier for parts lists."""

    def __init__(self, config: ClassifierConfig, classifier: "Classifier"):
        super().__init__(config, classifier)
        # Module-level logger and debug toggle
        self._logger = logging.getLogger(__name__)
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
        # Currently score-free: selection occurs in classify().
        # Kept for interface consistency and future heuristics.
        if self._debug_enabled and self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(
                "[parts_list] page=%s elements=%d",
                page_data.page_number,
                len(page_data.elements),
            )

    def classify(
        self,
        page_data: PageData,
        scores: Dict[Any, Dict[str, float]],
        labeled_elements: Dict[str, Any],
        to_remove: Set[int],
    ) -> None:
        steps = labeled_elements.get("step_number", [])
        if not steps:
            return

        texts: list[Text] = [e for e in page_data.elements if isinstance(e, Text)]
        if not texts:
            return

        drawings: list[Drawing] = [
            e for e in page_data.elements if isinstance(e, Drawing)
        ]
        if not drawings:
            return

        ABOVE_EPS = 2.0
        used_drawings: set[int] = set()
        if "parts_list" not in labeled_elements:
            labeled_elements["parts_list"] = []

        for step in steps:
            sb = step.bbox
            candidates: list[tuple[Drawing, int, float, float]] = []
            for d in drawings:
                if id(d) in used_drawings:
                    continue
                db = d.bbox
                if db.y1 > sb.y0 + ABOVE_EPS:
                    if self._debug_enabled and self._logger.isEnabledFor(logging.DEBUG):
                        self._logger.debug(
                            "[parts_list] reject d=%s: not above step sb=%s (db.y1=%s, sb.y0=%s, eps=%s)",
                            db,
                            sb,
                            db.y1,
                            sb.y0,
                            ABOVE_EPS,
                        )
                    continue

                contained = [
                    t
                    for t in texts
                    if t.bbox.fully_inside(db)
                    and (
                        t in labeled_elements.get("part_count", [])
                        or PartCountClassifier._score_part_count_text(t.text) >= 0.9
                    )
                ]
                if not contained:
                    if self._debug_enabled and self._logger.isEnabledFor(logging.DEBUG):
                        self._logger.debug(
                            "[parts_list] reject d=%s: contains no part_count texts",
                            db,
                        )
                    continue
                count = len(contained)
                proximity = max(0.0, sb.y0 - db.y1)
                area = db.area()
                candidates.append((d, count, proximity, area))

            if not candidates:
                if self._debug_enabled and self._logger.isEnabledFor(logging.DEBUG):
                    self._logger.debug(
                        "[parts_list] no candidates for step sb=%s (drawings=%d)",
                        sb,
                        len(drawings),
                    )
                continue

            candidates.sort(key=lambda x: x[2])
            chosen, _, _, _ = candidates[0]
            labeled_elements["parts_list"].append(chosen)
            used_drawings.add(id(chosen))

            if self._debug_enabled and self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug(
                    "[parts_list] choose d=%s for step sb=%s (candidates=%d)",
                    chosen.bbox,
                    sb,
                    len(candidates),
                )

            keep_ids: Set[int] = set()
            chosen_bbox = chosen.bbox
            for ele in page_data.elements:
                if ele.bbox.fully_inside(chosen_bbox) and any(
                    ele in v for v in labeled_elements.values() if isinstance(v, list)
                ):
                    # Keep labeled children (e.g., part_count texts) inside the chosen parts list.
                    # Unlabeled drawings inside should be eligible for pruning as duplicates/overlays.
                    keep_ids.add(id(ele))

            self.classifier._remove_child_bboxes(page_data, chosen, to_remove, keep_ids)
            self.classifier._remove_similar_bboxes(
                page_data, chosen, to_remove, keep_ids
            )
