"""
Page number classifier.
"""

import math
import re
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

from build_a_long.bounding_box_extractor.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.bounding_box_extractor.classifier.types import ClassifierConfig
from build_a_long.bounding_box_extractor.extractor import PageData
from build_a_long.bounding_box_extractor.extractor.page_elements import Text

if TYPE_CHECKING:
    from build_a_long.bounding_box_extractor.classifier.classifier import Classifier


class PageNumberClassifier(LabelClassifier):
    """Classifier for page numbers."""

    def __init__(self, config: ClassifierConfig, classifier: "Classifier"):
        super().__init__(config, classifier)

    def calculate_scores(
        self,
        page_data: PageData,
        scores: Dict[Any, Dict[str, float]],
        labeled_elements: Dict[str, Any],
    ) -> None:
        if not page_data.elements:
            return

        page_bbox = page_data.bbox
        assert page_bbox is not None
        page_height = page_bbox.y1 - page_bbox.y0

        for element in page_data.elements:
            if not isinstance(element, Text):
                continue

            text_score = self._score_page_number_text(element.text)
            position_score = self._score_page_number_position(
                element, page_bbox, page_height
            )

            final_score = (
                self.config.page_number_text_weight * text_score
                + self.config.page_number_position_weight * position_score
            )

            if element not in scores:
                scores[element] = {}
            scores[element]["page_number"] = final_score

    def classify(
        self,
        page_data: PageData,
        scores: Dict[Any, Dict[str, float]],
        labeled_elements: Dict[str, Any],
        to_remove: Set[int],
    ) -> None:
        if not page_data.elements:
            return

        candidates: list[tuple[Text, float, Optional[int]]] = []
        for element in page_data.elements:
            if not isinstance(element, Text):
                continue
            score = scores.get(element, {}).get("page_number", 0.0)
            if score < self.config.min_confidence_threshold:
                continue
            value = self._extract_page_number_value(element.text)
            candidates.append((element, score, value))

        matching = [c for c in candidates if c[2] == page_data.page_number]
        chosen: Optional[tuple[Text, float, Optional[int]]] = None
        if matching:
            chosen = max(matching, key=lambda c: c[1])
        elif candidates:
            chosen = max(candidates, key=lambda c: c[1])

        if chosen is None:
            return

        best_candidate, _, _ = chosen
        labeled_elements["page_number"] = best_candidate

        self.classifier._remove_child_bboxes(page_data, best_candidate, to_remove)
        self.classifier._remove_similar_bboxes(page_data, best_candidate, to_remove)

    def _score_page_number_text(self, text: str) -> float:
        text = text.strip()
        if re.match(r"^0+\d{1,3}$", text):
            return 0.95
        if re.match(r"^\d{1,3}$", text):
            return 1.0
        return 0.0

    def _score_page_number_position(
        self, element: Text, page_bbox, page_height: float
    ) -> float:
        bottom_threshold = page_bbox.y1 - (page_height * 0.1)
        element_center_y = (element.bbox.y0 + element.bbox.y1) / 2

        if element_center_y < bottom_threshold:
            return 0.0

        element_center_x = (element.bbox.x0 + element.bbox.x1) / 2
        dist_bottom_left = math.sqrt(
            (element_center_x - page_bbox.x0) ** 2
            + (element_center_y - page_bbox.y1) ** 2
        )
        dist_bottom_right = math.sqrt(
            (element_center_x - page_bbox.x1) ** 2
            + (element_center_y - page_bbox.y1) ** 2
        )
        min_dist = min(dist_bottom_left, dist_bottom_right)
        position_score = math.exp(-min_dist / self.config.page_number_position_scale)
        return position_score

    def _extract_page_number_value(self, text: str) -> Optional[int]:
        t = text.strip()
        m = re.match(r"^0*(\d{1,3})$", t)
        if m:
            return int(m.group(1))
        m = re.match(r"^(?:page|p\.?)\s*0*(\d{1,3})$", t, re.IGNORECASE)
        if m:
            return int(m.group(1))
        return None
