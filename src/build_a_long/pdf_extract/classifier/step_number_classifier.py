"""
Step number classifier.
"""

import re
from typing import TYPE_CHECKING, Any, Dict, Optional

from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.types import (
    ClassifierConfig,
    RemovalReason,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.page_elements import Text

if TYPE_CHECKING:
    from build_a_long.pdf_extract.classifier.classifier import Classifier


class StepNumberClassifier(LabelClassifier):
    """Classifier for step numbers."""

    outputs = {"step_number"}
    requires = {"page_number"}

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

        page_num_height: Optional[float] = None
        page_number_element = labeled_elements.get("page_number")
        if page_number_element:
            page_num_height = max(
                0.0, page_number_element.bbox.y1 - page_number_element.bbox.y0
            )

        for element in page_data.elements:
            if not isinstance(element, Text):
                continue
            text_score = self._score_step_number_text(element.text)
            if text_score == 0.0:
                continue

            size_score = 0.0
            if page_num_height and page_num_height > 0.0:
                h = max(0.0, element.bbox.y1 - element.bbox.y0)
                if h <= page_num_height * 1.1:
                    final = 0.0
                    if element not in scores:
                        scores[element] = {}
                    scores[element]["step_number"] = final
                    continue
                ratio_over = (h / page_num_height) - 1.0
                size_score = max(0.0, min(1.0, ratio_over / 0.5))

            final = (
                self.config.step_number_text_weight * text_score
                + self.config.step_number_size_weight * size_score
            )
            if element not in scores:
                scores[element] = {}
            scores[element]["step_number"] = final

    def classify(
        self,
        page_data: PageData,
        scores: Dict[Any, Dict[str, float]],
        labeled_elements: Dict[str, Any],
        to_remove: Dict[int, RemovalReason],
    ) -> None:
        if "step_number" not in labeled_elements:
            labeled_elements["step_number"] = []

        for element in page_data.elements:
            if not isinstance(element, Text):
                continue
            score = scores.get(element, {}).get("step_number", 0.0)
            if score >= self.config.min_confidence_threshold:
                labeled_elements["step_number"].append(element)
                self.classifier._remove_child_bboxes(page_data, element, to_remove)
                self.classifier._remove_similar_bboxes(page_data, element, to_remove)

    def _score_step_number_text(self, text: str) -> float:
        t = text.strip()
        if re.fullmatch(r"[1-9]\d{0,3}", t):
            return 1.0
        return 0.0
