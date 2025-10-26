"""
Part count classifier.

Purpose
-------
Detect part-count text like "2x", "3X", or "5×".

Debugging
---------
Enable DEBUG logs with LOG_LEVEL=DEBUG. Heavier trace can be enabled when
CLASSIFIER_DEBUG is set to "part_count" or "all".
"""

import logging
import re
import os
from typing import TYPE_CHECKING, Any, Dict, Set

from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.types import ClassifierConfig
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.page_elements import Text

if TYPE_CHECKING:
    from build_a_long.pdf_extract.classifier.classifier import Classifier


class PartCountClassifier(LabelClassifier):
    """Classifier for part counts."""

    outputs = {"part_count"}
    requires = set()

    def __init__(self, config: ClassifierConfig, classifier: "Classifier"):
        super().__init__(config, classifier)
        self._logger = logging.getLogger(__name__)
        self._debug_enabled = os.getenv("CLASSIFIER_DEBUG", "").lower() in (
            "part_count",
            "all",
        )

    def calculate_scores(
        self,
        page_data: PageData,
        scores: Dict[Any, Dict[str, float]],
        labeled_elements: Dict[str, Any],
    ) -> None:
        if not page_data.elements:
            return

        for element in page_data.elements:
            if not isinstance(element, Text):
                continue
            score = PartCountClassifier._score_part_count_text(element.text)
            if score > 0.0:
                if element not in scores:
                    scores[element] = {}
                scores[element]["part_count"] = score
                if self._debug_enabled and self._logger.isEnabledFor(logging.DEBUG):
                    self._logger.debug(
                        "[part_count] match text=%r score=%.2f bbox=%s",
                        element.text,
                        score,
                        element.bbox,
                    )

    def classify(
        self,
        page_data: PageData,
        scores: Dict[Any, Dict[str, float]],
        labeled_elements: Dict[str, Any],
        to_remove: Set[int],
    ) -> None:
        if "part_count" not in labeled_elements:
            labeled_elements["part_count"] = []

        for element in page_data.elements:
            if not isinstance(element, Text):
                continue
            score = scores.get(element, {}).get("part_count", 0.0)
            if score >= self.config.min_confidence_threshold:
                labeled_elements["part_count"].append(element)
                self.classifier._remove_child_bboxes(page_data, element, to_remove)
                self.classifier._remove_similar_bboxes(page_data, element, to_remove)

    @staticmethod
    def _score_part_count_text(text: str) -> float:
        t = text.strip()
        if re.fullmatch(r"\d{1,3}\s*[x×]", t, flags=re.IGNORECASE):
            return 1.0
        return 0.0
