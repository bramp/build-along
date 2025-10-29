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
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

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

log = logging.getLogger(__name__)


@dataclass
class _PartCountScore:
    """Internal score representation for part count classification."""

    text_score: float
    """Score based on how well the text matches part count patterns (0.0-1.0)."""

    def combined_score(self, config: ClassifierConfig) -> float:
        """Calculate final weighted score from components.

        For part count, we only have text_score, so just return it directly.
        """
        return self.text_score


class PartCountClassifier(LabelClassifier):
    """Classifier for part counts."""

    outputs = {"part_count"}
    requires = set()

    def __init__(self, config: ClassifierConfig, classifier: "Classifier"):
        super().__init__(config, classifier)
        # Store detailed scores for internal use
        self._detail_scores: Dict[Any, _PartCountScore] = {}
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

        # Clear previous detail scores for this page
        self._detail_scores.clear()

        for element in page_data.elements:
            if not isinstance(element, Text):
                continue

            text_score = PartCountClassifier._score_part_count_text(element.text)
            if text_score == 0.0:
                continue

            # Store detailed score
            detail_score = _PartCountScore(text_score=text_score)
            self._detail_scores[element] = detail_score

            # Calculate combined score
            final = detail_score.combined_score(self.config)

            if element not in scores:
                scores[element] = {}
            scores[element]["part_count"] = final

            if self._debug_enabled:
                log.debug(
                    "[part_count] match text=%r score=%.2f bbox=%s",
                    element.text,
                    final,
                    element.bbox,
                )

    def classify(
        self,
        page_data: PageData,
        scores: Dict[Any, Dict[str, float]],
        labeled_elements: Dict[str, Any],
        to_remove: Dict[int, RemovalReason],
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
