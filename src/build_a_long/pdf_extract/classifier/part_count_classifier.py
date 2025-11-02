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
from typing import Any, Dict

from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.types import (
    ClassifierConfig,
    RemovalReason,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.page_elements import Text

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

    def __init__(self, config: ClassifierConfig, classifier):
        super().__init__(config, classifier)
        self._debug_enabled = os.getenv("CLASSIFIER_DEBUG", "").lower() in (
            "part_count",
            "all",
        )

    def calculate_scores(
        self,
        page_data: PageData,
        scores: Dict[str, Dict[Any, Any]],
        labeled_elements: Dict[Any, str],
    ) -> None:
        if not page_data.elements:
            return

        # Initialize scores dict for this classifier
        if "part_count" not in scores:
            scores["part_count"] = {}

        for element in page_data.elements:
            if not isinstance(element, Text):
                continue

            text_score = PartCountClassifier._score_part_count_text(element.text)
            if text_score == 0.0:
                continue

            # Store detailed score object
            detail_score = _PartCountScore(text_score=text_score)
            scores["part_count"][element] = detail_score

            if self._debug_enabled:
                log.debug(
                    "[part_count] match text=%r score=%.2f bbox=%s",
                    element.text,
                    text_score,
                    element.bbox,
                )

    def classify(
        self,
        page_data: PageData,
        scores: Dict[str, Dict[Any, Any]],
        labeled_elements: Dict[Any, str],
        removal_reasons: Dict[int, RemovalReason],
    ) -> None:
        # Get pre-calculated scores for this classifier
        part_count_scores = scores.get("part_count", {})

        for element in page_data.elements:
            if not isinstance(element, Text):
                continue

            # Get the score object and compute combined score
            score_obj = part_count_scores.get(element)
            if not isinstance(score_obj, _PartCountScore):
                continue

            combined_score = score_obj.combined_score(self.config)
            if combined_score >= self.config.min_confidence_threshold:
                labeled_elements[element] = "part_count"
                self.classifier._remove_child_bboxes(
                    page_data, element, removal_reasons
                )
                self.classifier._remove_similar_bboxes(
                    page_data, element, removal_reasons
                )

    @staticmethod
    def _score_part_count_text(text: str) -> float:
        t = text.strip()
        if re.fullmatch(r"\d{1,3}\s*[x×]", t, flags=re.IGNORECASE):
            return 1.0
        return 0.0
