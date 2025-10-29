"""
Step number classifier.
"""

import re
from dataclasses import dataclass
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


@dataclass
class _StepNumberScore:
    """Internal score representation for step number classification."""

    text_score: float
    """Score based on how well the text matches step number patterns (0.0-1.0)."""

    size_score: float
    """Score based on element height relative to page number height (0.0-1.0)."""

    def combined_score(self, config: ClassifierConfig) -> float:
        """Calculate final weighted score from components."""
        # Sum the weighted components
        score = (
            config.step_number_text_weight * self.text_score
            + config.step_number_size_weight * self.size_score
        )
        # Normalize by the sum of weights to keep score in [0, 1]
        total_weight = config.step_number_text_weight + config.step_number_size_weight
        return score / total_weight if total_weight > 0 else 0.0


class StepNumberClassifier(LabelClassifier):
    """Classifier for step numbers."""

    outputs = {"step_number"}
    requires = {"page_number"}

    def __init__(self, config: ClassifierConfig, classifier: "Classifier"):
        super().__init__(config, classifier)
        # Store detailed scores for internal use
        self._detail_scores: Dict[Any, _StepNumberScore] = {}

    def _score_step_number_size(
        self, element: Text, page_num_height: Optional[float]
    ) -> float:
        """Score based on element height relative to page number height.

        Returns 0.0 if element is not significantly taller than page number,
        scaling up to 1.0 as element gets taller.
        """
        if not page_num_height or page_num_height <= 0.0:
            return 0.0

        h = max(0.0, element.bbox.y1 - element.bbox.y0)
        if h <= page_num_height * 1.1:
            return 0.0

        ratio_over = (h / page_num_height) - 1.0
        return max(0.0, min(1.0, ratio_over / 0.5))

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

        page_num_height: Optional[float] = None
        page_number_element = labeled_elements.get("page_number")
        if page_number_element:
            page_num_height = max(
                0.0, page_number_element.bbox.y1 - page_number_element.bbox.y0
            )

        # Get page bbox and height for bottom band check
        page_bbox = page_data.bbox
        assert page_bbox is not None
        page_height = page_bbox.y1 - page_bbox.y0

        for element in page_data.elements:
            if not isinstance(element, Text):
                continue

            # Skip elements in the bottom 10% of the page where page numbers typically appear
            element_center_y = (element.bbox.y0 + element.bbox.y1) / 2
            bottom_threshold = page_bbox.y1 - (page_height * 0.1)
            if element_center_y >= bottom_threshold:
                continue

            text_score = self._score_step_number_text(element.text)
            if text_score == 0.0:
                continue

            size_score = self._score_step_number_size(element, page_num_height)

            # If we have a page number for size comparison, require the element to be
            # taller than the page number (size_score > 0). This prevents small
            # numeric text from being classified as step numbers.
            if page_num_height and size_score == 0.0:
                continue

            # Store detailed score
            detail_score = _StepNumberScore(
                text_score=text_score,
                size_score=size_score,
            )
            self._detail_scores[element] = detail_score

            # Calculate combined score
            final = detail_score.combined_score(self.config)

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

        # Get the page number element to avoid classifying it as a step number
        page_number_element = labeled_elements.get("page_number")

        for element in page_data.elements:
            if not isinstance(element, Text):
                continue

            # Skip if this element is already labeled as the page number
            if element is page_number_element:
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
