"""
Page number classifier.
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.types import (
    ClassifierConfig,
    RemovalReason,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.page_elements import Text


@dataclass
class _PageNumberScore:
    """Internal score representation for page number classification."""

    text_score: float
    """Score based on how well the text matches page number patterns (0.0-1.0)."""

    position_score: float
    """Score based on position in bottom corners of page (0.0-1.0)."""

    page_value_score: float
    """Score based on how well the text matches the expected page number
    (0.0-1.0)."""

    # TODO Test this score is always between 0.0 and 1.0
    def combined_score(self, config: ClassifierConfig) -> float:
        """Calculate final weighted score from components."""
        # Sum the weighted components
        score = (
            config.page_number_text_weight * self.text_score
            + config.page_number_position_weight * self.position_score
            + config.page_number_page_value_weight * self.page_value_score
        )
        # Normalize by the sum of weights to keep score in [0, 1]
        total_weight = (
            config.page_number_text_weight
            + config.page_number_position_weight
            + config.page_number_page_value_weight
        )
        return score / total_weight if total_weight > 0 else 0.0


class PageNumberClassifier(LabelClassifier):
    """Classifier for page numbers."""

    outputs = {"page_number"}
    requires = set()

    def calculate_scores(
        self,
        page_data: PageData,
        scores: Dict[str, Dict[Any, Any]],
        labeled_elements: Dict[Any, str],
    ) -> None:
        if not page_data.elements:
            return

        page_bbox = page_data.bbox
        assert page_bbox is not None

        # TODO add height to bbox and use it here.
        page_height = page_bbox.y1 - page_bbox.y0

        # Initialize scores dict for this classifier
        if "page_number" not in scores:
            scores["page_number"] = {}

        for element in page_data.elements:
            if not isinstance(element, Text):
                continue

            # Calculate all score components
            text_score = self._score_page_number_text(element.text)
            position_score = self._score_page_number_position(
                element, page_bbox, page_height
            )
            page_value_score = self._score_page_number(
                element.text, page_data.page_number
            )

            # Create detailed score object
            page_score = _PageNumberScore(
                text_score=text_score,
                position_score=position_score,
                page_value_score=page_value_score,
            )

            # Store the score object directly in the scores dict
            scores["page_number"][element] = page_score

    def classify(
        self,
        page_data: PageData,
        scores: Dict[str, Dict[Any, Any]],
        labeled_elements: Dict[Any, str],
        removal_reasons: Dict[int, RemovalReason],
    ) -> None:
        if not page_data.elements:
            return

        page_bbox = page_data.bbox
        assert page_bbox is not None
        page_height = page_bbox.y1 - page_bbox.y0

        # Get pre-calculated scores for this classifier
        page_number_scores = scores.get("page_number", {})

        # Build list of candidates from pre-calculated scores
        candidates: list[tuple[Text, float]] = []
        for element in page_data.elements:
            if not isinstance(element, Text):
                continue

            # Get the score object and compute combined score
            score_obj = page_number_scores.get(element)
            if not isinstance(score_obj, _PageNumberScore):
                continue

            combined_score = score_obj.combined_score(self.config)
            if combined_score < self.config.min_confidence_threshold:
                continue

            # Require the element to be in the bottom band of the page; otherwise
            # it is very likely a step number or other numeric text.
            if self._score_page_number_position(element, page_bbox, page_height) == 0.0:
                continue

            candidates.append((element, combined_score))

        if not candidates:
            return

        best_candidate, _ = max(candidates, key=lambda c: c[1])

        labeled_elements[best_candidate] = "page_number"

        self.classifier._remove_child_bboxes(page_data, best_candidate, removal_reasons)
        self.classifier._remove_similar_bboxes(
            page_data, best_candidate, removal_reasons
        )

    def _score_page_number_text(self, text: str) -> float:
        # TODO The score should increase if the text matches the actual page we
        # are on.
        text = text.strip()
        if re.match(r"^0+\d{1,3}$", text):
            return 0.95
        if re.match(r"^\d{1,3}$", text):
            return 1.0
        return 0.0

    def _extract_page_number_value(self, text: str) -> Optional[int]:
        t = text.strip()
        m = re.match(r"^0*(\d{1,3})$", t)
        if m:
            return int(m.group(1))
        m = re.match(r"^(?:page|p\.?)\s*0*(\d{1,3})$", t, re.IGNORECASE)
        if m:
            return int(m.group(1))
        return None

    def _score_page_number(self, text: str, page_number: int) -> float:
        """Score how well the text matches the expected page number."""

        value = self._extract_page_number_value(text)
        if value is None:
            # Can't extract text.
            return 0.0

        # For every digit away from the expected page number,
        # reduce score by 10%
        diff = abs(value - page_number)
        return max(0.0, 1.0 - 0.1 * diff)

    def _is_in_bottom_band(self, element: Text, page_bbox, page_height: float) -> bool:
        """Check if the element is in the bottom 10% of the page height."""
        bottom_threshold = page_bbox.y1 - (page_height * 0.1)
        element_center_y = (element.bbox.y0 + element.bbox.y1) / 2
        return element_center_y >= bottom_threshold

    def _calculate_position_score(self, element: Text, page_bbox) -> float:
        """Calculate position score based on distance to bottom corners. Based
        on exp(-min_distance_to_bottom_corners / scale)"""
        element_center_x = (element.bbox.x0 + element.bbox.x1) / 2
        element_center_y = (element.bbox.y0 + element.bbox.y1) / 2
        dist_bottom_left = math.sqrt(
            (element_center_x - page_bbox.x0) ** 2
            + (element_center_y - page_bbox.y1) ** 2
        )
        dist_bottom_right = math.sqrt(
            (element_center_x - page_bbox.x1) ** 2
            + (element_center_y - page_bbox.y1) ** 2
        )
        min_dist = min(dist_bottom_left, dist_bottom_right)
        return math.exp(-min_dist / self.config.page_number_position_scale)

    def _score_page_number_position(
        self, element: Text, page_bbox, page_height: float
    ) -> float:
        # TODO Take the hint, and increase score if near expected position (of
        # expected size).

        if not self._is_in_bottom_band(element, page_bbox, page_height):
            return 0.0

        # TODO it might be simplier to check if in left or right band.
        return self._calculate_position_score(element, page_bbox)
