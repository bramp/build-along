"""
Page number classifier.
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.text_extractors import (
    extract_page_number_value,
)
from build_a_long.pdf_extract.classifier.types import (
    Candidate,
    ClassificationHints,
    ClassifierConfig,
    RemovalReason,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.lego_page_elements import PageNumber
from build_a_long.pdf_extract.extractor.page_elements import Element, Text


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
        hints: "Optional[ClassificationHints]" = None,
        constructed_elements: "Optional[Dict[Element, Any]]" = None,
        candidates: "Optional[Dict[str, List[Candidate]]]" = None,
    ) -> None:
        if not page_data.elements:
            return

        if constructed_elements is None:
            constructed_elements = {}
        if candidates is None:
            candidates = {}

        # Build candidate list from scored elements
        candidate_list = self._build_candidates(page_data, scores)

        # Apply hints to filter candidates
        candidate_list = self._apply_hints(candidate_list, hints)

        if not candidate_list:
            return

        # Select the winner
        winner = self._select_winner(candidate_list)
        if not winner:
            # All candidates failed to construct - store for diagnostics
            candidates["page_number"] = candidate_list
            return

        # Store results
        labeled_elements[winner.source_element] = "page_number"
        assert isinstance(winner.constructed, PageNumber)
        constructed_elements[winner.source_element] = winner.constructed
        candidates["page_number"] = candidate_list

        # Cleanup: remove child/similar bboxes
        self.classifier._remove_child_bboxes(
            page_data, winner.source_element, removal_reasons
        )
        self.classifier._remove_similar_bboxes(
            page_data, winner.source_element, removal_reasons
        )

    def _build_candidates(
        self, page_data: PageData, scores: Dict[str, Dict[Any, Any]]
    ) -> "List[Candidate]":
        """Build list of all candidate page numbers from scored elements.

        Returns:
            List of Candidate objects, including those that failed to construct.
        """
        page_bbox = page_data.bbox
        assert page_bbox is not None
        page_height = page_bbox.y1 - page_bbox.y0

        page_number_scores = scores.get("page_number", {})
        candidate_list: "List[Candidate]" = []

        for element in page_data.elements:
            if not isinstance(element, Text):
                continue

            # Get the score object and compute combined score
            score_obj = page_number_scores.get(element)
            if not isinstance(score_obj, _PageNumberScore):
                continue

            combined_score = score_obj.combined_score(self.config)

            # Still create candidates for low-scoring elements (for debugging/testing)
            # but they won't be selected as winners
            meets_threshold = combined_score >= self.config.min_confidence_threshold

            # Require the element to be in the bottom band of the page
            position_score = self._score_page_number_position(
                element, page_bbox, page_height
            )
            in_position = position_score > 0.0

            # Skip if both conditions fail
            if not meets_threshold and not in_position:
                continue

            # Try to construct the LegoElement (parse the text)
            value = extract_page_number_value(element.text)
            constructed_elem = None
            failure_reason = None

            if value is not None:
                constructed_elem = PageNumber(
                    value=value, bbox=element.bbox, id=element.id
                )
            else:
                failure_reason = (
                    f"Could not parse page number from text: '{element.text}'"
                )

            # Store candidate even if construction failed (for debugging)
            # Winner will be selected later by _select_winner()
            candidate_list.append(
                Candidate(
                    source_element=element,
                    label="page_number",
                    score=combined_score,
                    score_details=score_obj,
                    constructed=constructed_elem,
                    failure_reason=failure_reason,
                    is_winner=False,  # Will be set by _select_winner
                )
            )

        return candidate_list

    def _apply_hints(
        self,
        candidate_list: "List[Candidate]",
        hints: "Optional[ClassificationHints]",
    ) -> "List[Candidate]":
        """Apply hints to filter candidate list.

        Args:
            candidate_list: List of candidates to filter
            hints: Optional classification hints with element constraints

        Returns:
            Filtered list of candidates that match hint constraints.
        """
        if not hints or not hints.element_constraints:
            return candidate_list

        # Filter: keep only candidates that match constraints
        return [
            c
            for c in candidate_list
            if id(c.source_element) not in hints.element_constraints
            or hints.element_constraints[id(c.source_element)] == "page_number"
        ]

    def _select_winner(
        self, candidate_list: "List[Candidate]"
    ) -> "Optional[Candidate]":
        """Select the best candidate from the list.

        Only considers candidates that successfully constructed a PageNumber.
        Selects the highest scoring candidate and marks it as winner.

        Args:
            candidate_list: List of candidates to choose from

        Returns:
            The winning candidate, or None if no valid candidates exist.
        """
        # Choose best candidate that successfully constructed
        valid_candidates = [c for c in candidate_list if c.constructed is not None]
        if not valid_candidates:
            return None

        # Sort by score and pick winner
        best = max(valid_candidates, key=lambda c: c.score)
        best.is_winner = True
        return best

    def _score_page_number_text(self, text: str) -> float:
        # TODO The score should increase if the text matches the actual page we
        # are on.
        text = text.strip()
        if re.match(r"^0+\d{1,3}$", text):
            return 0.95
        if re.match(r"^\d{1,3}$", text):
            return 1.0
        return 0.0

    def _score_page_number(self, text: str, page_number: int) -> float:
        """Score how well the text matches the expected page number."""

        value = extract_page_number_value(text)
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
