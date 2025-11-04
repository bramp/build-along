"""
Step number classifier.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from build_a_long.pdf_extract.classifier.types import (
        Candidate,
        ClassificationHints,
    )
    from build_a_long.pdf_extract.extractor.lego_page_elements import LegoPageElement
    from build_a_long.pdf_extract.extractor.page_elements import Element

from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.text_extractors import (
    extract_step_number_value,
)
from build_a_long.pdf_extract.classifier.types import (
    Candidate,
    ClassifierConfig,
    RemovalReason,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.lego_page_elements import StepNumber
from build_a_long.pdf_extract.extractor.page_elements import Text


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

    def _score_step_number_size(
        self, element: Text, page_num_height: Optional[float]
    ) -> float:
        """Score based on element height relative to page number height.

        Returns 0.0 if element is not significantly taller than page number,
        scaling up to 1.0 as element gets taller.
        """
        if not page_num_height or page_num_height <= 0.0:
            return 0.0

        h = element.bbox.height
        if h <= page_num_height * 1.1:
            return 0.0

        ratio_over = (h / page_num_height) - 1.0
        return max(0.0, min(1.0, ratio_over / 0.5))

    def calculate_scores(
        self,
        page_data: PageData,
        scores: Dict[str, Dict[Any, Any]],
        labeled_elements: Dict[Any, str],
    ) -> None:
        if not page_data.elements:
            return

        page_num_height: Optional[float] = None
        # Find the page_number element to use for size comparison
        for element in page_data.elements:
            if labeled_elements.get(element) == "page_number":
                page_num_height = element.bbox.height
                break

        # Get page bbox and height for bottom band check
        page_bbox = page_data.bbox
        assert page_bbox is not None
        page_height = page_bbox.height

        # Initialize scores dict for this classifier
        if "step_number" not in scores:
            scores["step_number"] = {}

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

            # Store detailed score object
            detail_score = _StepNumberScore(
                text_score=text_score,
                size_score=size_score,
            )

            scores["step_number"][element] = detail_score

    def classify(
        self,
        page_data: PageData,
        scores: Dict[str, Dict[Any, Any]],
        labeled_elements: Dict[Any, str],
        removal_reasons: Dict[int, RemovalReason],
        hints: Optional["ClassificationHints"] = None,
        constructed_elements: Optional[Dict["Element", "LegoPageElement"]] = None,
        candidates: Optional[Dict[str, List["Candidate"]]] = None,
    ) -> None:
        if candidates is None:
            candidates = {}

        # Find the page number element to avoid classifying it as a step number
        page_number_element = None
        for element in page_data.elements:
            if labeled_elements.get(element) == "page_number":
                page_number_element = element
                break

        # Get pre-calculated scores for this classifier
        step_number_scores = scores.get("step_number", {})
        candidate_list: "List[Candidate]" = []

        for element in page_data.elements:
            if not isinstance(element, Text):
                continue

            # Skip if this element is already labeled as the page number
            if element is page_number_element:
                continue

            # Get the score object and compute combined score
            score_obj = step_number_scores.get(element)
            if not isinstance(score_obj, _StepNumberScore):
                continue

            combined_score = score_obj.combined_score(self.config)
            if combined_score < self.config.min_confidence_threshold:
                continue

            # Try to construct (parse step number value)
            value = extract_step_number_value(element.text)
            constructed_elem = None
            failure_reason = None

            if value is not None:
                constructed_elem = StepNumber(
                    value=value,
                    bbox=element.bbox,
                    id=element.id,
                )
            else:
                failure_reason = (
                    f"Could not parse step number from text: '{element.text}'"
                )

            # Create candidate
            candidate = Candidate(
                source_element=element,
                label="step_number",
                score=combined_score,
                score_details=score_obj,
                constructed=constructed_elem,
                failure_reason=failure_reason,
                is_winner=(value is not None),  # Winner if parsing succeeded
            )
            candidate_list.append(candidate)

            # If it's a winner, mark it in labeled_elements for backward compatibility
            if candidate.is_winner:
                labeled_elements[element] = "step_number"
                if constructed_elements is not None and constructed_elem is not None:
                    constructed_elements[element] = constructed_elem
                self.classifier._remove_child_bboxes(
                    page_data, element, removal_reasons
                )
                self.classifier._remove_similar_bboxes(
                    page_data, element, removal_reasons
                )

        # Store all candidates
        candidates["step_number"] = candidate_list

    def _score_step_number_text(self, text: str) -> float:
        """Score text based on how well it matches step number patterns.

        Returns:
            1.0 if text matches step number pattern, 0.0 otherwise
        """
        # Use the extraction function to validate format
        if extract_step_number_value(text) is not None:
            return 1.0
        return 0.0
