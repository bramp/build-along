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
    extract_part_count_value,
)
from build_a_long.pdf_extract.classifier.types import (
    Candidate,
    ClassifierConfig,
    RemovalReason,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.lego_page_elements import PartCount
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
        # Can the following go into the parent, and use "outputs" as identifier?
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
        # TODO Do we need to do this? or should scores already have it?
        if "part_count" not in scores:
            scores["part_count"] = {}

        for element in page_data.elements:
            if not isinstance(element, Text):
                continue

            text_score = PartCountClassifier._score_part_count_text(element.text)

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
        hints: Optional["ClassificationHints"],
        constructed_elements: Dict["Element", "LegoPageElement"],
        candidates: Dict[str, List["Candidate"]],
    ) -> None:
        # Get pre-calculated scores for this classifier
        part_count_scores = scores.get("part_count", {})
        candidate_list: "List[Candidate]" = []

        for element in page_data.elements:
            # TODO Support non-text elements - such as images of text.
            if not isinstance(element, Text):
                continue

            # Get the score object and compute combined score
            score = part_count_scores.get(element)
            if not isinstance(score, _PartCountScore):
                continue

            # Try to construct (parse part count value)
            value = extract_part_count_value(element.text)
            constructed_elem = None
            failure_reason = None

            if value is not None:
                constructed_elem = PartCount(
                    count=value,
                    bbox=element.bbox,
                    id=element.id,
                )
            else:
                failure_reason = (
                    f"Could not parse part count from text: '{element.text}'"
                )

            # Create candidate
            candidate = Candidate(
                source_element=element,
                label="part_count",
                score=score.combined_score(self.config),
                score_details=score,
                constructed=constructed_elem,
                failure_reason=failure_reason,
                is_winner=(value is not None),  # Winner if parsing succeeded
            )
            candidate_list.append(candidate)

            # If it's a winner, mark it in labeled_elements for backward compatibility
            if candidate.is_winner:
                labeled_elements[element] = "part_count"
                if constructed_elements is not None and constructed_elem is not None:
                    constructed_elements[element] = constructed_elem
                self.classifier._remove_child_bboxes(
                    page_data, element, removal_reasons
                )
                self.classifier._remove_similar_bboxes(
                    page_data, element, removal_reasons
                )

        # Store all candidates
        candidates["part_count"] = candidate_list

    @staticmethod
    def _score_part_count_text(text: str) -> float:
        """Score text based on how well it matches part count patterns.

        Returns:
            1.0 if text matches part count pattern, 0.0 otherwise
        """
        # Use the extraction function to validate format
        if extract_part_count_value(text) is not None:
            return 1.0
        return 0.0
