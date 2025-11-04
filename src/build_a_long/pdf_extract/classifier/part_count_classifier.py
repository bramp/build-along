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
from typing import Optional

from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.text_extractors import (
    extract_part_count_value,
)
from build_a_long.pdf_extract.classifier.types import (
    Candidate,
    ClassificationHints,
    ClassificationResult,
    ClassifierConfig,
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

    def evaluate(
        self,
        page_data: PageData,
        result: ClassificationResult,
    ) -> None:
        """Evaluate elements and create candidates for part counts.

        This method scores each text element, attempts to construct PartCount objects,
        and stores all candidates with their scores and any failure reasons.
        """
        if not page_data.elements:
            return

        for element in page_data.elements:
            if not isinstance(element, Text):
                continue

            text_score = PartCountClassifier._score_part_count_text(element.text)

            # Store detailed score object
            detail_score = _PartCountScore(text_score=text_score)

            if self._debug_enabled:
                log.debug(
                    "[part_count] match text=%r score=%.2f bbox=%s",
                    element.text,
                    text_score,
                    element.bbox,
                )

            # Try to construct (parse part count value)
            value = extract_part_count_value(element.text)
            constructed_elem = None
            failure_reason = None

            if text_score == 0.0:
                failure_reason = (
                    f"Text doesn't match part count pattern: '{element.text}'"
                )
            elif value is None:
                failure_reason = (
                    f"Could not parse part count from text: '{element.text}'"
                )
            else:
                constructed_elem = PartCount(
                    count=value,
                    bbox=element.bbox,
                    id=element.id,
                )

            # Add candidate
            result.add_candidate(
                "part_count",
                Candidate(
                    source_element=element,
                    label="part_count",
                    score=detail_score.combined_score(self.config),
                    score_details=detail_score,
                    constructed=constructed_elem,
                    failure_reason=failure_reason,
                    is_winner=False,  # Will be set by classify()
                ),
            )

    def classify(
        self,
        page_data: PageData,
        result: ClassificationResult,
        hints: Optional[ClassificationHints],
    ) -> None:
        """Select winning part counts from pre-built candidates."""
        # Get pre-built candidates
        candidate_list = result.candidates.get("part_count", [])

        # Mark winners (all successfully constructed candidates)
        for candidate in candidate_list:
            if candidate.constructed is None:
                # Already has failure_reason from calculate_scores
                continue

            # This is a winner!
            assert isinstance(candidate.constructed, PartCount)
            result.mark_winner(
                candidate, candidate.source_element, candidate.constructed
            )
            self.classifier._remove_child_bboxes(
                page_data, candidate.source_element, result
            )
            self.classifier._remove_similar_bboxes(
                page_data, candidate.source_element, result
            )

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
