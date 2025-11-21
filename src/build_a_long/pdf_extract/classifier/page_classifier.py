"""
Page classifier.

Purpose
-------
Build a complete Page element from classified components.
This classifier depends on page_number and step to construct the final Page.

Debugging
---------
Set environment variables to aid investigation without code changes:

- LOG_LEVEL=DEBUG
    Enables DEBUG-level logging (if not already configured by caller).
"""

import logging
from dataclasses import dataclass

from build_a_long.pdf_extract.classifier.classification_result import (
    Candidate,
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    NewBag,
    Page,
    PageNumber,
    ProgressBar,
    Step,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageClassifier(LabelClassifier):
    """Classifier for building the complete Page element."""

    outputs = frozenset({"page"})
    requires = frozenset({"page_number", "progress_bar", "new_bag", "step"})

    def evaluate(
        self,
        result: ClassificationResult,
    ) -> None:
        """Evaluate elements and create a Page candidate.

        Collects page_number, progress_bar, and step elements to build a
        complete Page. Uses get_winners_by_score() to select the best
        candidates based on scores.
        """
        page_data = result.page_data

        # Get best candidates using score-based selection
        # (max_count=1 for singleton elements)
        page_number_winners = result.get_winners_by_score(
            "page_number", PageNumber, max_count=1
        )
        page_number = page_number_winners[0] if page_number_winners else None

        progress_bar_winners = result.get_winners_by_score(
            "progress_bar", ProgressBar, max_count=1
        )
        progress_bar = progress_bar_winners[0] if progress_bar_winners else None

        # Get new bags using score-based selection
        new_bags = result.get_winners_by_score("new_bag", NewBag)

        # Get steps using score-based selection (StepClassifier now handles
        # deduplication in evaluate(), so all step candidates are valid)
        steps = result.get_winners_by_score("step", Step)

        # Sort steps by their step_number value
        steps.sort(key=lambda step: step.step_number.value)

        log.debug(
            "[page] page=%s page_number=%s progress_bar=%s new_bags=%d steps=%d",
            page_data.page_number,
            page_number.value if page_number else None,
            progress_bar is not None,
            len(new_bags),
            len(steps),
        )

        # Construct the Page
        constructed = Page(
            bbox=page_data.bbox,
            page_number=page_number,
            progress_bar=progress_bar,
            new_bags=new_bags,
            steps=steps,
            warnings=[],
            unprocessed_elements=[],
        )

        # Add candidate
        result.add_candidate(
            "page",
            Candidate(
                bbox=page_data.bbox,
                label="page",
                score=1.0,
                score_details=None,
                constructed=constructed,
                source_block=None,  # Synthetic element
                failure_reason=None,
            ),
        )

    def classify(self, result: ClassificationResult) -> None:
        """No-op - Page selection handled by result.page property.

        This is part of a refactoring to eliminate the is_winner flag and
        mark_winner() method. The Page candidate created in evaluate() is
        accessed via the ClassificationResult.page property which uses
        get_winners_by_score().
        """
        pass
