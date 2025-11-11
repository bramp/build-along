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
    requires = frozenset({"page_number", "progress_bar", "step"})

    def evaluate(
        self,
        result: ClassificationResult,
    ) -> None:
        """Evaluate elements and create a Page candidate.

        Collects page_number and step elements to build a complete Page.
        """
        page_data = result.page_data

        # Get winners with type safety
        page_number_winners = result.get_winners("page_number", PageNumber)
        page_number = page_number_winners[0] if page_number_winners else None

        progress_bar_winners = result.get_winners("progress_bar", ProgressBar)
        progress_bar = progress_bar_winners[0] if progress_bar_winners else None

        steps = result.get_winners("step", Step)

        # Sort steps by their step_number value
        steps.sort(key=lambda step: step.step_number.value)

        log.debug(
            "[page] page=%s page_number=%s progress_bar=%s steps=%d",
            page_data.page_number,
            page_number.value if page_number else None,
            progress_bar is not None,
            len(steps),
        )

        # Construct the Page
        constructed = Page(
            bbox=page_data.bbox,
            page_number=page_number,
            progress_bar=progress_bar,
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
                is_winner=False,  # Will be set by classify()
            ),
        )

    def classify(self, result: ClassificationResult) -> None:
        """Mark the Page candidate as winner."""
        candidate_list = result.get_candidates("page")

        for candidate in candidate_list:
            if candidate.constructed is None:
                continue

            assert isinstance(candidate.constructed, Page)

            # This is a winner!
            result.mark_winner(candidate, candidate.constructed)
