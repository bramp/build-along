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
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Page,
    PageNumber,
    Step,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageClassifier(LabelClassifier):
    """Classifier for building the complete Page element."""

    outputs = frozenset({"page"})
    requires = frozenset({"page_number", "step"})

    def evaluate(
        self,
        result: ClassificationResult,
    ) -> None:
        """Evaluate elements and create a Page candidate.

        Collects page_number and step elements to build a complete Page.
        """
        page_data = result.page_data
        # Get page_number candidate
        page_number_candidates = result.get_candidates("page_number")
        page_number: PageNumber | None = None
        for candidate in page_number_candidates:
            if (
                candidate.is_winner
                and candidate.constructed is not None
                and isinstance(candidate.constructed, PageNumber)
            ):
                page_number = candidate.constructed
                break

        # Get step candidates
        step_candidates = result.get_candidates("step")
        steps: list[Step] = []
        for candidate in step_candidates:
            if (
                candidate.is_winner
                and candidate.constructed is not None
                and isinstance(candidate.constructed, Step)
            ):
                steps.append(candidate.constructed)

        # Sort steps by their step_number value
        steps.sort(key=lambda step: step.step_number.value)

        log.debug(
            "[page] page=%s page_number=%s steps=%d",
            page_data.page_number,
            page_number.value if page_number else None,
            len(steps),
        )

        # Construct the Page
        constructed = Page(
            bbox=page_data.bbox,
            page_number=page_number,
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
