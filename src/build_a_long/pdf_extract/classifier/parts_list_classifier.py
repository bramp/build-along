"""
Parts list classifier.

Purpose
-------
Identify the drawing region(s) that represent the page's parts list.
We look for drawings that contain one or more Part elements.

Scoring is based solely on whether the drawing contains parts:
- 1.0 if the drawing contains parts
- 0.0 if the drawing has no parts

PartsListClassifier does NOT consider StepNumber proximity or alignment - that
scoring is done by StepClassifier when it pairs PartsLists with StepNumbers.

Debugging
---------
Set environment variables to aid investigation without code changes:

- LOG_LEVEL=DEBUG
    Enables DEBUG-level logging (if not already configured by caller).
"""

import logging
from collections.abc import Sequence
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
    Part,
    PartsList,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Drawing,
)

log = logging.getLogger(__name__)


@dataclass
@dataclass
class _PartsListScore:
    """Internal score representation for parts list classification."""

    has_parts: bool
    """Whether this drawing contains any parts."""

    def combined_score(self) -> float:
        """Calculate final score (simply 1.0 if has parts, 0.0 otherwise)."""
        return 1.0 if self.has_parts else 0.0


class PartsListClassifier(LabelClassifier):
    """Classifier for parts lists."""

    outputs = {"parts_list"}
    requires = {"part"}

    def evaluate(
        self,
        page_data: PageData,
        result: ClassificationResult,
    ) -> None:
        """Evaluate elements and create candidates for potential parts list drawings.

        Scores drawings based solely on whether they contain Part elements.
        Does NOT consider StepNumber proximity - that's done by StepClassifier.
        """

        # Get part candidates and their constructed Part elements
        part_candidates = result.get_candidates("part")
        parts: list[Part] = []
        for candidate in part_candidates:
            if (
                candidate.is_winner
                and candidate.constructed is not None
                and isinstance(candidate.constructed, Part)
            ):
                parts.append(candidate.constructed)
        if not parts:
            return

        drawings: list[Drawing] = [
            e for e in page_data.blocks if isinstance(e, Drawing)
        ]
        if not drawings:
            return

        log.debug(
            "[parts_list] page=%s blocks=%d drawings=%d parts=%d",
            page_data.page_number,
            len(page_data.blocks),
            len(drawings),
            len(parts),
        )

        # Score each drawing based on whether it contains parts
        for drawing in drawings:
            # Find all parts contained in this drawing
            contained = self._score_containing_parts(drawing, parts)

            # Create score
            score = _PartsListScore(has_parts=len(contained) > 0)

            # Only create candidates for drawings that have parts
            if not score.has_parts:
                continue

            constructed = PartsList(
                bbox=drawing.bbox,
                parts=contained,
            )

            # Add candidate
            result.add_candidate(
                "parts_list",
                Candidate(
                    bbox=drawing.bbox,
                    label="parts_list",
                    score=score.combined_score(),
                    score_details=score,
                    constructed=constructed,
                    source_block=drawing,
                    failure_reason=None,
                    is_winner=False,  # Will be set by classify()
                ),
            )

    def _score_containing_parts(
        self, drawing: Drawing, parts: Sequence[Part]
    ) -> list[Part]:
        """Find all parts that are contained within a drawing.

        Args:
            drawing: The drawing element to check
            parts: List of all Part elements on the page

        Returns:
            List of Part elements whose bboxes are fully inside the drawing
        """
        contained = [part for part in parts if part.bbox.fully_inside(drawing.bbox)]

        return contained

    def classify(self, page_data: PageData, result: ClassificationResult) -> None:
        """Mark all parts list candidates with parts as winners.

        All PartsList candidates created in evaluate() have parts, so we mark
        them all as winners. StepClassifier will later decide which PartsList
        to pair with which StepNumber.
        """
        candidate_list = result.get_candidates("parts_list")

        for candidate in sorted(candidate_list, key=lambda c: c.score, reverse=True):
            if candidate.constructed is None:
                continue

            # Check if removed by overlap
            if candidate.source_block is not None and result.is_removed(
                candidate.source_block
            ):
                continue

            # Mark as winner
            result.mark_winner(candidate, candidate.constructed)

            # Remove similar/overlapping drawings
            if candidate.source_block is not None:
                self.classifier._remove_similar_bboxes(
                    page_data, candidate.source_block, result
                )
