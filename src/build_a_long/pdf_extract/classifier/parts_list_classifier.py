"""
Parts list classifier.

Purpose
-------
Identify the drawing region(s) that represent the page's parts list.
We look for drawings that contain one or more Part elements.
Among candidates, we prefer more parts and smaller area.

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
class _PartsListScore:
    """Internal score representation for parts list classification."""

    part_count: int
    """Number of Part elements contained in the bounding box."""

    area: float
    """Area of the bounding box."""

    def sort_key(self) -> tuple[int, float]:
        """Return a tuple for sorting candidates.

        We prefer:
        1. More parts (higher is better, so negate for sorting)
        2. Smaller area (lower is better)
        """
        return (-self.part_count, self.area)


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

        Scores drawings based on the number of Part elements they contain.
        Creates candidates for all viable parts list drawings.
        """

        # Get part candidates and their constructed Part elements
        part_candidates = result.get_candidates("part")
        # Use constructed Part elements instead of source elements
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

        # Score each drawing based on parts contained
        for drawing in drawings:
            # Find all parts contained in this drawing
            contained = self._score_containing_parts(drawing, parts)
            if not contained:
                # Drawing contains no parts, skip it
                continue

            # Create score object
            score = _PartsListScore(
                part_count=len(contained),
                area=drawing.bbox.area,
            )

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
                    score=1.0,  # Parts list uses ranking rather than scores
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
        # Get pre-built candidates
        candidate_list = result.get_candidates("parts_list")

        # Sort the candidates based on our scoring criteria
        sorted_candidates = sorted(
            candidate_list,
            key=lambda c: (c.score_details.sort_key()),
        )

        # Mark winners (all successfully constructed candidates)
        for candidate in sorted_candidates:
            if candidate.constructed is None:
                # Already has failure_reason from calculate_scores
                continue

            assert isinstance(candidate.constructed, PartsList)

            # Check if this candidate has been removed due to overlap with a
            # previous winner (skip synthetic candidates without source_block)
            if candidate.source_block is not None and result.is_removed(
                candidate.source_block
            ):
                continue

            # This is a winner!
            result.mark_winner(candidate, candidate.constructed)
            # Note: Do NOT remove child bboxes - Parts are contained within
            if candidate.source_block is not None:
                self.classifier._remove_similar_bboxes(
                    page_data, candidate.source_block, result
                )
