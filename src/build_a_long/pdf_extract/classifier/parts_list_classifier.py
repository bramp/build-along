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

    parts: int
    """Whether this drawing contains any parts."""

    def combined_score(self) -> float:
        """Calculate final score (simply 1.0 if has parts, 0.0 otherwise)."""
        return 1.0 if self.parts > 0 else 0.0


@dataclass(frozen=True)
class PartsListClassifier(LabelClassifier):
    """Classifier for parts lists."""

    outputs = frozenset({"parts_list"})
    requires = frozenset({"part"})

    def evaluate(
        self,
        result: ClassificationResult,
    ) -> None:
        """Evaluate elements and create candidates for potential parts list drawings.

        Scores drawings based solely on whether they contain Part elements.
        Does NOT consider StepNumber proximity - that's done by StepClassifier.
        """

        # Get part winners with type safety
        parts = result.get_winners_by_score("part", Part)
        if not parts:
            return

        page_data = result.page_data
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

        # Pre-score all drawings to sort them by quality
        drawing_scores: list[tuple[Drawing, _PartsListScore, list[Part]]] = []
        for drawing in drawings:
            # Find all parts contained in this drawing
            contained = self._score_containing_parts(drawing, parts)

            # Create score
            score = _PartsListScore(parts=len(contained))
            drawing_scores.append((drawing, score, contained))

        # Sort by score (highest first), then by drawing ID for determinism
        drawing_scores.sort(
            key=lambda x: (x[1].combined_score(), -x[0].id), reverse=True
        )

        # Track accepted candidates to check for overlaps
        IOU_THRESHOLD = 0.9  # Consider candidates duplicate if IOU > this
        accepted_candidates: list[Candidate] = []

        # Process each drawing in score order
        for drawing, score, contained in drawing_scores:
            # Determine failure reason if any
            failure_reason = None
            constructed = None

            if not score.parts > 0:
                failure_reason = "Drawing contains no parts"
            # Check if drawing is suspiciously large (likely the entire page)
            # A legitimate parts list should be a reasonable fraction of the page
            if failure_reason is None and page_data.bbox:
                page_area = page_data.bbox.area
                drawing_area = drawing.bbox.area
                max_ratio = self.config.parts_list_max_area_ratio
                if page_area > 0 and drawing_area / page_area > max_ratio:
                    pct = drawing_area / page_area * 100
                    failure_reason = f"Drawing too large ({pct:.1f}% of page area)"
                    log.debug(
                        "[parts_list] Drawing %d rejected: %s",
                        drawing.id,
                        failure_reason,
                    )
            # Check for overlap with already-accepted candidates
            if failure_reason is None and accepted_candidates:
                for accepted in accepted_candidates:
                    # TODO Later we could optomise this with a spatial index if needed
                    overlap = drawing.bbox.iou(accepted.bbox)
                    if overlap > IOU_THRESHOLD:
                        failure_reason = (
                            f"Overlaps with {accepted.bbox} (IOU={overlap:.2f})"
                        )
                        log.debug(
                            "[parts_list] Drawing %d rejected: %s",
                            drawing.id,
                            failure_reason,
                        )
                        break

            # Only construct if no failure
            if failure_reason is None:
                constructed = PartsList(
                    bbox=drawing.bbox,
                    parts=contained,
                )

            # Create candidate
            candidate = Candidate(
                bbox=drawing.bbox,
                label="parts_list",
                score=score.combined_score(),
                score_details=score,
                constructed=constructed,
                source_block=drawing,
                failure_reason=failure_reason,
            )

            # Track accepted candidates for overlap checking
            if constructed is not None:
                accepted_candidates.append(candidate)

            # Add candidate to result (even if it failed, for debugging)
            result.add_candidate("parts_list", candidate)

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
        return [part for part in parts if part.bbox.fully_inside(drawing.bbox)]
