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
    LegoPageElements,
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

    parts: list[Part]
    """The Part elements contained in this drawing."""

    def combined_score(self) -> float:
        """Calculate final score (simply 1.0 if has parts, 0.0 otherwise)."""
        return 1.0 if len(self.parts) > 0 else 0.0


@dataclass(frozen=True)
class PartsListClassifier(LabelClassifier):
    """Classifier for parts lists."""

    outputs = frozenset({"parts_list"})
    requires = frozenset({"part"})

    def score(self, result: ClassificationResult) -> None:
        """Score drawings and create candidates for potential parts lists.

        Creates candidates with score details containing the Part elements,
        but does not construct the PartsList yet.
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
        drawing_scores: list[tuple[Drawing, _PartsListScore]] = []
        for drawing in drawings:
            # Find all parts contained in this drawing
            contained = self._find_containing_parts(drawing, parts)

            # Create score with Part references
            score = _PartsListScore(parts=contained)
            drawing_scores.append((drawing, score))

        # Sort by score (highest first), then by drawing ID for determinism
        drawing_scores.sort(
            key=lambda x: (x[1].combined_score(), -x[0].id), reverse=True
        )

        # Track accepted candidates to check for overlaps
        IOU_THRESHOLD = 0.9
        accepted_candidates: list[Candidate] = []

        # Process each drawing in score order
        for drawing, score in drawing_scores:
            combined = score.combined_score()

            # Skip candidates below minimum score threshold
            if combined < self.config.parts_list_min_score:
                log.debug(
                    "[parts_list] Skipping low-score candidate: drawing=%d "
                    "score=%.3f (below threshold %.3f)",
                    drawing.id,
                    combined,
                    self.config.parts_list_min_score,
                )
                continue

            # Determine failure reason if any
            failure_reason = None

            if not len(score.parts) > 0:
                failure_reason = "Drawing contains no parts"

            # Check if drawing is suspiciously large
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

            # Create candidate WITHOUT construction
            candidate = Candidate(
                bbox=drawing.bbox,
                label="parts_list",
                score=score.combined_score(),
                score_details=score,
                constructed=None,
                source_blocks=[drawing],
                failure_reason=failure_reason,
            )

            # Track accepted candidates for overlap checking
            if failure_reason is None:
                accepted_candidates.append(candidate)

            # Add candidate to result
            result.add_candidate("parts_list", candidate)

    def construct(self, result: ClassificationResult) -> None:
        """Construct PartsList elements from candidates."""
        candidates = result.get_candidates("parts_list")
        for candidate in candidates:
            try:
                elem = self._construct_single(candidate, result)
                candidate.constructed = elem
            except Exception as e:
                candidate.failure_reason = str(e)

    def _construct_single(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        """Construct a PartsList from a single candidate's score details.

        Uses the Part elements stored in the score to build the PartsList.
        """
        assert isinstance(candidate.score_details, _PartsListScore)
        score = candidate.score_details

        return PartsList(
            bbox=candidate.bbox,
            parts=score.parts,
        )

    def _find_containing_parts(
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
