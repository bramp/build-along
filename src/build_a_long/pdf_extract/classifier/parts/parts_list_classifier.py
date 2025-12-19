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

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.bbox import (
    BBox,
    filter_contained,
    group_by_similar_bbox,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Part,
    PartsList,
)
from build_a_long.pdf_extract.extractor.page_blocks import Drawing

log = logging.getLogger(__name__)


# TODO: Let's update the scoring to:
# 1) Higher score with more valid parts inside
# 2) Penalize overly large drawings (e.g. >50% of page area) OR ensure the Part
#    to size ratio is reasonable.
class _PartsListScore(Score):
    """Internal score representation for parts list classification."""

    part_candidates: list[Candidate] = []
    """List of part candidates inside this parts list."""

    def score(self) -> Weight:
        """Return the score."""
        # If we have a valid box and parts inside, it's a good candidate
        if self.part_candidates:
            return 1.0
        return 0.0


class PartsListClassifier(LabelClassifier):
    """Classifier for parts lists (lists of parts for a step)."""

    output = "parts_list"
    requires = frozenset({"part"})

    def _score(self, result: ClassificationResult) -> None:
        """Score drawings and create candidates for potential parts lists.

        Creates candidates with score details containing the Part candidates,
        but does not construct the PartsList yet.

        Drawings with similar bboxes are grouped into a single candidate
        (parts lists often have multiple overlapping Drawing blocks for
        border/fill). Overlap resolution between different parts lists
        happens at build time based on candidate scores.
        """
        # Get part candidates (not constructed elements)
        part_candidates = result.get_scored_candidates("part")
        if not part_candidates:
            return

        page_data = result.page_data
        drawings: list[Drawing] = [
            e for e in page_data.blocks if isinstance(e, Drawing)
        ]
        if not drawings:
            return

        log.debug(
            "[parts_list] page=%s blocks=%d drawings=%d part_candidates=%d",
            page_data.page_number,
            len(page_data.blocks),
            len(drawings),
            len(part_candidates),
        )

        # Group drawings with similar bboxes together
        # Each group represents a single potential parts list
        groups = group_by_similar_bbox(drawings, tolerance=2.0)

        log.debug(
            "[parts_list] Grouped %d drawings into %d unique bbox regions",
            len(drawings),
            len(groups),
        )

        # Create one candidate per group
        min_score = self.config.parts_list.min_score
        for group in groups:
            # Use union of all drawings' bboxes as the candidate bbox
            bbox = BBox.union_all([d.bbox for d in group])

            # Find all part candidates contained in this bbox
            contained = filter_contained(part_candidates, bbox)
            score = _PartsListScore(part_candidates=contained)

            # Skip candidates below min_score threshold
            if score.score() < min_score:
                continue

            # Determine failure reason if any
            failure_reason = None

            if len(contained) == 0:
                failure_reason = "Drawing contains no parts"

            # Check if drawing is suspiciously large
            if failure_reason is None and page_data.bbox:
                page_area = page_data.bbox.area
                drawing_area = bbox.area
                max_ratio = self.config.parts_list.max_area_ratio
                if page_area > 0 and drawing_area / page_area > max_ratio:
                    pct = drawing_area / page_area * 100
                    failure_reason = f"Drawing too large ({pct:.1f}% of page area)"
                    log.debug(
                        "[parts_list] Drawing %d rejected: %s",
                        group[0].id,
                        failure_reason,
                    )

            # Create candidate with all grouped drawings as source blocks
            candidate = Candidate(
                bbox=bbox,
                label="parts_list",
                score=score.score(),
                score_details=score,
                source_blocks=list(group),
                failure_reason=failure_reason,
            )

            result.add_candidate(candidate)

            if len(group) > 1:
                log.debug(
                    "[parts_list] Created candidate at %s from %d drawings "
                    "(score=%.2f, parts=%d)",
                    bbox,
                    len(group),
                    score.score(),
                    len(contained),
                )

    def build(self, candidate: Candidate, result: ClassificationResult) -> PartsList:
        """Construct a PartsList from a single candidate's score details.

        Validates and extracts Part elements from the parent candidates.
        """
        assert isinstance(candidate.score_details, _PartsListScore)
        score = candidate.score_details

        # Validate and extract Part elements from parent candidates
        parts: list[Part] = []
        for part_candidate in score.part_candidates:
            try:
                part_elem = result.build(part_candidate)
                assert isinstance(part_elem, Part)
                parts.append(part_elem)
            except Exception as e:
                log.warning(
                    "Failed to construct part candidate at %s: %s",
                    part_candidate.bbox,
                    e,
                )

        return PartsList(
            bbox=candidate.bbox,
            parts=parts,
        )
