"""
Parts classifier.

Purpose
-------
Associate each part_count text with exactly one image to create Part candidates.
These Part candidates will be consumed by PartsListClassifier to build PartsList
elements.

Heuristic
---------
- For each part_count Text, find candidate Images that are above
  (image.y1 <= count.y0 + VERT_EPS) and roughly left-aligned
  (|image.x0 - count.x0| <= ALIGN_EPS).
- Sort candidates by vertical distance (count.y0 - image.y1), then
  greedily match to enforce one-to-one pairing between part counts and
  images.
- Create Part candidates with the paired PartCount and Drawing (image).

Debugging
---------
Enable with `LOG_LEVEL=DEBUG` for structured logs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from build_a_long.pdf_extract.classifier.classification_result import (
    Candidate,
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Part,
    PartCount,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Image,
)

log = logging.getLogger(__name__)


@dataclass
class _PartPairScore:
    """Internal score representation for part pairing."""

    distance: float
    """Vertical distance from part count text to image (lower is better)."""

    part_count: PartCount
    """The constructed PartCount element."""

    image: Image
    """The image element (will become diagram in Part)."""

    def sort_key(self) -> float:
        """Return sort key for matching (prefer smaller distance)."""
        return self.distance


# TODO Should this be called PartClassifier instead?
@dataclass(frozen=True)
class PartsClassifier(LabelClassifier):
    """Classifier for Part elements (pairs of part_count + image)."""

    outputs = frozenset({"part"})
    requires = frozenset({"part_count"})

    def evaluate(
        self,
        result: ClassificationResult,
    ) -> None:
        """Evaluate elements and create scores for part pairings.

        Scores are based on vertical distance and horizontal alignment between
        part count elements and images.
        """
        page_data = result.page_data

        # Get part_count candidates with type safety, selecting by score
        part_counts = result.get_winners_by_score("part_count", PartCount)

        if not part_counts:
            return

        # Get all images on the page
        images: list[Image] = [e for e in page_data.blocks if isinstance(e, Image)]

        if not images:
            return

        # Build candidate pairings and match them directly
        candidate_edges = self._build_candidate_edges(
            part_counts, images, page_data.bbox.width if page_data.bbox else 100.0
        )

        # Match and create Part candidates
        self._match_and_create_parts(candidate_edges, result)

    def classify(self, result: ClassificationResult) -> None:
        """No-op: All work done in evaluate()."""
        pass

    def _build_candidate_edges(
        self,
        part_counts: list[PartCount],
        images: list[Image],
        page_width: float,
    ) -> list[_PartPairScore]:
        """Build candidate pairings between part counts and images.

        Returns a list of score objects representing valid pairings.
        """
        VERT_EPS = 2.0  # allow minor overlap/touching
        ALIGN_EPS = max(2.0, 0.02 * page_width)

        edges: list[_PartPairScore] = []
        for pc in part_counts:
            cb = pc.bbox
            for img in images:
                ib = img.bbox
                # Image should be above the count and left-aligned
                if ib.y1 <= cb.y0 + VERT_EPS and abs(ib.x0 - cb.x0) <= ALIGN_EPS:
                    distance = max(0.0, cb.y0 - ib.y1)
                    score = _PartPairScore(
                        distance=distance,
                        part_count=pc,
                        image=img,
                    )
                    edges.append(score)
        return edges

    def _match_and_create_parts(
        self, candidate_edges: list[_PartPairScore], result: ClassificationResult
    ) -> None:
        """Match part counts with images and create Part candidates.

        Args:
            candidate_edges: List of candidate pairings to consider
            result: Classification result to add Part candidates to
        """
        if not candidate_edges:
            return

        # Sort by distance (closest pairs first)
        candidate_edges.sort(key=lambda score: score.sort_key())

        matched_counts: set[int] = set()
        matched_images: set[int] = set()

        for score in candidate_edges:
            pc = score.part_count
            img = score.image

            # Skip if already matched
            if id(pc) in matched_counts or id(img) in matched_images:
                continue

            matched_counts.add(id(pc))
            matched_images.add(id(img))

            # Create a Part from this pairing
            # The bbox is the union of the part_count and image bboxes
            combined_bbox = BBox(
                x0=min(pc.bbox.x0, img.bbox.x0),
                y0=min(pc.bbox.y0, img.bbox.y0),
                x1=max(pc.bbox.x1, img.bbox.x1),
                y1=max(pc.bbox.y1, img.bbox.y1),
            )

            part = Part(
                bbox=combined_bbox,
                count=pc,
                # Note: diagram field is optional and not set here
                # The image is tracked via the score_details
            )

            # Create a candidate for this Part
            result.add_candidate(
                "part",
                Candidate(
                    bbox=combined_bbox,
                    label="part",
                    score=1.0,  # Matched based on distance
                    score_details=score,
                    constructed=part,
                    source_block=None,  # Synthetic element, no single source
                    failure_reason=None,
                ),
            )
