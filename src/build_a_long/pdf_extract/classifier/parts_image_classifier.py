"""
Part image classifier.

Purpose
-------
Associate each part_count text with exactly one image inside the chosen parts list,
using a heuristic: choose the image that is just above and left-aligned with the part
count. Enforce a one-to-one mapping between part counts and images.

Heuristic
---------
- Consider only Image elements fully inside any labeled parts_list Drawing.
- For each part_count Text inside the same parts_list, create candidate edges to
  Images that are above (image.y1 <= count.y0 + VERT_EPS) and roughly left-aligned
  (|image.x0 - count.x0| <= ALIGN_EPS).
- Sort candidates by vertical distance (count.y0 - image.y1), then greedily match
  to enforce one-to-one pairing.

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
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.lego_page_elements import LegoPageElements
from build_a_long.pdf_extract.extractor.page_blocks import (
    Image,
)

log = logging.getLogger(__name__)


@dataclass
class _PartImageScore:
    """Internal score representation for part image pairing."""

    distance: float
    """Vertical distance from part count text to image (lower is better)."""

    part_count_candidate: Candidate
    """The part count candidate."""

    image: Image
    """The image element."""

    def sort_key(self) -> float:
        """Return sort key for matching (prefer smaller distance)."""
        return self.distance


# TODO Should this be called PartImageClassifier instead?
@dataclass(frozen=True)
class PartsImageClassifier(LabelClassifier):
    """Classifier for part images paired with part count texts."""

    outputs = frozenset({"part_image"})
    requires = frozenset({"parts_list", "part_count"})

    def score(self, result: ClassificationResult) -> None:
        """Score part image pairings and create candidates.

        Creates candidates with score details containing the part count candidate
        and image, but does not construct any element (this is metadata only).
        """
        page_data = result.page_data

        # Get candidates that have been successfully constructed
        part_count_candidates = [
            c for c in result.get_candidates("part_count") if c.constructed is not None
        ]
        parts_list_candidates = [
            c for c in result.get_candidates("parts_list") if c.constructed is not None
        ]

        if not part_count_candidates or not parts_list_candidates:
            return

        images = self._get_images_in_parts_lists(page_data, parts_list_candidates)
        if not images:
            return

        # Build candidate pairings
        candidate_edges = self._build_candidate_edges(
            part_count_candidates,
            images,
            page_data.bbox.width if page_data.bbox else 100.0,
        )

        # Sort and match
        candidate_edges.sort(key=lambda score: score.sort_key())
        matched_count_candidates: set[int] = set()
        matched_images: set[int] = set()

        for score in candidate_edges:
            pc_candidate = score.part_count_candidate
            img = score.image
            if (
                id(pc_candidate) in matched_count_candidates
                or id(img) in matched_images
            ):
                continue
            matched_count_candidates.add(id(pc_candidate))
            matched_images.add(id(img))

            # Create candidate WITHOUT construction (part_image is metadata only)
            result.add_candidate(
                "part_image",
                Candidate(
                    bbox=img.bbox,
                    label="part_image",
                    score=1.0,
                    score_details=score,
                    constructed=None,
                    source_blocks=[img],
                    failure_reason=None,
                ),
            )

        if log.isEnabledFor(logging.DEBUG):
            unmatched_c = [
                pc
                for pc in result.get_candidates("part_count")
                if pc.constructed is not None and id(pc) not in matched_count_candidates
            ]
            unmatched_i = [im for im in images if id(im) not in matched_images]
            if unmatched_c:
                log.debug("[part_image] unmatched part_counts: %d", len(unmatched_c))
            if unmatched_i:
                log.debug("[part_image] unmatched images: %d", len(unmatched_i))

    def construct(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        """Part images don't produce constructed elements.

        This classifier only creates metadata about pairings, not actual elements.
        Return None to indicate no construction needed.
        """
        # PartsImageClassifier doesn't construct elements, it just tracks pairings
        # Return None is not valid for LegoPageElements, so we need to handle this
        # differently. Actually, looking at the code, part_image candidates are
        # never constructed - they're just metadata.
        raise NotImplementedError(
            "PartsImageClassifier does not construct elements - "
            "it only creates metadata candidates"
        )

    def evaluate(
        self,
        result: ClassificationResult,
    ) -> None:
        """Evaluate elements and create scores for part image pairings.

        Scores are based on vertical distance and horizontal alignment between
        part count texts and images within parts lists.
        """
        # Just call score - no construction needed for this classifier
        self.score(result)

    def _build_candidate_edges(
        self,
        part_count_candidates: list[Candidate],
        images: list[Image],
        page_width: float,
    ) -> list[_PartImageScore]:
        """Build candidate pairings between part count candidates and images.

        Returns a list of score objects representing valid pairings.
        """
        VERT_EPS = 2.0  # allow minor overlap/touching
        ALIGN_EPS = max(2.0, 0.02 * page_width)

        edges: list[_PartImageScore] = []
        for pc_candidate in part_count_candidates:
            cb = pc_candidate.bbox
            for img in images:
                ib = img.bbox
                if ib.y1 <= cb.y0 + VERT_EPS and abs(ib.x0 - cb.x0) <= ALIGN_EPS:
                    distance = max(0.0, cb.y0 - ib.y1)
                    score = _PartImageScore(
                        distance=distance,
                        part_count_candidate=pc_candidate,
                        image=img,
                    )
                    edges.append(score)
        return edges

    def _get_images_in_parts_lists(
        self, page_data: PageData, parts_list_candidates: list[Candidate]
    ) -> list[Image]:
        """Get images that are inside any parts_list candidate's bounding box."""

        def inside_any_parts_list(img: Image) -> bool:
            return any(
                img.bbox.fully_inside(candidate.bbox)
                for candidate in parts_list_candidates
            )

        return [
            e
            for e in page_data.blocks
            if isinstance(e, Image) and inside_any_parts_list(e)
        ]
