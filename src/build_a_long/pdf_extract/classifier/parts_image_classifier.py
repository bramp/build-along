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
from build_a_long.pdf_extract.extractor.page_blocks import (
    Drawing,
    Image,
    Text,
)

log = logging.getLogger(__name__)


@dataclass
class _PartImageScore:
    """Internal score representation for part image pairing."""

    distance: float
    """Vertical distance from part count text to image (lower is better)."""

    part_count: Text
    """The part count text element."""

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

    def evaluate(
        self,
        result: ClassificationResult,
    ) -> None:
        """Evaluate elements and create scores for part image pairings.

        Scores are based on vertical distance and horizontal alignment between
        part count texts and images within parts lists.
        """
        page_data = result.page_data

        # Get source blocks for part_counts and parts_lists
        part_count_candidates = [
            c
            for c in result.get_candidates("part_count")
            if c.constructed is not None
            and c.source_blocks
            and isinstance(c.source_blocks[0], Text)
        ]
        parts_list_candidates = [
            c
            for c in result.get_candidates("parts_list")
            if c.constructed is not None
            and c.source_blocks
            and isinstance(c.source_blocks[0], Drawing)
        ]

        if not part_count_candidates or not parts_list_candidates:
            return

        # Extract source blocks (Text and Drawing) for matching
        part_counts: list[Text] = [c.source_blocks[0] for c in part_count_candidates]  # type: ignore
        parts_lists: list[Drawing] = [c.source_blocks[0] for c in parts_list_candidates]  # type: ignore
        if not part_counts or not parts_lists:
            return

        images = self._get_images_in_parts_lists(page_data, parts_lists)
        if not images:
            return

        # Build candidate pairings and match them directly
        candidate_edges = self._build_candidate_edges(
            part_counts, images, page_data.bbox.width if page_data.bbox else 100.0
        )
        self._match_and_label_parts(candidate_edges, part_counts, images, result)

    def _match_and_label_parts(
        self,
        edges: list[_PartImageScore],
        part_counts: list[Text],
        images: list[Image],
        result: ClassificationResult,
    ):
        """Match part counts with images using greedy matching based on distance."""
        edges.sort(key=lambda score: score.sort_key())
        matched_counts: set[int] = set()
        matched_images: set[int] = set()

        for score in edges:
            pc = score.part_count
            img = score.image
            if id(pc) in matched_counts or id(img) in matched_images:
                continue
            matched_counts.add(id(pc))
            matched_images.add(id(img))
            # Create a candidate for the matched image
            # containing the pair relationship
            result.add_candidate(
                "part_image",
                Candidate(
                    bbox=img.bbox,
                    label="part_image",
                    score=1.0,  # Matched based on distance, not a traditional score
                    score_details=score,
                    constructed=None,
                    source_blocks=[img],
                    failure_reason=None,
                ),
            )

        if log.isEnabledFor(logging.DEBUG):
            unmatched_c = [pc for pc in part_counts if id(pc) not in matched_counts]
            unmatched_i = [im for im in images if id(im) not in matched_images]
            if unmatched_c:
                log.debug("[part_image] unmatched part_counts: %d", len(unmatched_c))
            if unmatched_i:
                log.debug("[part_image] unmatched images: %d", len(unmatched_i))

    def _build_candidate_edges(
        self,
        part_counts: list[Text],
        images: list[Image],
        page_width: float,
    ) -> list[_PartImageScore]:
        """Build candidate pairings between part counts and images.

        Returns a list of score objects representing valid pairings.
        """
        VERT_EPS = 2.0  # allow minor overlap/touching
        ALIGN_EPS = max(2.0, 0.02 * page_width)

        edges: list[_PartImageScore] = []
        for pc in part_counts:
            cb = pc.bbox
            for img in images:
                ib = img.bbox
                if ib.y1 <= cb.y0 + VERT_EPS and abs(ib.x0 - cb.x0) <= ALIGN_EPS:
                    distance = max(0.0, cb.y0 - ib.y1)
                    score = _PartImageScore(
                        distance=distance,
                        part_count=pc,
                        image=img,
                    )
                    edges.append(score)
        return edges

    def _get_images_in_parts_lists(
        self, page_data: PageData, parts_lists: list[Drawing]
    ) -> list[Image]:
        def inside_any_parts_list(img: Image) -> bool:
            return any(img.bbox.fully_inside(pl.bbox) for pl in parts_lists)

        return [
            e
            for e in page_data.blocks
            if isinstance(e, Image) and inside_any_parts_list(e)
        ]
