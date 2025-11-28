"""
Part image classifier.

Purpose
-------
Creates PartImage candidates from Image blocks on the page.
These candidates are then paired with part counts by PartsClassifier.

This classifier simply wraps each Image as a potential part image candidate,
without filtering or dependencies on other classifiers.

Debugging
---------
Enable with `LOG_LEVEL=DEBUG` for structured logs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    PartImage,
    Shine,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Blocks,
    Image,
)

log = logging.getLogger(__name__)


class _PartImageScore(Score):
    """Score details for a part image candidate.

    Attributes:
        image: The source Image block being scored as a potential part image
        shine_candidate: Optional shine candidate associated with this image
    """

    image: Image
    shine_candidate: Candidate | None = None

    def score(self) -> Weight:
        """Return the score value (always 1.0 for part images)."""
        return 1.0


# TODO Should this be called PartImageClassifier instead?
@dataclass(frozen=True)
class PartsImageClassifier(LabelClassifier):
    """Classifier for part images based on size heuristics.

    Filters images to find those that could be part diagrams based on their
    size relative to the page. Typically part images are around 1/10 of the
    page width/height.

    Does NOT pair images with part counts - that's done by PartsClassifier.
    """

    output = "part_image"
    requires = frozenset({"shine"})

    def _score(self, result: ClassificationResult) -> None:
        """Create PartImage candidates from all Image blocks.

        Simply wraps each Image block as a PartImage candidate.
        PartsClassifier will handle pairing with part counts.
        """
        page_data = result.page_data

        # Get all images from the page
        images: list[Image] = [e for e in page_data.blocks if isinstance(e, Image)]

        if not images:
            log.debug(
                "[part_image] No images found on page %s",
                page_data.page_number,
            )
            return

        # Get shine candidates
        shine_candidates = result.get_scored_candidates(
            "shine",
            valid_only=False,
            exclude_failed=True,
        )

        # Create a PartImage candidate for each Image
        for img in images:
            # Find best matching shine
            shine_cand = self._find_matching_shine(img, shine_candidates)

            score_details = _PartImageScore(image=img, shine_candidate=shine_cand)

            # If we found a shine, include it in source blocks and expand bbox
            source_blocks: list[Blocks] = [img]
            bbox = img.bbox
            if shine_cand:
                source_blocks.extend(shine_cand.source_blocks)
                bbox = bbox.union(shine_cand.bbox)

            result.add_candidate(
                Candidate(
                    bbox=bbox,
                    label="part_image",
                    score=1.0,
                    score_details=score_details,
                    source_blocks=source_blocks,
                ),
            )

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "[part_image] Created %d part_image candidates on page %s",
                len(images),
                page_data.page_number,
            )

    def build(self, candidate: Candidate, result: ClassificationResult) -> PartImage:
        """Construct a PartImage element from a single part_image candidate.

        Args:
            candidate: The part_image candidate to construct
            result: Classification result for context

        Returns:
            PartImage: The constructed part image element
        """
        assert isinstance(candidate.score_details, _PartImageScore)
        ps = candidate.score_details

        shine: Shine | None = None
        if ps.shine_candidate:
            try:
                shine_elem = result.build(ps.shine_candidate)
                assert isinstance(shine_elem, Shine)
                shine = shine_elem
            except Exception as e:
                log.warning(
                    "Failed to construct optional shine at %s: %s",
                    ps.shine_candidate.bbox,
                    e,
                )

        # Simply create a PartImage with the candidate's bbox
        # Note: candidate.bbox might be larger than ps.image.bbox if it includes shine
        return PartImage(bbox=candidate.bbox, shine=shine)

    def _find_matching_shine(
        self, image: Image, shine_candidates: list[Candidate]
    ) -> Candidate | None:
        """Find a shine candidate that matches this image.

        Shines are small stars typically in the top-right corner of the image.
        """
        best_shine = None
        best_dist = float("inf")
        MAX_DIST = 10.0  # Maximum distance to consider match

        for shine in shine_candidates:
            # Check if shine is near the image
            # Use intersection or proximity
            if not shine.bbox.overlaps(image.bbox):
                # Also check proximity if not overlapping
                dist = shine.bbox.min_distance(image.bbox)
                if dist > MAX_DIST:
                    continue
            else:
                dist = 0.0

            # Prefer shines closer to top-right corner
            # Image TR corner
            tr_x = image.bbox.x1
            tr_y = image.bbox.y0  # y0 is top in this coordinate system?
            # Wait, coordinate system. usually y0 is top?
            # In BBox implementation: y0 is min_y (top in PDF if y starts at bottom? or top-left origin?)
            # The extractor uses PDF coordinates where (0,0) is usually bottom-left.
            # But let's check the data.
            # Page bbox: 0,0 to 552, 496.
            # Text at y=471 is page number at bottom. So y=0 is likely top.
            # Or y=0 is bottom?
            # Text "15" (page num) at y0=471.89.
            # Text "2x" (top of page) at y0=47.82.
            # So y=0 is TOP.
            # Therefore y0 is top edge (smaller value).

            # Top-Right corner of image: (x1, y0)
            shine_center_x = (shine.bbox.x0 + shine.bbox.x1) / 2
            shine_center_y = (shine.bbox.y0 + shine.bbox.y1) / 2

            # Distance from shine center to Image TR corner
            dx = shine_center_x - tr_x
            dy = shine_center_y - tr_y
            corner_dist = (dx * dx + dy * dy) ** 0.5

            if corner_dist < best_dist:
                best_dist = corner_dist
                best_shine = shine

        # Heuristic: Shine should be reasonably close to Top Right
        # We already filtered by general proximity/overlap
        return best_shine
