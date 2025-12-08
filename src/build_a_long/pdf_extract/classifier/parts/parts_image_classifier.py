"""
Part image classifier.

Purpose
-------
Creates PartImage candidates from Image blocks on the page.
These candidates are then paired with part counts by PartsClassifier.

This classifier scores images based on their size (part images are typically
small-medium sized, around 1/20 to 1/5 of page dimensions). Shine elements
are discovered and attached at build time.

Debugging
---------
Enable with `LOG_LEVEL=DEBUG` for structured logs.
"""

from __future__ import annotations

import logging

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
    Image,
)

log = logging.getLogger(__name__)


class _PartImageScore(Score):
    """Score details for a part image candidate.

    Scores based on intrinsic size properties - part images are typically
    small to medium sized (1/20 to 1/5 of page dimensions).

    Attributes:
        image: The source Image block being scored as a potential part image
        size_score: Score based on image dimensions relative to page (0.0-1.0)
    """

    image: Image
    size_score: float

    def score(self) -> Weight:
        """Return the overall score based on size."""
        return self.size_score


# TODO Should this be called PartImageClassifier instead?
class PartsImageClassifier(LabelClassifier):
    """Classifier for part images based on size heuristics.

    Scores images based on their size relative to the page. Part images are
    typically small to medium sized (around 1/20 to 1/5 of page dimensions).

    Shine elements are discovered and attached at build time by looking for
    overlapping shine candidates in the top-right area of the image.

    Does NOT pair images with part counts - that's done by PartsClassifier.
    """

    output = "part_image"
    requires = frozenset({"shine"})

    def _score(self, result: ClassificationResult) -> None:
        """Create PartImage candidates from Image blocks with size-based scoring.

        Scores images based on their dimensions relative to the page.
        Shine discovery is deferred to build time.
        """
        page_data = result.page_data
        page_width = page_data.bbox.width
        page_height = page_data.bbox.height

        # Get all images from the page
        images: list[Image] = [e for e in page_data.blocks if isinstance(e, Image)]

        if not images:
            log.debug(
                "[part_image] No images found on page %s",
                page_data.page_number,
            )
            return

        # Create a PartImage candidate for each Image with size-based score
        for img in images:
            size_score = self._compute_size_score(img, page_width, page_height)

            score_details = _PartImageScore(image=img, size_score=size_score)

            result.add_candidate(
                Candidate(
                    bbox=img.bbox,
                    label="part_image",
                    score=size_score,
                    score_details=score_details,
                    source_blocks=[img],
                ),
            )

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "[part_image] Created %d part_image candidates on page %s",
                len(images),
                page_data.page_number,
            )

    def _compute_size_score(
        self, img: Image, page_width: float, page_height: float
    ) -> float:
        """Compute a score based on image size relative to page.

        Part images are typically small to medium sized:
        - Width: ~1/20 to ~1/5 of page width (25-100 pixels on a 500px page)
        - Height: ~1/20 to ~1/5 of page height

        Very small images (icons) and very large images (diagrams) score lower.

        Args:
            img: The image to score
            page_width: Width of the page
            page_height: Height of the page

        Returns:
            Score from 0.0 to 1.0
        """
        img_width = img.bbox.width
        img_height = img.bbox.height

        # Calculate size ratios
        width_ratio = img_width / page_width if page_width > 0 else 0
        height_ratio = img_height / page_height if page_height > 0 else 0

        # Ideal range: 5% to 20% of page dimension
        # Score peaks at ~10% (typical part image size)
        def ratio_score(ratio: float) -> float:
            if ratio < 0.02:
                # Too small (icons, dots)
                return ratio / 0.02 * 0.3
            elif ratio < 0.05:
                # Small but acceptable
                return 0.3 + (ratio - 0.02) / 0.03 * 0.4
            elif ratio <= 0.20:
                # Ideal range
                return 0.7 + (1.0 - abs(ratio - 0.10) / 0.10) * 0.3
            elif ratio <= 0.40:
                # Large but could still be a part
                return 0.7 - (ratio - 0.20) / 0.20 * 0.4
            else:
                # Too large (likely a diagram)
                return max(0.1, 0.3 - (ratio - 0.40) / 0.60 * 0.3)

        width_score = ratio_score(width_ratio)
        height_score = ratio_score(height_ratio)

        # Average the two dimensions
        return (width_score + height_score) / 2

    def build(self, candidate: Candidate, result: ClassificationResult) -> PartImage:
        """Construct a PartImage element from a single part_image candidate.

        Discovers and attaches any shine element that overlaps with the image
        in the top-right corner area.

        Args:
            candidate: The part_image candidate to construct
            result: Classification result for context

        Returns:
            PartImage: The constructed part image element
        """
        assert isinstance(candidate.score_details, _PartImageScore)
        ps = candidate.score_details

        # Find and build shine at build time (not pre-assigned during scoring)
        shine = self._find_and_build_shine(ps.image, result)

        # Compute final bbox - expand to include shine if found
        bbox = candidate.bbox
        if shine:
            bbox = bbox.union(shine.bbox)

        return PartImage(bbox=bbox, shine=shine, image_id=ps.image.image_id)

    def _find_and_build_shine(
        self, image: Image, result: ClassificationResult
    ) -> Shine | None:
        """Find and build a shine element for this image.

        Looks for shine candidates that overlap with the image, preferring
        those closer to the top-right corner (where shines typically appear).

        Args:
            image: The image to find a shine for
            result: Classification result containing shine candidates

        Returns:
            Built Shine element, or None if no matching shine found
        """
        # Get available (not yet built) shine candidates
        shine_candidates = result.get_scored_candidates(
            "shine",
            valid_only=False,
            exclude_failed=True,
        )

        best_shine_candidate: Candidate | None = None
        best_dist = float("inf")

        for shine_cand in shine_candidates:
            # Skip if already built (consumed by another part image)
            if shine_cand.constructed is not None:
                continue

            # Shine must overlap with the image to be considered a match.
            # Shines that are merely "near" but outside the image are likely
            # other elements (e.g., decorations).
            if not shine_cand.bbox.overlaps(image.bbox):
                continue

            # Prefer shines closer to top-right corner of the image
            # Image TR corner: (x1, y0) where y0 is top in PDF coordinates
            tr_x = image.bbox.x1
            tr_y = image.bbox.y0

            shine_center = shine_cand.bbox.center

            # Distance from shine center to Image TR corner
            dx = shine_center[0] - tr_x
            dy = shine_center[1] - tr_y
            corner_dist = (dx * dx + dy * dy) ** 0.5

            if corner_dist < best_dist:
                best_dist = corner_dist
                best_shine_candidate = shine_cand

        if best_shine_candidate is None:
            return None

        # Build the shine
        try:
            shine_elem = result.build(best_shine_candidate)
            assert isinstance(shine_elem, Shine)
            log.debug(
                "[part_image] Found shine at %s for image at %s (dist=%.1f)",
                shine_elem.bbox,
                image.bbox,
                best_dist,
            )
            return shine_elem
        except Exception as e:
            log.debug(
                "[part_image] Failed to build shine at %s: %s",
                best_shine_candidate.bbox,
                e,
            )
            return None
