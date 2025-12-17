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
from typing import ClassVar

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.rule_based_classifier import (
    RuleBasedClassifier,
    RuleScore,
)
from build_a_long.pdf_extract.classifier.rules import (
    IsInstanceFilter,
    Rule,
    SizeRatioRule,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    PartImage,
    Shine,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Image,
)

log = logging.getLogger(__name__)


class PartsImageClassifier(RuleBasedClassifier):
    """Classifier for part images based on size heuristics.

    Scores images based on their size relative to the page. Part images are
    typically small to medium sized (around 1/20 to 1/5 of page dimensions).

    Shine elements are discovered and attached at build time by looking for
    overlapping shine candidates in the top-right area of the image.

    Does NOT pair images with part counts - that's done by PartsClassifier.
    """

    output: ClassVar[str] = "part_image"
    requires: ClassVar[frozenset[str]] = frozenset({"shine"})

    @property
    def rules(self) -> list[Rule]:
        return [
            # Only consider Image elements
            IsInstanceFilter(Image),
            # Score based on size relative to page dimensions
            # Replicates the logic from _compute_size_score
            SizeRatioRule(
                ideal_ratio=0.10,  # Peak score at 10%
                min_ratio=0.015,  # Start scoring at 1.5% (captures tiny parts like single studs)
                max_ratio=0.40,  # End scoring at 4%
                weight=1.0,
                name="size_ratio",
                required=True,
            ),
        ]

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
        # Get score details (not strictly needed for logic but good for assertion)
        assert isinstance(candidate.score_details, RuleScore)

        # Get the image block
        image_block = next(b for b in candidate.source_blocks if isinstance(b, Image))
        assert isinstance(image_block, Image)

        # Find and build shine at build time (not pre-assigned during scoring)
        shine = self._find_and_build_shine(image_block, result)

        # Compute final bbox - expand to include shine if found
        bbox = candidate.bbox
        if shine:
            bbox = bbox.union(shine.bbox)

        return PartImage(
            bbox=bbox,
            shine=shine,
            image_id=image_block.image_id,
            digest=image_block.digest,
            xref=image_block.xref,
        )

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
