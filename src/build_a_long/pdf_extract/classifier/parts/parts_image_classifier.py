"""
Part image classifier.

Purpose
-------
Creates PartImage candidates from Image blocks on the page.
These candidates are then paired with part counts by PartsClassifier.

This classifier scores images based on their size (part images are typically
small-medium sized, around 1/20 to 1/5 of page dimensions). Shine elements
are spatially scored and attached at scoring time, creating separate candidates
with and without shine for the constraint solver to choose.

Shine Discovery
---------------
Shines appear in the top-right corner of part images, overlapping with the image.
For each image, we create:
- One candidate WITHOUT shine (base case)
- One candidate WITH each potential shine (scored by proximity to top-right corner)

The constraint solver ensures each shine is used at most once across all PartImages.

Debugging
---------
Enable with `LOG_LEVEL=DEBUG` for structured logs.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import ClassVar

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.constraint_model import ConstraintModel
from build_a_long.pdf_extract.classifier.rule_based_classifier import (
    RuleBasedClassifier,
    RuleScore,
)
from build_a_long.pdf_extract.classifier.rules import (
    IsInstanceFilter,
    Rule,
    RuleContext,
    SizeRatioRule,
)
from build_a_long.pdf_extract.classifier.rules.scale import LinearScale
from build_a_long.pdf_extract.classifier.rules.scoring import (
    score_exponential_decay,
)
from build_a_long.pdf_extract.classifier.score import Score
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    PartImage,
    Shine,
)
from build_a_long.pdf_extract.extractor.page_blocks import Image

log = logging.getLogger(__name__)


# =============================================================================
# Score Model
# =============================================================================


class PartImageScore(Score):
    """Score for a PartImage candidate.

    Tracks both the base image score and optional shine score.
    """

    # Base image scores (from RuleBasedClassifier rules)
    base_score: float
    """Score from size-based rules."""

    base_details: RuleScore
    """Detailed rule breakdown for base score."""

    # Optional shine association
    shine_candidate: Candidate | None = None
    """The shine candidate associated with this PartImage, or None."""

    shine_score: float = 0.0
    """Spatial score for shine proximity (0-1, higher = better position)."""

    shine_distance: float = 0.0
    """Distance from shine center to image top-right corner."""

    def score(self) -> float:
        """Combined score: base score + shine bonus."""
        # If shine present, add a small bonus based on shine position quality
        # The shine_score is already 0-1 based on exponential decay from TR corner
        if self.shine_candidate is not None:
            # Weight shine contribution - good shine position gives bonus
            return self.base_score + (self.shine_score * 0.1)
        return self.base_score

    @property
    def summary(self) -> str:
        if self.shine_candidate is not None:
            return f"base={self.base_score:.2f}, shine_score={self.shine_score:.2f}"
        return f"base={self.base_score:.2f}, no_shine"


# =============================================================================
# Classifier
# =============================================================================


class PartsImageClassifier(RuleBasedClassifier):
    """Classifier for part images based on size heuristics.

    Scores images based on their size relative to the page. Part images are
    typically small to medium sized (around 1/20 to 1/5 of page dimensions).

    Creates multiple candidates per image:
    - One without shine (base case)
    - One for each potential shine (scored by spatial proximity to top-right)

    The constraint solver ensures each shine is used at most once.
    """

    output: ClassVar[str] = "part_image"
    requires: ClassVar[frozenset[str]] = frozenset({"shine"})

    # Shine search parameters - based on analysis of golden data
    # Valid shines are typically 3-8 units from image TR corner
    SHINE_MAX_DISTANCE: ClassVar[float] = 10.0
    """Maximum distance from image TR corner to shine center."""

    @property
    def rules(self) -> Sequence[Rule]:
        """Rules for base image scoring."""
        return [
            # Only consider Image elements
            IsInstanceFilter(Image),
            # Score based on size relative to page dimensions
            SizeRatioRule(
                scale=LinearScale({0.015: 0.0, 0.10: 1.0, 0.40: 0.0}),
                weight=1.0,
                name="size_ratio",
                required=True,
            ),
        ]

    def _score(self, result: ClassificationResult) -> None:
        """Score images and create candidates with/without shine options.

        For each image that passes the base rules:
        1. Create a candidate WITHOUT shine (using base _score_block)
        2. Create candidates WITH each potential shine (scored by position)
        """
        context = RuleContext(result.page_data, self.config, result)

        # Get shine candidates for spatial matching
        shine_candidates = list(result.get_scored_candidates("shine"))

        for block in result.page_data.blocks:
            # Use parent's _score_block to get base candidate
            base_candidate = self._score_block(block, context, result)
            if base_candidate is None:
                continue

            # Get the base score details (RuleScore from parent)
            base_details = base_candidate.score_details
            assert isinstance(base_details, RuleScore)
            base_score = base_details.score()

            # Find potential shines for this image using the full candidate bbox
            # (which includes shadows/effects from _get_additional_source_blocks)
            shine_matches = self._find_shine_candidates(
                base_candidate.bbox, shine_candidates
            )

            # Create candidate WITHOUT shine
            no_shine_score = PartImageScore(
                base_score=base_score,
                base_details=base_details,
                shine_candidate=None,
            )
            no_shine_cand = Candidate(
                bbox=base_candidate.bbox,
                label=self.output,
                score=no_shine_score.score(),
                score_details=no_shine_score,
                source_blocks=base_candidate.source_blocks,
            )
            result.add_candidate(no_shine_cand)

            # Create additional candidates WITH shine
            for shine_cand, distance, shine_score in shine_matches:
                # Remove shine's source blocks from part_image's source blocks
                # The shine candidate owns its blocks; we just reference the shine
                shine_block_ids = {b.id for b in shine_cand.source_blocks}
                filtered_source_blocks = [
                    b
                    for b in base_candidate.source_blocks
                    if b.id not in shine_block_ids
                ]

                # If all blocks were removed, skip this shine
                if not filtered_source_blocks:
                    continue

                # Recalculate bbox from remaining source blocks + shine bbox
                source_bbox = BBox.union_all([b.bbox for b in filtered_source_blocks])
                combined_bbox = source_bbox.union(shine_cand.bbox)

                with_shine_score = PartImageScore(
                    base_score=base_score,
                    base_details=base_details,
                    shine_candidate=shine_cand,
                    shine_score=shine_score,
                    shine_distance=distance,
                )

                with_shine_cand = Candidate(
                    bbox=combined_bbox,
                    label=self.output,
                    score=with_shine_score.score(),
                    score_details=with_shine_score,
                    source_blocks=filtered_source_blocks,
                )

                result.add_candidate(with_shine_cand)
                log.debug(
                    "[part_image] Created shine option: image=%s, shine=%s, "
                    "distance=%.1f, shine_score=%.2f",
                    base_candidate.bbox,
                    shine_cand.bbox,
                    distance,
                    shine_score,
                )

    def _find_shine_candidates(
        self,
        candidate_bbox: BBox,
        shine_candidates: Sequence[Candidate],
    ) -> list[tuple[Candidate, float, float]]:
        """Find shine candidates that could belong to this part image.

        Shines appear in the top-right corner of part images, overlapping with
        the image. We score them by distance from the candidate TR corner to the
        shine center.

        Args:
            candidate_bbox: The part image candidate's bounding box (includes effects)
            shine_candidates: Available shine candidates

        Returns:
            List of (shine_candidate, distance, score) tuples, sorted by score desc
        """
        matches: list[tuple[Candidate, float, float]] = []

        # Top-right corner of candidate (x1, y0 in PDF coordinates where y0 is top)
        tr_x = candidate_bbox.x1
        tr_y = candidate_bbox.y0

        for shine_cand in shine_candidates:
            # Shine must overlap with the candidate
            if not shine_cand.bbox.overlaps(candidate_bbox):
                continue

            # Calculate distance from image TR corner to shine center
            shine_center = shine_cand.bbox.center
            dx = shine_center[0] - tr_x
            dy = shine_center[1] - tr_y
            distance = (dx * dx + dy * dy) ** 0.5

            # Skip if too far from TR corner
            if distance > self.SHINE_MAX_DISTANCE:
                continue

            # Score by exponential decay from TR corner
            # Closer to TR = higher score
            shine_score = score_exponential_decay(distance, scale=3.0)

            matches.append((shine_cand, distance, shine_score))

        # Sort by score descending (best matches first)
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches

    def add_constraints(
        self, result: ClassificationResult, model: ConstraintModel
    ) -> None:
        """Add constraints for part_image candidates.

        Two types of constraints:
        1. Each shine can be used by at most one part_image
        2. If a part_image uses a shine, the standalone shine candidate is excluded
        """
        # Group candidates by which shine they use
        shine_to_part_images: dict[int, list[Candidate]] = {}

        for cand in result.get_scored_candidates(self.output):
            if not isinstance(cand.score_details, PartImageScore):
                continue

            shine_cand = cand.score_details.shine_candidate
            if shine_cand is not None:
                if shine_cand.id not in shine_to_part_images:
                    shine_to_part_images[shine_cand.id] = []
                shine_to_part_images[shine_cand.id].append(cand)

        # For each shine used by part_images:
        # 1. Part_images that share a shine are mutually exclusive (at most one)
        # 2. If any part_image with this shine is selected, the standalone shine
        #    candidate cannot also be selected (they share source blocks)
        for shine_id, part_image_cands in shine_to_part_images.items():
            # Get the shine candidate for this ID
            shine_cand = part_image_cands[0].score_details
            assert isinstance(shine_cand, PartImageScore)
            standalone_shine = shine_cand.shine_candidate

            if standalone_shine is not None:
                # Each part_image with shine is mutually exclusive with the
                # standalone shine (they share the shine's source blocks)
                for pi_cand in part_image_cands:
                    model.at_most_one_of([pi_cand, standalone_shine])

            # Part_images that share a shine are mutually exclusive
            if len(part_image_cands) > 1:
                model.at_most_one_of(part_image_cands)
                log.debug(
                    "[part_image] Shine %d shared by %d part_images",
                    shine_id,
                    len(part_image_cands),
                )

    def build(self, candidate: Candidate, result: ClassificationResult) -> PartImage:
        """Construct a PartImage element from a candidate.

        The shine is already determined at scoring time and stored in the
        score_details. We just need to build it.

        Args:
            candidate: The part_image candidate to construct
            result: Classification result for context

        Returns:
            PartImage: The constructed part image element
        """
        # Get the image block
        image_block = next(b for b in candidate.source_blocks if isinstance(b, Image))
        assert isinstance(image_block, Image)

        # Get shine from score details
        shine: Shine | None = None
        if isinstance(candidate.score_details, PartImageScore):
            shine_cand = candidate.score_details.shine_candidate
            if shine_cand is not None:
                # Build the shine element
                try:
                    shine_elem = result.build(shine_cand)
                    assert isinstance(shine_elem, Shine)
                    shine = shine_elem
                    log.debug(
                        "[part_image] Built shine at %s for image at %s",
                        shine.bbox,
                        image_block.bbox,
                    )
                except Exception as e:
                    log.debug(
                        "[part_image] Failed to build shine at %s: %s",
                        shine_cand.bbox,
                        e,
                    )

        return PartImage(
            bbox=candidate.bbox,
            shine=shine,
            image_id=image_block.image_id,
            digest=image_block.digest,
            xref=image_block.xref,
        )
