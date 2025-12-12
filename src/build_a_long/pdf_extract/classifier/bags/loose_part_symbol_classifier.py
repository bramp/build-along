"""
Loose part symbol classifier.

Purpose
-------
Identify "loose part" symbols that appear next to OpenBag circles containing
a Part instead of a bag number. The symbol indicates that an extra part is
needed that's not found in the main bag. It is a small, square-ish cluster
of drawings that provides additional visual context for which part to find.

Heuristic
---------
1. Find clusters of drawings/images that are:
   - Located to the right of an OpenBag circle
   - Have a roughly square aspect ratio (0.7 to 1.4)
   - Contain multiple drawing elements (>=3)
2. Score based on proximity to OpenBag circles and aspect ratio

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
)
from build_a_long.pdf_extract.classifier.rules import (
    AspectRatioRule,
    CurveCountRule,
    IsInstanceFilter,
    Rule,
    SizeRangeRule,
)
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    LoosePartSymbol,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Drawing,
    Image,
)

log = logging.getLogger(__name__)


class _LoosePartSymbolScore(Score):
    """Score for loose part symbol classification."""

    aspect_score: float
    """Score based on how square the cluster is (0.0-1.0)."""

    size_score: float
    """Score based on how close the size is to the ideal (~68.5 points)."""

    def score(self) -> Weight:
        """Calculate final weighted score from components."""
        return (self.aspect_score + self.size_score) / 2.0


class LoosePartSymbolClassifier(RuleBasedClassifier):
    """Classifier for loose part symbols.

    Identifies symbol clusters that appear in the upper portion of pages.
    These are small, square-ish clusters of drawings that provide visual context.
    The OpenBagClassifier will claim matching symbols during its build phase.
    """

    output: ClassVar[str] = "loose_part_symbol"
    requires: ClassVar[frozenset[str]] = frozenset()  # No dependencies - runs early

    @property
    def rules(self) -> list[Rule]:
        """Rules to identify the anchor block (small circle)."""
        return [
            IsInstanceFilter(Drawing),
            # Max size 80 (individual block)
            SizeRangeRule(
                max_width=80.0,
                max_height=80.0,
                weight=0.0,
                required=True,
                name="max_size",
            ),
            # Min size 20
            SizeRangeRule(
                min_width=20.0,
                min_height=20.0,
                weight=0.0,
                required=True,
                name="min_size",
            ),
            # Aspect ratio 0.8 - 1.25 (circular)
            AspectRatioRule(
                min_ratio=0.8,
                max_ratio=1.25,
                weight=0.0,
                required=True,
                name="circular_aspect",
            ),
            # Curves >= 4 (circle)
            CurveCountRule(min_count=4, weight=1.0, required=True),
        ]

    def _score(self, result: ClassificationResult) -> None:
        """Find symbol clusters using anchor-and-cluster strategy.

        1. Use rules to find potential anchor blocks (small circles).
        2. For each anchor, find nearby blocks to form a cluster.
        3. Score the cluster and update the candidate.
        """
        # 1. Find anchors using base rules
        super()._score(result)

        # 2. Refine candidates (which currently represent just the anchors)
        # Get all candidates including failed ones, but we only care about valid ones here
        # actually super()._score only adds valid ones.
        candidates = result.get_scored_candidates(self.output, valid_only=False)

        # We need to iterate carefully as we might modify them
        for candidate in candidates:
            # Skip if already marked as failed by rules
            if not candidate.is_valid:
                continue

            anchor = candidate.source_blocks[0]
            if not isinstance(anchor, Drawing):
                # Should be guaranteed by IsInstanceFilter, but safe check
                candidate.failure_reason = "Anchor is not a Drawing"
                continue

            # Find blocks near this anchor (including images, which rules filtered out)
            # Re-fetch small blocks for clustering context
            # Optimization: filter only relevant blocks once?
            # Or just iterate all blocks? `_find_nearby_blocks` iterates.
            # Original code filtered `small_blocks` first.
            cluster_blocks = self._find_cluster_blocks(anchor, result.page_data.blocks)

            if len(cluster_blocks) < 3:
                candidate.failure_reason = (
                    f"Cluster too small (found {len(cluster_blocks)} blocks, need 3)"
                )
                continue

            # Calculate combined bbox
            symbol_bbox = BBox.union_all([b.bbox for b in cluster_blocks])

            # Check aspect ratio - should be roughly square (0.6 to 1.6)
            aspect = symbol_bbox.width / symbol_bbox.height if symbol_bbox.height else 0
            if not (0.6 <= aspect <= 1.6):
                candidate.failure_reason = f"Bad cluster aspect ratio: {aspect:.2f}"
                continue

            # Check total size
            config = self.config.loose_part_symbol
            ideal_size = config.ideal_size
            tolerance = config.size_tolerance
            min_size = ideal_size * (1.0 - tolerance)
            max_size = ideal_size * (1.0 + tolerance)
            avg_size = (symbol_bbox.width + symbol_bbox.height) / 2.0

            if not (min_size <= avg_size <= max_size):
                candidate.failure_reason = f"Bad cluster size: {avg_size:.1f} (range {min_size:.1f}-{max_size:.1f})"
                continue

            # Score based on aspect ratio (closer to 1.0 is better)
            aspect_score = 1.0 - abs(aspect - 1.0) * 0.5
            # Score based on size (closer to ideal_size is better)
            size_deviation = abs(avg_size - ideal_size) / ideal_size
            size_score = 1.0 - size_deviation

            new_score_details = _LoosePartSymbolScore(
                aspect_score=aspect_score,
                size_score=size_score,
            )

            # Update candidate
            candidate.bbox = symbol_bbox
            candidate.source_blocks = list(cluster_blocks)
            candidate.score_details = new_score_details
            candidate.score = new_score_details.score()

            log.debug(
                "[loose_part_symbol] Refined candidate: bbox=%s score=%.2f blocks=%d",
                symbol_bbox,
                candidate.score,
                len(cluster_blocks),
            )

    def _find_cluster_blocks(
        self,
        anchor: Drawing,
        all_blocks: list,
    ) -> list[Drawing | Image]:
        """Find blocks that are near the anchor block."""
        result: list[Drawing | Image] = [anchor]
        max_block_size = 80.0
        max_distance = 30.0

        for block in all_blocks:
            if block is anchor:
                continue

            # Only consider small Drawings or Images
            if not isinstance(block, Drawing | Image):
                continue

            if (
                block.bbox.width >= max_block_size
                or block.bbox.height >= max_block_size
            ):
                continue

            # Check distance from anchor
            # Use BBox.min_distance
            dist = anchor.bbox.min_distance(block.bbox)
            if dist <= max_distance:
                result.append(block)

        return result

    def build(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LoosePartSymbol:
        """Construct a LoosePartSymbol element from a candidate."""
        return LoosePartSymbol(bbox=candidate.bbox)  # type: ignore[return-value]
