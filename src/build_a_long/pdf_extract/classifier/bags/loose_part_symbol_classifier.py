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
from collections.abc import Sequence
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
    AspectRatioRule,
    CurveCountRule,
    IsInstanceFilter,
    Rule,
    SizeRangeRule,
)
from build_a_long.pdf_extract.classifier.score import Weight
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    LoosePartSymbol,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Blocks,
    Drawing,
    Image,
)

log = logging.getLogger(__name__)


class _LoosePartSymbolScore(RuleScore):
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
    The OpenBagClassifier will consume matching symbols during its build phase.

    Implementation Pattern: Anchor-and-Cluster
    -------------------------------------------
    This classifier uses a clustering pattern where:

    1. Rules find anchor blocks (small circular drawings)
    2. _get_additional_source_blocks() discovers nearby blocks to form complete clusters
    3. _create_score() validates the cluster properties (size, aspect) and scores it

    The anchor + nearby blocks form a SINGLE visual element (the symbol).
    This pattern is similar to RotationSymbolClassifier which clusters Drawing blocks.
    """

    output: ClassVar[str] = "loose_part_symbol"
    requires: ClassVar[frozenset[str]] = frozenset()

    @property
    def effects_margin(self) -> float | None:
        return None

    @property
    def rules(self) -> Sequence[Rule]:
        """Rules to identify the anchor block (small circle)."""
        return [
            IsInstanceFilter(Drawing),
            # Size 20-80 (individual block)
            SizeRangeRule(
                min_width=20.0,
                min_height=20.0,
                max_width=80.0,
                max_height=80.0,
                weight=0.0,
                required=True,
                name="size",
            ),
            # Aspect ratio 0.9 - 1.1 (circular)
            AspectRatioRule(
                min_ratio=0.9,
                max_ratio=1.1,
                weight=0.0,
                required=True,
                name="circular_aspect",
            ),
            # Curves >= 4 (circle)
            CurveCountRule(min_count=4, weight=1.0, required=True),
        ]

    def _get_additional_source_blocks(
        self, block: Blocks, result: ClassificationResult
    ) -> Sequence[Blocks]:
        """Find blocks that form a cluster with the anchor block."""
        if not isinstance(block, Drawing):
            return []

        # Find nearby blocks to form the symbol cluster
        cluster = self._find_cluster_blocks(block, result.page_data.blocks)
        # Remove the anchor itself (it's already the primary block)
        return [b for b in cluster if b is not block]

    def _create_score(
        self,
        components: dict[str, float],
        total_score: float,
        source_blocks: Sequence[Blocks],
    ) -> _LoosePartSymbolScore:
        """Validate and score the complete cluster.

        Rejects clusters that don't meet size/aspect requirements.
        """
        # Need at least 3 blocks to form a valid symbol cluster
        if len(source_blocks) < 3:
            return _LoosePartSymbolScore(
                components=components,
                total_score=0.0,
                aspect_score=0.0,
                size_score=0.0,
            )

        # Calculate combined bbox of the cluster
        symbol_bbox = BBox.union_all([b.bbox for b in source_blocks])

        # Check aspect ratio - should be roughly square (0.9 to 1.1)
        aspect = symbol_bbox.width / symbol_bbox.height if symbol_bbox.height else 0
        if not (0.9 <= aspect <= 1.1):
            return _LoosePartSymbolScore(
                components=components,
                total_score=0.0,
                aspect_score=0.0,
                size_score=0.0,
            )

        # Check total size
        config = self.config.loose_part_symbol
        ideal_size = config.ideal_size
        tolerance = config.size_tolerance
        min_size = ideal_size * (1.0 - tolerance)
        max_size = ideal_size * (1.0 + tolerance)
        avg_size = (symbol_bbox.width + symbol_bbox.height) / 2.0

        if not (min_size <= avg_size <= max_size):
            return _LoosePartSymbolScore(
                components=components,
                total_score=0.0,
                aspect_score=0.0,
                size_score=0.0,
            )

        # Score based on aspect ratio (closer to 1.0 is better)
        aspect_score = 1.0 - abs(aspect - 1.0) * 0.5
        # Score based on size (closer to ideal_size is better)
        size_deviation = abs(avg_size - ideal_size) / ideal_size
        size_score = 1.0 - size_deviation

        log.debug(
            "[loose_part_symbol] Cluster validated: bbox=%s score=%.2f blocks=%d",
            symbol_bbox,
            (aspect_score + size_score) / 2.0,
            len(source_blocks),
        )

        return _LoosePartSymbolScore(
            components=components,
            total_score=total_score,
            aspect_score=aspect_score,
            size_score=size_score,
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
