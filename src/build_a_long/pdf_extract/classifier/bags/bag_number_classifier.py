"""
Bag number classifier.

Purpose
-------
Identify bag numbers in LEGO instructions. These are typically large text
numbers (1, 2, 3, etc.) that appear in the top-left area of a page,
surrounded by a cluster of images forming a "New Bag" visual element.

Heuristic
---------
- Look for Text elements containing single digits or small integers
- Typically larger font size than step numbers
- Located in the top portion of the page (upper 40%)
- Often left-aligned or centered within the left portion of the page
- Usually surrounded by multiple Image/Drawing blocks forming a bag icon

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
    BagNumberFontSizeRule,
    BagNumberTextRule,
    IsInstanceFilter,
    Rule,
    TopLeftPositionScore,
)
from build_a_long.pdf_extract.classifier.rules.scale import (
    DiscreteScale,
    LinearScale,
)
from build_a_long.pdf_extract.classifier.text import (
    extract_bag_number_value,
)
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    BagNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image, Text

log = logging.getLogger(__name__)

# Margin to expand text bbox when looking for shadow/effect images
_SHADOW_MARGIN = 10.0


class BagNumberClassifier(RuleBasedClassifier):
    """Classifier for bag numbers."""

    output = "bag_number"
    requires: ClassVar[frozenset[str]] = frozenset()

    @property
    def effects_margin(self) -> float | None:
        return None

    @property
    def rules(self) -> Sequence[Rule]:
        config = self.config

        return [
            # Must be text
            IsInstanceFilter(Text),
            # Score based on text pattern (digits 1-99)
            BagNumberTextRule(
                weight=config.bag_number.text_weight,
                name="text_score",
                required=True,
            ),
            # Score based on position (top-left preferred)
            TopLeftPositionScore(
                # 1.0 at top (0%), decays to 0.0 at 40% down
                vertical_scale=LinearScale({0.0: 1.0, 0.4: 0.0}),
                # 1.0 for left 50%, 0.3 for right 50%
                # TODO Why is this better than linear?
                horizontal_scale=DiscreteScale(
                    {
                        (0.0, 0.5): 1.0,  # Left 50%: full score
                        (0.5, 1.0): 0.3,  # Right 50%: reduced score
                    }
                ),
                weight=config.bag_number.position_weight,
                name="position_score",
                required=True,
            ),
            # Score based on font size (large text)
            BagNumberFontSizeRule(
                # 1.0 at 60pt, decays to 0.5 at 120pt
                scale=LinearScale({60.0: 1.0, 120.0: 0.5}),
                weight=config.bag_number.font_size_weight,
                name="font_size_score",
                required=True,
            ),
        ]

    def build(self, candidate: Candidate, result: ClassificationResult) -> BagNumber:
        """Construct a BagNumber element from a candidate.

        The candidate may include additional source blocks (e.g., text outline
        effects) beyond the primary Text block.

        Also consumes nearby Images/Drawings that are likely drop shadows or
        other text effects (within a small margin of the text bbox).
        """
        # Get the primary text block (first in source_blocks)
        assert len(candidate.source_blocks) >= 1
        block = candidate.source_blocks[0]
        assert isinstance(block, Text)

        # Get score details
        score_details = candidate.score_details
        assert isinstance(score_details, RuleScore)

        # Parse the bag number value
        value = extract_bag_number_value(block.text)
        if value is None:
            raise ValueError(f"Could not parse bag number from text: '{block.text}'")

        # Find and consume nearby shadow/effect images by adding them to source_blocks
        shadow_blocks = self._find_shadow_blocks(block, result)
        for shadow_block in shadow_blocks:
            if shadow_block not in candidate.source_blocks:
                candidate.source_blocks.append(shadow_block)
                log.debug(
                    "[bag_number] Consumed shadow/effect block: %s",
                    shadow_block.bbox,
                )

        # Compute bbox as union of all source blocks
        bbox = BBox.union_all([b.bbox for b in candidate.source_blocks])

        # Successfully constructed
        return BagNumber(value=value, bbox=bbox)

    def _find_shadow_blocks(
        self, text_block: Text, result: ClassificationResult
    ) -> list[Image | Drawing]:
        """Find Images/Drawings that are likely drop shadows or text effects.

        These are blocks that are contained within a slightly expanded version
        of the text bbox and are similar in size to the text. This helps avoid
        consuming unrelated blocks like bag icon images that happen to overlap.

        Only considers unconsumed blocks to avoid conflicts with other
        classifiers that have already consumed blocks.

        Args:
            text_block: The primary text block.
            result: Classification result for accessing page data.

        Returns:
            List of Image/Drawing blocks that should be consumed.
        """
        text_bbox = text_block.bbox
        expanded_bbox = text_bbox.expand(_SHADOW_MARGIN)

        shadow_blocks: list[Image | Drawing] = []

        # Only consider unconsumed Image/Drawing blocks
        for block in result.get_unconsumed_blocks((Image, Drawing)):
            # Skip the text block itself (shouldn't happen since we filter by type)
            if block is text_block:
                continue

            # Block must be contained in the expanded bbox
            if not expanded_bbox.contains(block.bbox):
                continue

            # Block must be similar in size to the text (within 2x)
            # Shadows shouldn't be much larger than the text they shadow
            size_ratio = block.bbox.area / text_bbox.area if text_bbox.area > 0 else 0
            if size_ratio > 2.0:
                continue

            # Type assertion: we filtered by (Image, Drawing) above
            assert isinstance(block, (Image, Drawing))
            shadow_blocks.append(block)

        return shadow_blocks
