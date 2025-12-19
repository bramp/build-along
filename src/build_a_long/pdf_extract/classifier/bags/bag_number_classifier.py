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
    requires = frozenset()

    @property
    def min_score(self) -> float:
        return self.config.bag_number.min_score

    @property
    def rules(self) -> list[Rule]:
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
                weight=config.bag_number.position_weight,
                name="position_score",
                required=True,
            ),
            # Score based on font size (large text)
            BagNumberFontSizeRule(
                weight=config.bag_number.font_size_weight,
                name="font_size_score",
                required=True,
            ),
        ]

    def build(self, candidate: Candidate, result: ClassificationResult) -> BagNumber:
        """Construct a BagNumber element from a candidate.

        The candidate may include additional source blocks (e.g., text outline
        effects) beyond the primary Text block.

        Also claims nearby Images/Drawings that are likely drop shadows or
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

        # Find and claim nearby shadow/effect images by adding them to source_blocks
        shadow_blocks = self._find_shadow_blocks(block, result)
        for shadow_block in shadow_blocks:
            if shadow_block not in candidate.source_blocks:
                candidate.source_blocks.append(shadow_block)
                log.debug(
                    "[bag_number] Claimed shadow/effect block: %s",
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

        These are blocks that overlap significantly with the text bbox or
        are contained within a slightly expanded version of it.

        Args:
            text_block: The primary text block.
            result: Classification result for accessing page data.

        Returns:
            List of Image/Drawing blocks that should be claimed.
        """
        page_data = result.page_data
        text_bbox = text_block.bbox
        expanded_bbox = text_bbox.expand(_SHADOW_MARGIN)

        shadow_blocks: list[Image | Drawing] = []

        for block in page_data.blocks:
            if not isinstance(block, Image | Drawing):
                continue

            # Skip the text block itself
            if block is text_block:
                continue

            # Check if block overlaps significantly with text or is contained
            # in the expanded bbox
            if expanded_bbox.contains(block.bbox) or text_bbox.iou(block.bbox) > 0.3:
                shadow_blocks.append(block)

        return shadow_blocks
