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
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    BagNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import Text

log = logging.getLogger(__name__)


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
        """Construct a BagNumber element from a single candidate."""
        # Get the source text block
        assert len(candidate.source_blocks) == 1
        block = candidate.source_blocks[0]
        assert isinstance(block, Text)

        # Get score details
        score_details = candidate.score_details
        assert isinstance(score_details, RuleScore)

        # Parse the bag number value
        value = extract_bag_number_value(block.text)
        if value is None:
            raise ValueError(f"Could not parse bag number from text: '{block.text}'")

        # Successfully constructed
        return BagNumber(value=value, bbox=block.bbox)
