"""
Part count classifier.

Purpose
-------
Detect part-count text like "2x", "3X", or "5×".

Debugging
---------
Enable DEBUG logs with LOG_LEVEL=DEBUG.
"""

import logging
from collections.abc import Sequence
from typing import Literal

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.rule_based_classifier import (
    RuleBasedClassifier,
    RuleScore,
)
from build_a_long.pdf_extract.classifier.rules import (
    FontSizeMatch,
    IsInstanceFilter,
    MaxScoreRule,
    PartCountTextRule,
    Rule,
)
from build_a_long.pdf_extract.classifier.rules.scale import LinearScale
from build_a_long.pdf_extract.classifier.text import (
    extract_part_count_value,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    PartCount,
)
from build_a_long.pdf_extract.extractor.page_blocks import Text

log = logging.getLogger(__name__)


class PartCountClassifier(RuleBasedClassifier):
    """Classifier for part counts."""

    output = "part_count"
    requires = frozenset()

    @property
    def effects_margin(self) -> float | None:
        return 2.0

    @property
    def min_score(self) -> float:
        return self.config.part_count.min_score

    @property
    def rules(self) -> Sequence[Rule]:
        config = self.config
        part_count_size = config.font_size_hints.part_count_size or 10.0
        catalog_part_count_size = config.font_size_hints.catalog_part_count_size or 10.0
        return [
            # Must be text
            IsInstanceFilter(Text),
            # Score based on text pattern (matches "2x" etc)
            PartCountTextRule(
                weight=config.part_count.text_weight,
                name="text_score",
                required=True,
            ),
            # Score based on matching EITHER instruction or catalog font size
            # Takes the maximum score of the two
            MaxScoreRule(
                rules=[
                    FontSizeMatch(
                        scale=LinearScale(
                            {
                                part_count_size * 0.5: 0.0,
                                part_count_size: 1.0,
                                part_count_size * 1.5: 0.0,
                            }
                        ),
                        name="instruction_font_size",
                        weight=1.0,  # Weight handled by MaxScoreRule
                    ),
                    FontSizeMatch(
                        scale=LinearScale(
                            {
                                catalog_part_count_size * 0.5: 0.0,
                                catalog_part_count_size: 1.0,
                                catalog_part_count_size * 1.5: 0.0,
                            }
                        ),
                        name="catalog_font_size",
                        weight=1.0,  # Weight handled by MaxScoreRule
                    ),
                ],
                weight=config.part_count.font_size_weight,
                name="font_size_score",
            ),
        ]

    def build(self, candidate: Candidate, result: ClassificationResult) -> PartCount:
        """Construct a PartCount element from a candidate.

        The candidate may include additional source blocks (e.g., text outline
        effects) beyond the primary Text block.
        """
        # Get the primary text block (first in source_blocks)
        assert len(candidate.source_blocks) >= 1
        block = candidate.source_blocks[0]
        assert isinstance(block, Text)

        # Get score details
        score_details = candidate.score_details
        assert isinstance(score_details, RuleScore)

        # Parse the part count value
        value = extract_part_count_value(block.text)
        if value is None:
            raise ValueError(f"Could not parse part count from text: '{block.text}'")

        # Determine matched hint
        # This logic was previously in _score, but can be re-evaluated here
        matched_hint: Literal["part_count", "catalog_part_count"] | None = None

        # Re-calculate font matches to know which one won
        # Note: This duplicates the calculation in MaxScoreRule but avoids
        # complex plumbing to get metadata out of the rule engine.
        # Given it's just a float comparison, it's cheap.

        instruction_size = self.config.font_size_hints.part_count_size
        catalog_size = self.config.font_size_hints.catalog_part_count_size

        instruction_score = 0.0
        if instruction_size is not None and block.font_size is not None:
            # Use same logic as FontSizeMatch
            diff = abs(block.font_size - instruction_size) / instruction_size
            instruction_score = max(0.0, 1.0 - (diff * 2.0))

        catalog_score = 0.0
        if catalog_size is not None and block.font_size is not None:
            diff = abs(block.font_size - catalog_size) / catalog_size
            catalog_score = max(0.0, 1.0 - (diff * 2.0))

        # Only set matched_hint if we had a decent match
        if max(instruction_score, catalog_score) > 0:
            if instruction_score > catalog_score:
                matched_hint = "part_count"
            else:
                matched_hint = "catalog_part_count"

        # Use candidate.bbox which is the union of all source blocks
        return PartCount(count=value, bbox=candidate.bbox, matched_hint=matched_hint)
