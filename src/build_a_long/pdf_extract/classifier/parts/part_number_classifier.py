"""
Part number (element ID) classifier.

Purpose
-------
Detect LEGO part numbers (element IDs) - typically 6-7 digit numbers that appear
on catalog pages below part counts.

Debugging
---------
Enable DEBUG logs with LOG_LEVEL=DEBUG.
"""

import logging
from collections.abc import Sequence

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
    PartNumberTextRule,
    Rule,
)
from build_a_long.pdf_extract.classifier.text import extract_element_id
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    PartNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import Text

log = logging.getLogger(__name__)


class PartNumberClassifier(RuleBasedClassifier):
    """Classifier for LEGO part numbers (element IDs)."""

    output = "part_number"
    requires = frozenset()

    @property
    def min_score(self) -> float:
        return self.config.part_number.min_score

    @property
    def rules(self) -> Sequence[Rule]:
        config = self.config
        return [
            # Must be text
            IsInstanceFilter(Text),
            # Score based on text pattern and length distribution
            PartNumberTextRule(
                weight=config.part_count.text_weight,  # Reuse PartCount weights
                name="text_score",
                required=True,
            ),
            # Score based on font size hints
            FontSizeMatch(
                target_size=config.font_size_hints.catalog_element_id_size,
                weight=config.part_count.font_size_weight,  # Reuse PartCount weights
                name="font_size_score",
            ),
        ]

    def build(self, candidate: Candidate, result: ClassificationResult) -> PartNumber:
        """Construct a PartNumber element from a candidate.

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

        # Extract and validate element ID
        element_id = extract_element_id(block.text)
        if element_id is None:
            raise ValueError(f"Text doesn't match part number pattern: '{block.text}'")

        # Use candidate.bbox which is the union of all source blocks
        return PartNumber(element_id=element_id, bbox=candidate.bbox)
