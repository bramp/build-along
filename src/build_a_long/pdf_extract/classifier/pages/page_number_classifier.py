"""
Page number classifier.
"""

import logging
from collections.abc import Sequence

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.rule_based_classifier import (
    RuleBasedClassifier,
)
from build_a_long.pdf_extract.classifier.rules import (
    CornerDistanceScore,
    FontSizeMatch,
    InBottomBandFilter,
    IsInstanceFilter,
    PageNumberTextRule,
    PageNumberValueMatch,
    Rule,
)
from build_a_long.pdf_extract.classifier.rules.scale import (
    LinearScale,
)
from build_a_long.pdf_extract.classifier.text import (
    extract_page_number_value,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    PageNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import Text

log = logging.getLogger(__name__)


class PageNumberClassifier(RuleBasedClassifier):
    """Classifier for page numbers."""

    output = "page_number"
    requires = frozenset()

    @property
    def effects_margin(self) -> float | None:
        return 2.0

    @property
    def min_score(self) -> float:
        return self.config.page_number.min_score

    @property
    def rules(self) -> Sequence[Rule]:
        config = self.config
        page_number_size = config.font_size_hints.page_number_size or 12.0
        return [
            # Must be text
            IsInstanceFilter(Text),
            # Must be in bottom 10% of page (hard filter)
            InBottomBandFilter(
                threshold_ratio=0.1,
                name="position_band",
            ),
            # Score based on text pattern (matches page number format)
            PageNumberTextRule(
                weight=config.page_number.text_weight, name="text_score"
            ),
            # Score based on proximity to bottom corners
            CornerDistanceScore(
                scale=config.page_number.position_scale,
                weight=config.page_number.position_weight,
                name="position_score",
            ),
            # Score based on matching the expected page number value
            # 1.0 at exact match, 0.0 at 10 pages away
            PageNumberValueMatch(
                scale=LinearScale({0: 1.0, 10: 0.0}),
                weight=config.page_number.page_value_weight,
                name="page_value_score",
            ),
            # Score based on font size hints (triangular: 0 at edges, 1.0 at center)
            FontSizeMatch(
                scale=LinearScale(
                    {
                        page_number_size * 0.5: 0.0,
                        page_number_size: 1.0,
                        page_number_size * 1.5: 0.0,
                    }
                ),
                weight=config.page_number.font_size_weight,
                name="font_size_score",
            ),
        ]

    def build(self, candidate: Candidate, result: ClassificationResult) -> PageNumber:
        """Construct a PageNumber element from a candidate.

        The candidate may include additional source blocks (e.g., text outline
        effects) beyond the primary Text block.
        """
        # Get the primary text block (first in source_blocks)
        assert len(candidate.source_blocks) >= 1
        block = candidate.source_blocks[0]
        assert isinstance(block, Text)

        # Parse the page number value
        value = extract_page_number_value(block.text)
        if value is None:
            raise ValueError(f"Could not parse page number from text: '{block.text}'")

        # Use candidate.bbox which is the union of all source blocks
        return PageNumber(value=value, bbox=candidate.bbox)
