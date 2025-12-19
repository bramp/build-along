"""
Step count classifier.

Purpose
-------
Detect step-count text like "2x" that appears in substep callout boxes.
These are similar to part counts but use a larger font size (typically 16pt),
between part count size and step number size.

Debugging
---------
Enable DEBUG logs with LOG_LEVEL=DEBUG.
"""

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
    FontSizeRangeRule,
    IsInstanceFilter,
    PartCountTextRule,
    Rule,
)
from build_a_long.pdf_extract.classifier.text import (
    extract_part_count_value,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    StepCount,
)
from build_a_long.pdf_extract.extractor.page_blocks import Text

log = logging.getLogger(__name__)


class StepCountClassifier(RuleBasedClassifier):
    """Classifier for step counts (substep counts like "2x").

    These are count labels that appear inside substep callout boxes,
    indicating how many times to build the sub-assembly.
    They use a font size between part counts and step numbers.
    """

    output: ClassVar[str] = "step_count"
    requires: ClassVar[frozenset[str]] = frozenset()

    @property
    def min_score(self) -> float:
        return self.config.step_count.min_score

    @property
    def rules(self) -> list[Rule]:
        config = self.config
        step_count_config = config.step_count
        hints = config.font_size_hints

        return [
            # Must be text
            IsInstanceFilter(Text),
            # Check if text matches count pattern (e.g., "2x", "4x")
            PartCountTextRule(
                weight=step_count_config.text_weight,
                name="text_score",
                required=True,
            ),
            # Score font size: should be >= part_count_size and <= step_number_size
            FontSizeRangeRule(
                min_size=hints.part_count_size,
                max_size=hints.step_number_size,
                tolerance=1.0,
                weight=step_count_config.font_size_weight,
                name="font_size_score",
            ),
        ]

    def build(self, candidate: Candidate, result: ClassificationResult) -> StepCount:
        """Construct a StepCount element from a candidate.

        The candidate may include additional source blocks (e.g., text outline
        effects) beyond the primary Text block.
        """
        # Get the primary text block (first in source_blocks)
        assert len(candidate.source_blocks) >= 1
        block = candidate.source_blocks[0]
        assert isinstance(block, Text)

        # Parse the count value
        value = extract_part_count_value(block.text)
        if value is None:
            raise ValueError(f"Could not parse step count from text: '{block.text}'")

        # Use candidate.bbox which is the union of all source blocks
        return StepCount(count=value, bbox=candidate.bbox)
