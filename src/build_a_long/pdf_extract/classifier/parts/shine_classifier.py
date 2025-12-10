"""
Shine classifier.

Purpose
-------
Identify small star-like drawings that indicate shiny/metallic parts.
These appear in the top-right area of part images.
"""

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
    IsInstanceFilter,
    Rule,
    SizeRangeRule,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import Shine
from build_a_long.pdf_extract.extractor.page_blocks import Drawing


class ShineClassifier(RuleBasedClassifier):
    """Classifier for 'shine' or 'sparkle' effects."""

    output: ClassVar[str] = "shine"
    requires: ClassVar[frozenset[str]] = frozenset()

    @property
    def rules(self) -> list[Rule]:
        return [
            IsInstanceFilter(Drawing),
            SizeRangeRule(
                min_width=5.0,
                max_width=15.0,
                min_height=5.0,
                max_height=15.0,
                weight=0.3,
                required=True,
                name="Size",
            ),
            AspectRatioRule(
                min_ratio=0.7,
                max_ratio=1.4,
                weight=0.7,
                required=True,
                name="AspectRatio",
            ),
        ]

    def build(self, candidate: Candidate, result: ClassificationResult) -> Shine:
        """Construct a Shine element from a single candidate."""
        return Shine(bbox=candidate.bbox)
