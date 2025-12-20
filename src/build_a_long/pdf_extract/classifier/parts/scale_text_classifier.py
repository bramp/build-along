"""
Scale text classifier.

Purpose
-------
Identify the "1:1" text label associated with scale indicators.
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
    IsInstanceFilter,
    Rule,
)
from build_a_long.pdf_extract.classifier.rules.base import RuleContext
from build_a_long.pdf_extract.classifier.text import is_scale_text
from build_a_long.pdf_extract.extractor.lego_page_elements import ScaleText
from build_a_long.pdf_extract.extractor.page_blocks import Block, Text

log = logging.getLogger(__name__)


class ScaleTextMatchRule(Rule):
    """Score text that matches the 1:1 scale pattern."""

    def __init__(
        self,
        weight: float = 1.0,
        name: str = "ScaleTextMatch",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required

    def calculate(self, block: Block, context: RuleContext) -> float:
        if not isinstance(block, Text):
            return 0.0
        if is_scale_text(block.text):
            return 1.0
        return 0.0


class ScaleTextClassifier(RuleBasedClassifier):
    """Classifier for 1:1 scale text labels."""

    output: ClassVar[str] = "scale_text"
    requires: ClassVar[frozenset[str]] = frozenset()

    @property
    def rules(self) -> list[Rule]:
        return [
            IsInstanceFilter(Text),
            ScaleTextMatchRule(
                weight=1.0,
                required=True,
                name="Value",
            ),
        ]

    def build(self, candidate: Candidate, result: ClassificationResult) -> ScaleText:
        """Construct a ScaleText element from a candidate."""
        text_block = next(b for b in candidate.source_blocks if isinstance(b, Text))
        assert isinstance(text_block, Text)

        return ScaleText(
            bbox=candidate.bbox,
            text=text_block.text,
        )
