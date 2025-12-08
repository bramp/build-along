"""
Step number classifier.
"""

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
    InBottomBandFilter,
    IsInstanceFilter,
    Rule,
    StepNumberTextRule,
)
from build_a_long.pdf_extract.classifier.text import (
    extract_step_number_value,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    StepNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import Text


class StepNumberClassifier(RuleBasedClassifier):
    """Classifier for step numbers."""

    output = "step_number"
    requires = frozenset()

    @property
    def min_score(self) -> float:
        return self.config.step_number.min_score

    @property
    def rules(self) -> list[Rule]:
        config = self.config
        return [
            # Must be text
            IsInstanceFilter(Text),
            # Must NOT be in the bottom 10% of the page (page number area)
            InBottomBandFilter(
                threshold_ratio=0.1,
                name="not_in_bottom_band",
                invert=True,
            ),
            # Score based on text pattern
            StepNumberTextRule(
                weight=config.step_number.text_weight,
                name="text_score",
                required=True,  # Require it to look like a step number
            ),
            # Score based on font size hints
            FontSizeMatch(
                target_size=config.font_size_hints.step_number_size,
                weight=config.step_number.font_size_weight,
                name="font_size_score",
            ),
        ]

    def build(self, candidate: Candidate, result: ClassificationResult) -> StepNumber:
        """Construct a StepNumber element from a candidate.

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

        # Parse the step number value
        value = extract_step_number_value(block.text)
        if value is None:
            raise ValueError(f"Could not parse step number from text: '{block.text}'")

        # Successfully constructed
        return StepNumber(value=value, bbox=block.bbox)
