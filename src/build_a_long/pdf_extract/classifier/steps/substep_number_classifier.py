"""
SubStep number classifier.

This classifier finds smaller step numbers that appear inside subassembly boxes
or as naked substeps alongside main steps. These have a smaller font size than
regular step numbers.
"""

from typing import ClassVar

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.rule_based_classifier import (
    RuleBasedClassifier,
    StepNumberScore,
)
from build_a_long.pdf_extract.classifier.rules import (
    FontSizeSmallerThanRule,
    IsInstanceFilter,
    Rule,
    StepNumberTextRule,
    StepValueMaxFilter,
)
from build_a_long.pdf_extract.classifier.text import (
    extract_step_number_value,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    StepNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import Block, Text


class SubStepNumberClassifier(RuleBasedClassifier):
    """Classifier for substep numbers (small step numbers inside subassemblies).

    Substep numbers differ from main step numbers:
    - Smaller font size (typically ~70% of main step number size)
    - Lower values (1, 2, 3, 4 instead of 337, 338, 339)
    - Located inside subassembly boxes or as naked substeps

    The output is 'substep_number' which SubStepClassifier uses to pair
    with diagrams.

    Conflict resolution: StepNumbers are built first during the build phase,
    consuming their text blocks. SubStepNumbers that share the same blocks
    will fail to build, which is the desired behavior.
    """

    output: ClassVar[str] = "substep_number"
    requires: ClassVar[frozenset[str]] = frozenset()

    @property
    def min_score(self) -> float:
        return self.config.substep_number.min_score

    @property
    def rules(self) -> list[Rule]:
        config = self.config
        hints = config.font_size_hints
        substep_config = config.substep_number

        return [
            # Must be text
            IsInstanceFilter(Text),
            # Must look like a step number (digits only)
            StepNumberTextRule(
                weight=substep_config.text_weight,
                name="text_score",
                required=True,
            ),
            # Must be a small value (substeps are 1, 2, 3, 4... not 100+)
            StepValueMaxFilter(
                max_value=substep_config.max_value,
                weight=substep_config.value_weight,
                name="value_score",
                required=True,
            ),
            # Should have smaller font than main step numbers
            FontSizeSmallerThanRule(
                reference_size=hints.step_number_size,
                threshold_ratio=substep_config.size_ratio,
                weight=substep_config.font_size_weight,
                name="font_size_score",
            ),
        ]

    def _create_score(
        self,
        block: Block,
        components: dict[str, float],
        total_score: float,
    ) -> StepNumberScore:
        """Create a StepNumberScore that includes the parsed step value."""
        step_value = 0
        if isinstance(block, Text):
            parsed = extract_step_number_value(block.text)
            if parsed is not None:
                step_value = parsed

        return StepNumberScore(
            components=components,
            total_score=total_score,
            step_value=step_value,
        )

    def build(self, candidate: Candidate, result: ClassificationResult) -> StepNumber:
        """Construct a StepNumber element from a substep_number candidate.

        Note: We build a StepNumber element (not a separate SubStepNumber type)
        because the final element structure uses StepNumber for both main steps
        and substeps.
        """
        assert len(candidate.source_blocks) >= 1
        block = candidate.source_blocks[0]
        assert isinstance(block, Text)

        # Get step value from score (already parsed during scoring)
        score_details = candidate.score_details
        assert isinstance(score_details, StepNumberScore)
        value = score_details.step_value

        if value == 0:
            raise ValueError(
                f"Could not parse substep number from text: '{block.text}'"
            )

        return StepNumber(value=value, bbox=block.bbox)
