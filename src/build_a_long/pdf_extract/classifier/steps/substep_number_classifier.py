"""
SubStep number classifier.

This classifier finds smaller step numbers that appear inside subassembly boxes
or as naked substeps alongside main steps. These have a smaller font size than
regular step numbers.

Spatial constraints are handled by the parent classifiers:
- SubStepClassifier pairs substep_numbers with diagrams (spatial proximity scoring)
- SubAssemblyClassifier claims substeps inside its white boxes
- StepClassifier uses remaining substeps as naked substeps

Only substep_numbers that can pair with diagrams will be selected as part of
a SubStep. This naturally excludes page numbers in the bottom band since there
are no diagrams near them.

Sequential constraints (1, 2, 3 must be in order) are enforced by
SubAssemblyClassifier for each subassembly box independently.
"""

import logging
from collections.abc import Sequence
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
    InBottomBandFilter,
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
from build_a_long.pdf_extract.extractor.page_blocks import Blocks, Text

log = logging.getLogger(__name__)


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
    def max_score(self) -> float:
        # Lower than step_count (0.8) so "2x" wins over "2" when blocks overlap
        return 0.7

    @property
    def min_score(self) -> float:
        return self.config.substep_number.min_score

    @property
    def rules(self) -> Sequence[Rule]:
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
            # OPTIMIZATION: Exclude bottom 10% where page numbers typically live.
            # This avoids block exclusivity conflicts with page_number candidates.
            # May be removed if structural bonuses and font hints become sufficient.
            InBottomBandFilter(
                threshold_ratio=0.1,
                invert=True,  # Exclude bottom band, not include
                name="not_in_page_number_area",
            ),
        ]

    def _create_score(
        self,
        components: dict[str, float],
        total_score: float,
        source_blocks: Sequence[Blocks],
    ) -> StepNumberScore:
        """Create a StepNumberScore that includes the parsed step value."""
        step_value = 0
        block = source_blocks[0]  # Primary block that passed the rules
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

        # Use candidate.bbox which is the union of all source blocks
        return StepNumber(value=value, bbox=candidate.bbox)
