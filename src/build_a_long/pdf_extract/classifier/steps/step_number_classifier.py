"""
Step number classifier.
"""

import logging
from collections.abc import Sequence

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.constraint_model import ConstraintModel
from build_a_long.pdf_extract.classifier.rule_based_classifier import (
    RuleBasedClassifier,
    StepNumberScore,
)
from build_a_long.pdf_extract.classifier.rules import (
    FontSizeMatch,
    InBottomBandFilter,
    IsInstanceFilter,
    Rule,
    StepNumberTextRule,
)
from build_a_long.pdf_extract.classifier.rules.scale import LinearScale
from build_a_long.pdf_extract.classifier.text import (
    extract_step_number_value,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    StepNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import Blocks, Text


class StepNumberClassifier(RuleBasedClassifier):
    """Classifier for step numbers."""

    output = "step_number"
    requires = frozenset()

    @property
    def min_score(self) -> float:
        return self.config.step_number.min_score

    @property
    def rules(self) -> Sequence[Rule]:
        config = self.config
        step_number_size = config.font_size_hints.step_number_size or 10.0
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
                scale=LinearScale(
                    {
                        step_number_size * 0.5: 0.0,
                        step_number_size: 1.0,
                        step_number_size * 1.5: 0.0,
                    }
                ),
                weight=config.step_number.font_size_weight,
                name="font_size_score",
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

    def declare_constraints(
        self, model: ConstraintModel, result: ClassificationResult
    ) -> None:
        """Declare constraints for step_number candidates.

        Constraints:
        - Uniqueness: At most one step_number per step_value (e.g., only one "1")
        - The solver will pick the highest-scoring candidate for each step_value
        """
        candidates = result.get_candidates(self.output)
        candidates_in_model = [c for c in candidates if model.has_candidate(c)]

        if not candidates_in_model:
            return

        # Group by step_value
        by_value: dict[int, list[Candidate]] = {}
        for cand in candidates_in_model:
            if isinstance(cand.score_details, StepNumberScore):
                step_value = cand.score_details.step_value
                if step_value not in by_value:
                    by_value[step_value] = []
                by_value[step_value].append(cand)

        # Add at_most_one constraint for each step_value
        log = logging.getLogger(__name__)
        for step_value, cands in by_value.items():
            if len(cands) > 1:
                model.at_most_one_of(cands)
                log.debug(
                    "[step_number] Uniqueness constraint: at most one step_number "
                    "with value=%d (%d candidates)",
                    step_value,
                    len(cands),
                )

    def build(self, candidate: Candidate, result: ClassificationResult) -> StepNumber:
        """Construct a StepNumber element from a candidate.

        The candidate may include additional source blocks (e.g., text outline
        effects) beyond the primary Text block.
        """
        # Get the primary text block (first in source_blocks)
        assert len(candidate.source_blocks) >= 1
        block = candidate.source_blocks[0]
        assert isinstance(block, Text)

        # Get step value from score (already parsed during scoring)
        score_details = candidate.score_details
        assert isinstance(score_details, StepNumberScore)
        value = score_details.step_value

        if value == 0:
            raise ValueError(f"Could not parse step number from text: '{block.text}'")

        # Use candidate.bbox which is the union of all source blocks
        return StepNumber(value=value, bbox=candidate.bbox)
