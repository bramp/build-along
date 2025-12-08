"""
Rule-based classifier implementation.
"""

from __future__ import annotations

import logging
from abc import abstractmethod

from build_a_long.pdf_extract.classifier.block_filter import (
    find_text_outline_effects,
)
from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.rules import Rule, RuleContext
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.page_blocks import Text

log = logging.getLogger(__name__)


class RuleScore(Score):
    """Generic score based on rules."""

    components: dict[str, float]
    total_score: float

    def score(self) -> Weight:
        return self.total_score

    def get(self, rule_name: str, default: float = 0.0) -> float:
        """Get the score for a specific rule name."""
        return self.components.get(rule_name, default)


class RuleBasedClassifier(LabelClassifier):
    """Base class for classifiers that use a list of rules to score candidates."""

    @property
    @abstractmethod
    def rules(self) -> list[Rule]:
        """Get the list of rules for this classifier."""
        pass

    @property
    def min_score(self) -> float:
        """Minimum score threshold for acceptance. Defaults to 0.0."""
        return 0.0

    def _score(self, result: ClassificationResult) -> None:
        """Score blocks using rules."""
        context = RuleContext(result.page_data, self.config)
        rules = self.rules

        for block in result.page_data.blocks:
            components = {}
            weighted_sum = 0.0
            total_weight = 0.0
            failed = False

            for rule in rules:
                score = rule.calculate(block, context)

                # If rule returns None, it's skipped (not applicable)
                if score is None:
                    continue

                # If required rule fails (score 0), fail the block immediately
                if rule.required and score == 0.0:
                    failed = True
                    break

                rule_weight = rule.weight  # Using direct weight from Rule instance

                weighted_sum += score * rule_weight
                total_weight += rule_weight
                components[rule.name] = score

            if failed:
                continue

            # Calculate final score
            if total_weight > 0:
                final_score = weighted_sum / total_weight
            else:
                final_score = 0.0

            # Check classifier-specific acceptance logic
            if not self._should_accept(final_score):
                continue

            # Build source blocks list, including text outline effects for Text blocks
            source_blocks: list = [block]
            if isinstance(block, Text):
                outline_effects = find_text_outline_effects(
                    block, result.page_data.blocks
                )
                source_blocks.extend(outline_effects)

            # Create candidate
            candidate = Candidate(
                bbox=block.bbox,
                label=self.output,
                score=final_score,
                score_details=RuleScore(components=components, total_score=final_score),
                source_blocks=source_blocks,
            )
            result.add_candidate(candidate)

    def _should_accept(self, score: float) -> bool:
        """Determine if a score is high enough to be a candidate.

        Subclasses can override this.
        """
        return score >= self.min_score
