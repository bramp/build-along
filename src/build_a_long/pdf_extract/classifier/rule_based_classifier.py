"""
Rule-based classifier implementation.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from build_a_long.pdf_extract.classifier.block_filter import (
    find_contained_effects,
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
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Block, Blocks

if TYPE_CHECKING:
    pass

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


class StepNumberScore(RuleScore):
    """Score for step number candidates that includes the parsed step value.

    This avoids re-parsing the step number from source blocks when the value
    is needed later (e.g., for building StepNumber elements or sorting).
    """

    step_value: int
    """The parsed step number value (e.g., 1, 2, 3, 42)."""


class RuleBasedClassifier(LabelClassifier):
    """Base class for classifiers that use a list of rules to score candidates."""

    @property
    @abstractmethod
    def rules(self) -> Sequence[Rule]:
        """Get the list of rules for this classifier."""
        pass

    @property
    def min_score(self) -> float:
        """Minimum score threshold for acceptance. Defaults to 0.0."""
        return 0.0

    @property
    def effects_margin(self) -> float | None:
        """Margin to expand block bbox to find visual effects (outlines, shadows).

        If None, no automatic effect finding is performed.
        Defaults to 2.0.
        """
        return 2.0

    @property
    def effects_max_area_ratio(self) -> float | None:
        """Maximum ratio of effect block area to primary block area.

        Used to avoid consuming unrelated large blocks as effects.
        Defaults to None (no ratio check).
        """
        return None

    def _create_score(
        self,
        block: Block,
        components: dict[str, float],
        total_score: float,
    ) -> RuleScore:
        """Create the score object for a candidate.

        Subclasses can override this to return a more specific score type
        that contains additional information (e.g., parsed values).

        Args:
            block: The block being scored
            components: Dictionary of rule name to score
            total_score: The weighted total score

        Returns:
            A RuleScore (or subclass) instance
        """
        return RuleScore(components=components, total_score=total_score)

    def _score(self, result: ClassificationResult) -> None:
        """Score blocks using rules."""
        context = RuleContext(result.page_data, self.config, result)
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
                    # log.debug(
                    #    "[%s] block_id=%s failed required rule '%s'",
                    #    self.output,
                    #    block.id,
                    #    rule.name,
                    # )
                    break

                rule_weight = rule.weight  # Using direct weight from Rule instance

                weighted_sum += score * rule_weight
                total_weight += rule_weight
                components[rule.name] = score

            if failed:
                continue

            # Calculate final score
            final_score = weighted_sum / total_weight if total_weight > 0 else 0.0

            # Check classifier-specific acceptance logic
            if not self._should_accept(final_score):
                log.debug(
                    "[%s] block_id=%s "
                    "rejected: score=%.3f < min_score=%.3f components=%s",
                    self.output,
                    block.id,
                    final_score,
                    self.min_score,
                    components,
                )
                continue

            log.debug(
                "[%s] block_id=%s accepted: score=%.3f components=%s",
                self.output,
                block.id,
                final_score,
                components,
            )

            # Build source blocks list, deduplicating as we go
            seen_ids: set[int] = {block.id}
            source_blocks: list[Blocks] = [block]

            # Automatically find visual effects (outlines, shadows)
            margin = self.effects_margin
            if margin is not None:
                effects = find_contained_effects(
                    block,
                    result.page_data.blocks,
                    margin=margin,
                    max_area_ratio=self.effects_max_area_ratio,
                )
                for b in effects:
                    if b.id not in seen_ids:
                        seen_ids.add(b.id)
                        source_blocks.append(b)

            # Add any classifier-specific additional source blocks
            for b in self._get_additional_source_blocks(block, result):
                if b.id not in seen_ids:
                    seen_ids.add(b.id)
                    source_blocks.append(b)

            # Create score object (subclasses can override _create_score)
            score_details = self._create_score(block, components, final_score)

            # Compute bbox as the union of all source blocks
            # This ensures the candidate bbox matches the source_blocks union,
            # required by validation (assert_element_bbox_matches_source_and_children)
            candidate_bbox = BBox.union_all([b.bbox for b in source_blocks])

            # Create candidate
            candidate = Candidate(
                bbox=candidate_bbox,
                label=self.output,
                score=final_score,
                score_details=score_details,
                source_blocks=source_blocks,
            )
            result.add_candidate(candidate)

    def _get_additional_source_blocks(
        self, block: Block, result: ClassificationResult
    ) -> Sequence[Blocks]:
        """Get additional source blocks to include with the candidate.

        Subclasses can override this to include related blocks (e.g.,
        overlapping drawings, drop shadows) in the candidate's source_blocks.
        These blocks will be marked as removed if the candidate wins.
        """
        return []

    def _should_accept(self, score: float) -> bool:
        """Determine if a score is high enough to be a candidate.

        Subclasses can override this.
        """
        return score >= self.min_score
