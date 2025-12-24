"""
Rule-based classifier implementation.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Sequence

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
from build_a_long.pdf_extract.extractor.page_blocks import Blocks

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
    """Base class for classifiers that use a list of rules to score candidates.

    This class provides a declarative way to create classifiers using rules.
    Instead of implementing custom scoring logic, subclasses declare a list
    of Rule objects that are evaluated for each block.

    How It Works
    ------------

    1. Define your rules in the `rules` property
    2. Rules are evaluated sequentially for each block
    3. Each rule returns a score (0.0 to 1.0) or None (skipped)
    4. Required rules with score 0.0 cause immediate rejection
    5. Final score is weighted average of all applicable rules
    6. Blocks meeting `min_score` threshold become candidates

    Scoring Calculation
    -------------------

    .. code-block:: python

        final_score = sum(rule.score * rule.weight) / sum(rule.weight)

    - If any required rule scores 0.0, the block is rejected immediately
    - Rules can return None to be skipped (not counted in average)
    - All rule scores and weights are stored in the score_details for debugging

    Best Practices
    --------------

    **Rule Design:**
    - Use Filter rules to eliminate invalid blocks early
    - Use Score rules to rate blocks on intrinsic properties
    - Set `required=True` for rules that MUST pass (e.g., type filters)
    - Use weights to emphasize important scoring factors

    **Score Object:**
    - Override `_create_score()` to return custom Score subclass
    - Use custom scores to store parsed values (e.g., step numbers)
    - Always inherit from `Score` abstract base class

    **Source Blocks:**
    - Override `_get_additional_source_blocks()` to include related blocks
    - Default implementation finds visual effects (shadows, outlines)
    - Set `effects_margin` to automatically include nearby drawings/images

    Attributes (Override in Subclass)
    ----------------------------------

    - ``output``: The label name this classifier produces
    - ``requires``: Set of labels that must be classified first
    - ``rules``: List of Rule objects for scoring (see Rules section)
    - ``min_score``: Minimum score threshold for acceptance (default 0.0)
    - ``max_score``: Score scaling factor (default 1.0, use 0.8 for intrinsic)
    - ``effects_margin``: Distance to search for visual effects (default None)

    Score Capping (max_score)
    -------------------------

    Classifiers fall into two categories:

    **Intrinsic Classifiers (max_score = 0.8):**
    Identify elements based solely on their own properties (size, shape, text,
    position). These should set ``max_score = 0.8`` so composite classifiers
    can outcompete them when both match the same blocks.

    Examples: step_number, part_count, shine, page_number, progress_bar_bar

    **Important:** When adding ``max_score = 0.8`` to an intrinsic classifier,
    also scale down its ``min_score`` threshold by 0.8 (e.g., 0.5 → 0.4) to
    maintain the same acceptance rate.

    **Composite Classifiers (max_score = 1.0, default):**
    Assemble child elements into parent structures. These naturally score
    higher because they combine multiple verified children and add bonuses
    for complete structures.

    Examples: progress_bar, step, parts_list, part

    Example Implementation
    ----------------------

    .. code-block:: python

        class MyClassifier(RuleBasedClassifier):
            output = "my_label"
            requires = frozenset()  # Or frozenset({"dependency"})

            @property
            def max_score(self) -> float:
                return 0.8  # Intrinsic classifier

            @property
            def min_score(self) -> float:
                return 0.4  # Scaled from 0.5 for max_score=0.8

            @property
            def rules(self) -> Sequence[Rule]:
                return [
                    # Filter: Only accept Text blocks
                    IsInstanceFilter((Text,)),

                    # Required rule: Must be in top half of page
                    PositionScore(
                        scale=LinearScale({0.0: 1.0, 0.5: 0.0}),
                        weight=1.0,
                        required=True,
                    ),

                    # Optional scoring: Prefer larger text
                    FontSizeScore(
                        target_size=24.0,
                        weight=0.5,
                    ),
                ]

            # Optional: Custom score with parsed data
            def _create_score(
                self, components: dict[str, float], total: float,
                source_blocks: Sequence[Blocks]
            ) -> RuleScore:
                # Parse and store additional info from primary block
                value = self._parse_value(source_blocks[0])
                return MyCustomScore(
                    components=components,
                    total_score=total,
                    parsed_value=value,
                )

            def build(self, candidate, result) -> MyElement:
                score = candidate.score_details
                assert isinstance(score, MyCustomScore)
                return MyElement(
                    bbox=candidate.bbox,
                    value=score.parsed_value,
                )

    Built-in Hooks
    --------------

    Override these methods to customize behavior:

    - `_should_accept(score)`: Custom acceptance logic beyond min_score
    - `_create_score()`: Return custom Score subclass with additional data
    - `_get_additional_source_blocks()`: Include related blocks (shadows, etc.)

    Visual Effects Support
    ----------------------

    RuleBasedClassifier automatically includes nearby Drawing/Image blocks
    as visual effects (outlines, shadows) if `effects_margin` is set:

    .. code-block:: python

        @property
        def effects_margin(self) -> float | None:
            return 2.0  # Include blocks within 2 units

        @property
        def effects_max_area_ratio(self) -> float | None:
            return 5.0  # Effect can be at most 5x the primary block area

    This ensures that when a candidate wins, all associated visual effects
    are consumed together, preventing other classifiers from incorrectly
    using shadow/outline blocks.

    See Also
    --------
    - Classifier: Main orchestrator with comprehensive best practices
    - Rule: Base class for scoring rules
    - rules module: Available rule implementations
    """

    @property
    @abstractmethod
    def rules(self) -> Sequence[Rule]:
        """Get the list of rules for this classifier."""
        pass

    # TODO This property might want to be renamed, to threshold_score or similar
    @property
    def min_score(self) -> float:
        """Minimum score threshold for acceptance. Defaults to 0.0."""
        return 0.0

    @property
    def max_score(self) -> float:
        """Score scaling factor.

        The final score is multiplied by this value, allowing intrinsic
        classifiers to produce lower scores than composite classifiers.

        Intrinsic classifiers (that identify elements by their own properties)
        should set this to 0.8 to ensure composite classifiers can win when
        competing for the same blocks. See Score class docstring for details.

        Defaults to 1.0 (no scaling).
        """
        return 1.0

    @property
    def effects_margin(self) -> float | None:
        """Margin to expand block bbox to find visual effects (outlines, shadows).

        If None, no automatic effect finding is performed.
        Defaults to 2.0.
        """
        return 2.0

    def _create_score(
        self,
        components: dict[str, float],
        total_score: float,
        source_blocks: Sequence[Blocks],
    ) -> RuleScore:
        """Create the score object for a candidate.

        Subclasses can override this to return a more specific score type
        that contains additional information (e.g., parsed values, cluster
        validation results).

        Args:
            components: Dictionary of rule name to score
            total_score: The weighted total score from rules
            source_blocks: All blocks that will be part of the candidate.
                The primary block (that passed the rules) is source_blocks[0].
                Additional blocks from _get_additional_source_blocks() follow.

        Returns:
            A RuleScore (or subclass) instance

        TODO: Consider adding a cluster_rules property if multiple classifiers
        need to validate/score complete clusters. This would allow expressing
        cluster validation (e.g., count >= 3, cluster bbox aspect ratio) as
        declarative rules instead of imperative code in _create_score().
        For now, the imperative approach is simpler for the few classifiers
        that need it (e.g., LoosePartSymbolClassifier).
        """
        return RuleScore(components=components, total_score=total_score)

    def _score(self, result: ClassificationResult) -> None:
        """Score blocks using rules and add candidates.

        Iterates over all blocks, calls _score_block() for each, and adds
        resulting candidates to the result.

        Subclasses that need to create multiple candidates per block (e.g.,
        with variants) can override this method, call _score_block() to get
        the base candidate, then create variants and add them manually.
        """
        context = RuleContext(result.page_data, self.config, result)

        for block in result.page_data.blocks:
            candidate = self._score_block(block, context, result)
            if candidate is not None:
                result.add_candidate(candidate)

    def _score_block(
        self,
        block: Blocks,
        context: RuleContext,
        result: ClassificationResult,
    ) -> Candidate | None:
        """Score a single block and return a candidate if it passes.

        This method applies rules to a block, builds source blocks, creates
        the score object, and returns a Candidate if the block passes all
        requirements. Returns None if the block fails any required rule or
        doesn't meet the minimum score threshold.

        Subclasses can call this directly when they need to create multiple
        variants of a candidate (e.g., with/without optional attachments).

        Args:
            block: The block to score
            context: Rule context with page data and config
            result: Classification result for accessing other candidates

        Returns:
            A Candidate if the block passes, None otherwise
        """
        rules = self.rules
        components: dict[str, float] = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for rule in rules:
            score = rule.calculate(block, context)

            # If rule returns None, it's skipped (not applicable)
            if score is None:
                continue

            # If required rule fails (score 0), fail the block immediately
            if rule.required and score == 0.0:
                return None

            rule_weight = rule.weight
            weighted_sum += score * rule_weight
            total_weight += rule_weight
            components[rule.name] = score

        if total_weight == 0:
            return None

        # Calculate final score from rules, scaled by max_score
        final_score = (weighted_sum / total_weight) * self.max_score

        # Build source blocks list, deduplicating as we go
        seen_ids: set[int] = {block.id}
        source_blocks: list[Blocks] = [block]

        # Add any classifier-specific additional source blocks
        for b in self._get_additional_source_blocks(block, result):
            if b.id not in seen_ids:
                seen_ids.add(b.id)
                source_blocks.append(b)

        # Create score object (subclasses can override _create_score)
        # This can validate the complete cluster and adjust the score
        score_details = self._create_score(components, final_score, source_blocks)

        # Get actual score (may differ from final_score after validation)
        actual_score = score_details.score()

        # Check classifier-specific acceptance logic on the actual score
        if not self._should_accept(actual_score):
            log.debug(
                "[%s] block_id=%s rejected: score=%.3f < min_score=%.3f components=%s",
                self.output,
                block.id,
                actual_score,
                self.min_score,
                components,
            )
            return None

        log.debug(
            "[%s] block_id=%s cluster accepted: score=%.3f components=%s",
            self.output,
            block.id,
            actual_score,
            components,
        )

        # Compute bbox as the union of all source blocks
        # This ensures the candidate bbox matches the source_blocks union,
        # required by validation (assert_element_bbox_matches_source_and_children)
        candidate_bbox = BBox.union_all([b.bbox for b in source_blocks])

        # Create and return candidate
        return Candidate(
            bbox=candidate_bbox,
            label=self.output,
            score=actual_score,
            score_details=score_details,
            source_blocks=source_blocks,
        )

    def _get_additional_source_blocks(
        self, block: Blocks, result: ClassificationResult
    ) -> Sequence[Blocks]:
        """Get additional source blocks to include with the candidate.

        Subclasses can override this to include related blocks (e.g.,
        overlapping drawings, drop shadows) in the candidate's source_blocks.
        These blocks will be marked as removed if the candidate wins.

        The default implementation automatically includes Drawing/Image blocks
        that appear to be visual effects (outlines, shadows) by calling
        find_contained_effects if self.effects_margin is not None.
        """
        margin = self.effects_margin
        if margin is not None:
            return find_contained_effects(
                block,
                result.page_data.blocks,
                margin=margin,
            )
        return []

    def _should_accept(self, score: float) -> bool:
        """Determine if a score is high enough to be a candidate.

        Subclasses can override this.
        """
        return score >= self.min_score
