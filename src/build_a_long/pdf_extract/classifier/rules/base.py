"""Base classes for scoring rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.page_blocks import Block


@dataclass
class RuleContext:
    """Context passed to rules during evaluation."""

    page_data: PageData
    config: ClassifierConfig


class Rule(ABC):
    """Abstract base class for scoring rules.

    Attributes:
        name: A descriptive name for the rule (used in debug logs/score details).
        weight: The default weight (0.0 or greater) contribution of this rule to the
            total score.
        required: If True, a calculated score of 0.0 will immediately disqualify
            the candidate, regardless of how high other rule scores are.
            Useful for hard filters (e.g. "Must be Text").
    """

    name: str
    weight: float = 1.0

    # If true, a score of 0.0 from this rule will immediately disqualify the candidate
    # regardless of other scores.
    required: bool = False

    @abstractmethod
    def calculate(self, block: Block, context: RuleContext) -> float | None:
        """Calculate a score between 0.0 and 1.0 for the given block.

        Returns:
            float: Score between 0.0 and 1.0
            None: If the rule cannot be evaluated (e.g. missing hints) and should
                  be ignored (not contributing to the weighted sum).
        """
        pass


class Filter(Rule):
    """Base class for filters (rules that don't contribute to score)."""

    weight: float = 0.0
    required: bool = True


class IsInstanceFilter(Filter):
    """Filter that checks if a block is of a specific type or types."""

    def __init__(
        self, block_type: type[Block] | tuple[type[Block], ...], name: str = ""
    ):
        if isinstance(block_type, tuple):
            names = "|".join(t.__name__ for t in block_type)
            default_name = f"IsInstance({names})"
        else:
            default_name = f"IsInstance({block_type.__name__})"

        self.name = name if name else default_name
        self.block_type = block_type

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        return 1.0 if isinstance(block, self.block_type) else 0.0


class MaxScoreRule(Rule):
    """Rule that returns the maximum score from a list of sub-rules.

    Useful when there are multiple ways to satisfy a criteria (e.g. matching
    one of several font size hints).
    """

    def __init__(
        self,
        rules: list[Rule],
        weight: float = 1.0,
        name: str = "MaxScore",
        required: bool = False,
    ):
        self.name = name
        self.rules = rules
        self.weight = weight
        self.required = required

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        scores = []
        for rule in self.rules:
            score = rule.calculate(block, context)
            if score is not None:
                scores.append(score)

        if not scores:
            return None

        return max(scores)
