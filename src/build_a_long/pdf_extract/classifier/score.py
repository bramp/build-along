"""Score base class and type definitions."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Annotated, Protocol

from annotated_types import Ge, Le
from pydantic import BaseModel, ConfigDict

# Weight value constrained to [0.0, 1.0] range
Weight = Annotated[float, Ge(0), Le(1)]


class Score(BaseModel):
    """Abstract base class for score_details objects.

    All score_details stored in Candidate objects must inherit from this class.
    This ensures a consistent interface for accessing the final score value.

    The score() method MUST return a value in the range [0.0, 1.0] where:
    - 0.0 indicates lowest confidence/worst match
    - 1.0 indicates highest confidence/best match

    IMPORTANT: Pairing Classifiers and Constraint Solver Integration
    -----------------------------------------------------------------
    When a classifier creates candidates that PAIR multiple elements together
    (e.g., Part pairing PartCount with PartImage), the score() method MUST
    incorporate ALL factors that distinguish good pairings from bad ones:

    - Distance between elements (closer is usually better)
    - Alignment quality (left/center/right alignment as expected)
    - Relative positioning (above/below/beside as expected)
    - Any domain-specific heuristics

    This is CRITICAL because the constraint solver maximizes total score to
    select optimal one-to-one matchings. If all candidate pairings have similar
    scores, the solver cannot distinguish good from bad and may choose suboptimal
    pairings.

    See rules/scoring.py for utility functions:
    - score_exponential_decay(): For distance-based scoring
    - score_triangular(): For value-in-range scoring (e.g., alignment)
    - score_linear(): For linear interpolation

    Score classes should use generic Candidate[T] types to enable automatic
    constraint mapping by SchemaConstraintGenerator:

        class _PartsListScore(Score):
            # Candidate[Part] auto-maps to PartsList.parts (Sequence[Part])
            part_candidates: list[Candidate[Part]] = []

        class _PartPairScore(Score):
            # Candidate[PartCount] auto-maps to Part.count
            part_count_candidate: Candidate[PartCount]
            # Candidate[PartImage] auto-maps to Part.diagram
            part_image_candidate: Candidate[PartImage]

    Example implementations:
        class _PageNumberScore(Score):
            text_score: float
            position_score: float

            def score(self) -> Weight:
                # Returns normalized score in range [0.0, 1.0]
                return (self.text_score + self.position_score) / 2.0
    """

    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def score(self) -> Weight:
        """Return the final computed score value.

        Returns:
            Weight: A score value in the range [0.0, 1.0] where 0.0 is the
                lowest confidence and 1.0 is the highest confidence.
        """
        ...


class HasScore(Protocol):
    """Protocol for objects that have a score attribute."""

    @property
    def score(self) -> float: ...


def find_best_scoring[S: HasScore](items: Sequence[S]) -> S | None:
    """Find the item with the highest score in the list.

    Args:
        items: List of items with a score property.

    Returns:
        The item with the highest score, or None if the list is empty.
    """
    if not items:
        return None
    return max(items, key=lambda x: x.score)
