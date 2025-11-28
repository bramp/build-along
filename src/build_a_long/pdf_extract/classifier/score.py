"""Score base class and type definitions."""

from __future__ import annotations

from abc import abstractmethod
from typing import Annotated

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
