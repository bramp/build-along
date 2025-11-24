"""
Base class for label classifiers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.extractor.page_blocks import Text

# TODO Maybe classifers need a interface, where they have
#      either scoring functions, or filter functions.
#      * Expected a page number, filter text that isn't numeric.
#      * Expected the number in the corner, score based on position.
#      Then we can abstract out common code/functions, to keep the code DRY.


@dataclass(frozen=True)
class LabelClassifier(ABC):
    """Abstract base class for a single label classifier.

    Classifiers are frozen dataclasses to enforce statelessness - they cannot
    modify their attributes after initialization. All state must be stored in
    ClassificationResult.
    """

    config: ClassifierConfig

    # Class-level metadata to declare pipeline dependencies.
    # Subclasses should override these at the class level
    outputs: ClassVar[frozenset[str]] = frozenset()
    requires: ClassVar[frozenset[str]] = frozenset()

    def _score_font_size(self, element: Text, expected_size: float | None) -> float:
        """Score based on how well font size matches an expected size.

        Returns 1.0 if font size matches the expected size exactly, scaling down
        based on the difference. Returns 0.5 if no expected size is provided or
        if the element has no font size metadata.

        Args:
            element: Text element to score
            expected_size: Expected font size, or None if no hint available

        Returns:
            Score from 0.0 to 1.0
        """
        # Use the font_size from the PDF metadata, not bbox height
        actual_size = element.font_size
        if actual_size is None or expected_size is None:
            # No font size metadata or no hint available, return neutral score
            return 0.5

        # Exact match gets score of 1.0
        if abs(actual_size - expected_size) < 0.01:
            return 1.0

        # Calculate relative difference
        diff_ratio = abs(actual_size - expected_size) / expected_size

        # Score decreases as difference increases
        # Within 10% difference: score > 0.8
        # Within 20% difference: score > 0.6
        # Within 50% difference: score > 0.0
        return max(0.0, 1.0 - (diff_ratio * 2.0))

    @abstractmethod
    def score(self, result: ClassificationResult) -> None:
        """Score elements and create candidates WITHOUT construction.

        This method should:
        1. Score each element for this label
        2. Create Candidate objects with scores and score_details
        3. Set constructed=None and failure_reason=None for all candidates
        4. Store candidates in the result via result.add_candidate()

        This is the first phase of the two-phase classification process.
        Construction happens later in construct().

        Args:
            result: The classification result to populate with candidates
        """
        pass

    @abstractmethod
    def construct(self, result: ClassificationResult) -> None:
        """Construct LegoPageElements from candidates.

        This method should:
        1. Get candidates for each label this classifier outputs
        2. Select which candidates to construct (all, winners only, etc.)
        3. Build LegoPageElements from selected candidates
        4. Update candidate.constructed and candidate.failure_reason

        Default pattern for constructing all candidates:
            for label in self.outputs:
                candidates = result.get_candidates(label)
                for candidate in candidates:
                    try:
                        elem = self._construct_single(candidate, result)
                        candidate.constructed = elem
                    except Exception as e:
                        candidate.failure_reason = str(e)

        This is the second phase of the two-phase classification process.
        Scoring happens first in score().

        Args:
            result: The classification result containing candidates to construct
        """
        pass
