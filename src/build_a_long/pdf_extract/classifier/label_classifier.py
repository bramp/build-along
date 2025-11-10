"""
Base class for label classifiers.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.page_blocks import Text

if TYPE_CHECKING:
    from build_a_long.pdf_extract.classifier.classifier import Classifier

# TODO Maybe classifers need a interface, where they have
#      either scoring functions, or filter functions.
#      * Expected a page number, filter text that isn't numeric.
#      * Expected the number in the corner, score based on position.
#      Then we can abstract out common code/functions, to keep the code DRY.


class LabelClassifier(ABC):
    """Abstract base class for a single label classifier."""

    # Class-level metadata to declare pipeline dependencies. Subclasses should
    # override these to advertise what labels they produce and require.
    outputs: set[str] = set()
    requires: set[str] = set()

    def __init__(self, config: ClassifierConfig, classifier: Classifier):
        self.config = config
        self.classifier = classifier

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
    def evaluate(
        self,
        page_data: PageData,
        result: ClassificationResult,
    ) -> None:
        """Evaluate elements and create candidates for the label.

        This method should:
        1. Score each element for this label
        2. Attempt to construct LegoPageElements from viable candidates
        3. Store candidates (both successful and failed) with rejection reasons

        Args:
            page_data: The page data containing all elements
            result: The classification result to populate with candidates
        """
        pass

    @abstractmethod
    def classify(
        self,
        page_data: PageData,
        result: ClassificationResult,
    ) -> None:
        """Classify the elements for the label by selecting winners.

        Args:
            page_data: The page data containing all elements
            result: The classification result containing candidates and to update
                with winners
        """
        pass
