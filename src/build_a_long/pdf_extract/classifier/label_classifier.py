"""
Base class for label classifiers.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from build_a_long.pdf_extract.classifier.types import (
    ClassificationHints,
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.extractor import PageData

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
        hints: "Optional[ClassificationHints]",
    ) -> None:
        """Classify the elements for the label by selecting winners.

        Args:
            page_data: The page data containing all elements
            result: The classification result containing candidates and to update with winners
            hints: Optional hints to guide classification (e.g., exclude specific elements)
        """
        pass
