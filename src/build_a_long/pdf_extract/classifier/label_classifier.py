"""
Base class for label classifiers.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

from build_a_long.pdf_extract.classifier.types import (
    Candidate,
    ClassificationHints,
    ClassifierConfig,
    RemovalReason,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.page_elements import Element

if TYPE_CHECKING:
    from build_a_long.pdf_extract.classifier.classifier import Classifier
    from build_a_long.pdf_extract.extractor.lego_page_elements import LegoPageElement

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

    def __init__(self, config: ClassifierConfig, classifier: "Classifier"):
        self.config = config
        self.classifier = classifier

    @abstractmethod
    def evaluate(
        self,
        page_data: PageData,
        labeled_elements: Dict[Element, str],
        candidates: "Dict[str, List[Candidate]]",
    ) -> None:
        """Evaluate elements and create candidates for the label.

        This method should:
        1. Score each element for this label
        2. Attempt to construct LegoPageElements from viable candidates
        3. Store candidates (both successful and failed) with rejection reasons

        Args:
            page_data: The page data containing all elements
            labeled_elements: Elements labeled by earlier classifiers
            candidates: Dict to store all candidates with scores and failure reasons
        """
        pass

    @abstractmethod
    def classify(
        self,
        page_data: PageData,
        labeled_elements: Dict[Element, str],
        removal_reasons: Dict[int, RemovalReason],
        hints: "Optional[ClassificationHints]",
        constructed_elements: "Dict[Element, LegoPageElement]",
        candidates: "Dict[str, List[Candidate]]",
    ) -> None:
        """Classify the elements for the label by selecting winners.

        Args:
            page_data: The page data containing all elements
            labeled_elements: Elements labeled so far (by earlier classifiers)
            removal_reasons: Reasons why elements were removed
            hints: Optional hints to guide classification (e.g., exclude specific elements)
            constructed_elements: Dict to store constructed LegoPageElements
            candidates: Dict to store all candidates (for re-evaluation/debugging)
        """
        pass
