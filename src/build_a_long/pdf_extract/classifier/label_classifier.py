"""
Base class for label classifiers.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from build_a_long.pdf_extract.classifier.types import (
    Candidate,
    ClassificationHints,
    ClassifierConfig,
    RemovalReason,
    ScoreKey,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.page_elements import Element

if TYPE_CHECKING:
    from build_a_long.pdf_extract.classifier.classifier import Classifier
    from build_a_long.pdf_extract.extractor.lego_page_elements import LegoPageElement


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
    def calculate_scores(
        self,
        page_data: PageData,
        scores: Dict[str, Dict[ScoreKey, Any]],
        labeled_elements: Dict[Element, str],
    ) -> None:
        """Calculate the scores for the label."""
        pass

    @abstractmethod
    def classify(
        self,
        page_data: PageData,
        scores: Dict[str, Dict[ScoreKey, Any]],
        labeled_elements: Dict[Element, str],
        removal_reasons: Dict[int, RemovalReason],
        hints: "Optional[ClassificationHints]" = None,
        constructed_elements: "Optional[Dict[Element, LegoPageElement]]" = None,
        candidates: "Optional[Dict[str, List[Candidate]]]" = None,
    ) -> None:
        """Classify the elements for the label.

        Args:
            page_data: The page data containing all elements
            scores: Pre-calculated scores for all classifiers
            labeled_elements: Elements labeled so far (by earlier classifiers)
            removal_reasons: Reasons why elements were removed
            hints: Optional hints to guide classification (e.g., exclude specific elements)
            constructed_elements: Dict to store constructed LegoPageElements
            candidates: Dict to store all candidates (for re-evaluation/debugging)
        """
        pass
