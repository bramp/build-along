"""
Base class for label classifiers.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Set

from build_a_long.pdf_extract.classifier.types import ClassifierConfig
from build_a_long.pdf_extract.extractor import PageData

if TYPE_CHECKING:
    from build_a_long.pdf_extract.classifier.classifier import Classifier


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
        scores: Dict[Any, Dict[str, float]],
        labeled_elements: Dict[str, Any],
    ) -> None:
        """Calculate the scores for the label."""
        pass

    @abstractmethod
    def classify(
        self,
        page_data: PageData,
        scores: Dict[Any, Dict[str, float]],
        labeled_elements: Dict[str, Any],
        to_remove: Set[int],
    ) -> None:
        """Classify the elements for the label."""
        pass
