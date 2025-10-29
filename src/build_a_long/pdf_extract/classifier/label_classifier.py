"""
Base class for label classifiers.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

from build_a_long.pdf_extract.classifier.types import (
    ClassifierConfig,
    RemovalReason,
    ScoreKey,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.page_elements import Element

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
        to_remove: Dict[int, RemovalReason],
    ) -> None:
        """Classify the elements for the label."""
        pass
