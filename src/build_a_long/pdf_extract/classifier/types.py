"""
Data classes for the classifier.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

from build_a_long.pdf_extract.extractor.page_elements import Element


# Score key can be either a single Element or a tuple of Elements (for pairings)
ScoreKey = Union[Element, Tuple[Element, ...]]


@dataclass
class RemovalReason:
    """Tracks why an element was removed during classification."""

    reason_type: str
    """Type of removal: 'child_bbox' or 'similar_bbox'"""

    target_element: Any
    """The element that caused this removal"""


@dataclass(frozen=True)
class ClassifierConfig:
    """Configuration for the classifier."""

    min_confidence_threshold: float = 0.5

    page_number_text_weight: float = 0.7
    page_number_position_weight: float = 0.3
    page_number_position_scale: float = 50.0
    page_number_page_value_weight: float = 1.0

    step_number_text_weight: float = 0.8
    step_number_size_weight: float = 0.2

    def __post_init__(self) -> None:
        for weight in self.__dict__.values():
            if weight < 0:
                raise ValueError("All weights must be greater than or equal to 0.")


@dataclass
class ClassificationResult:
    """Represents the outcome of a single classification run.

    This class encapsulates the results of element classification, including
    labels, scores, and removal information. External code should use the
    accessor methods rather than accessing internal fields directly to maintain
    encapsulation.
    """

    _labeled_elements: Dict[Element, str] = field(default_factory=dict)
    """Maps elements to their assigned labels (e.g., element -> 'page_number')"""
    _scores: Dict[str, Dict[ScoreKey, Any]] = field(default_factory=dict)
    """Maps label name to a dict of score keys to score objects.
    
    Score keys can be either:
    - A single Element (for most classifiers)
    - A tuple of Elements (for pairings, e.g., part_image uses (part_count, image))
    
    Score values are classifier-specific dataclasses (e.g., _PageNumberScore).
    """
    warnings: List[str] = field(default_factory=list)
    _removal_reasons: Dict[int, RemovalReason] = field(default_factory=dict)
    """Maps element IDs to the reason they were removed"""

    # Persisted relations discovered during classification. For now, we record
    # part image pairings as (part_count_text, image) tuples.
    part_image_pairs: List[Tuple[Any, Any]] = field(default_factory=list)

    def get_label(self, element: Element) -> str | None:
        """Get the label for an element from this classification result.

        Args:
            element: The element to get the label for

        Returns:
            The label string if found, None otherwise
        """
        return self._labeled_elements.get(element)

    def get_elements_by_label(self, label: str) -> List[Element]:
        """Get all elements with the given label.

        Args:
            label: The label to search for

        Returns:
            List of elements with that label
        """
        return [elem for elem, lbl in self._labeled_elements.items() if lbl == label]

    def is_removed(self, element: Element) -> bool:
        """Check if an element has been marked for removal.

        Args:
            element: The element to check

        Returns:
            True if the element is marked for removal, False otherwise
        """
        return id(element) in self._removal_reasons

    def get_removal_reason(self, element: Element) -> RemovalReason | None:
        """Get the reason why an element was removed.

        Args:
            element: The element to get the removal reason for

        Returns:
            The RemovalReason if the element was removed, None otherwise
        """
        return self._removal_reasons.get(id(element))

    def get_scores_for_label(self, label: str) -> Dict[ScoreKey, Any]:
        """Get all scores for a specific label.

        Args:
            label: The label to get scores for

        Returns:
            Dictionary mapping score keys to score objects for that label
        """
        return self._scores.get(label, {})

    def has_label(self, label: str) -> bool:
        """Check if any elements have been assigned the given label.

        Args:
            label: The label to check for

        Returns:
            True if at least one element has this label, False otherwise
        """
        return label in self._scores or any(
            lbl == label for lbl in self._labeled_elements.values()
        )


@dataclass
class ClassificationHints:
    """Hints to guide the classification process."""

    element_constraints: Dict[int, str] = field(default_factory=dict)
    global_goals: Dict[int, str] = field(default_factory=dict)
