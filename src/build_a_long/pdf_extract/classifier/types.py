"""
Data classes for the classifier.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


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
    """Represents the outcome of a single classification run."""

    labeled_elements: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[Any, Dict[str, float]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    to_remove: Dict[int, RemovalReason] = field(default_factory=dict)
    """Maps element IDs to the reason they were removed"""
    # Persisted relations discovered during classification. For now, we record
    # part image pairings as (part_count_text, image) tuples.
    part_image_pairs: List[Tuple[Any, Any]] = field(default_factory=list)


@dataclass
class ClassificationHints:
    """Hints to guide the classification process."""

    element_constraints: Dict[int, str] = field(default_factory=dict)
    global_goals: Dict[int, str] = field(default_factory=dict)
