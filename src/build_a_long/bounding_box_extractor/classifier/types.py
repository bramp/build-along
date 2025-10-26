"""
Data classes for the classifier.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass
class ClassifierConfig:
    """Configuration for the classifier."""

    min_confidence_threshold: float = 0.5
    page_number_text_weight: float = 0.7
    page_number_position_weight: float = 0.3
    page_number_position_scale: float = 50.0
    step_number_text_weight: float = 0.8
    step_number_size_weight: float = 0.2


@dataclass
class ClassificationResult:
    """Represents the outcome of a single classification run."""

    labeled_elements: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[Any, Dict[str, float]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    to_remove: Set[int] = field(default_factory=set)


@dataclass
class ClassificationHints:
    """Hints to guide the classification process."""

    element_constraints: Dict[int, str] = field(default_factory=dict)
    global_goals: Dict[int, str] = field(default_factory=dict)
