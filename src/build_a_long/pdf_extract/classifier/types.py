"""
Data classes for the classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from build_a_long.pdf_extract.extractor.page_elements import Element

if TYPE_CHECKING:
    from build_a_long.pdf_extract.extractor.lego_page_elements import LegoPageElement


# Score key can be either a single Element or a tuple of Elements (for pairings)
ScoreKey = Union[Element, Tuple[Element, ...]]


@dataclass
class RemovalReason:
    """Tracks why an element was removed during classification."""

    reason_type: str
    """Type of removal: 'child_bbox' or 'similar_bbox'"""

    target_element: Any
    """The element that caused this removal"""


@dataclass
class Candidate:
    """A candidate element with its score and constructed LegoElement.

    Represents a single element that was considered for a particular label,
    including its score, the constructed LegoPageElement (if successful),
    and information about why it succeeded or failed.

    This enables:
    - Re-evaluation with hints (exclude specific candidates)
    - Debugging (see all candidates and why they won/lost)
    - UI support (show users alternatives)
    """

    source_element: Element
    """The raw element that was scored"""

    label: str
    """The label this candidate would have (e.g., 'page_number')"""

    # TODO Maybe score is redudant with score_details?
    score: float
    """Combined score (0.0-1.0)"""

    score_details: Any
    """The detailed score object (e.g., _PageNumberScore)"""

    constructed: "Optional[LegoPageElement]"
    """The constructed LegoElement if parsing succeeded, None if failed"""

    failure_reason: "Optional[str]" = None
    """Why construction failed, if it did"""

    # TODO Is this redudant with being in the constructed_elements?
    is_winner: bool = False
    """Whether this candidate was selected as the winner"""


@dataclass(frozen=True)
class ClassifierConfig:
    """Configuration for the classifier."""

    # TODO Not sure what this value is used for
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
    labels, scores, and removal information. The candidates field is now the
    primary source of truth for classification results, containing all scored
    elements, their constructed LegoPageElements, and winner information.

    External code should use the accessor methods rather than accessing internal
    fields directly to maintain encapsulation.
    """

    warnings: List[str] = field(default_factory=list)

    _removal_reasons: Dict[int, RemovalReason] = field(default_factory=dict)
    """Maps element IDs to the reason they were removed"""

    constructed_elements: "Dict[Element, LegoPageElement]" = field(default_factory=dict)
    """Maps source elements to their constructed LegoPageElements.
    
    Only contains elements that were successfully labeled and constructed.
    The builder should use these pre-constructed elements rather than
    re-parsing the source elements.
    """

    candidates: Dict[str, List[Candidate]] = field(default_factory=dict)
    """Maps label names to lists of all candidates considered for that label.
    
    Each candidate includes:
    - The source element
    - Its score and score details
    - The constructed LegoPageElement (if successful)
    - Failure reason (if construction failed)
    - Whether it was the winner
    
    This enables:
    - Re-evaluation with hints (exclude specific candidates)
    - Debugging (see why each candidate won/lost)
    - UI support (show users alternatives)
    """

    # Legacy: Persisted relations discovered during classification
    # TODO: Migrate to candidates pattern
    part_image_pairs: List[Tuple[Any, Any]] = field(default_factory=list)

    def get_label(self, element: Element) -> str | None:
        """Get the label for an element from this classification result.

        Args:
            element: The element to get the label for

        Returns:
            The label string if found, None otherwise
        """
        # Search through all candidates to find the winning label for this element
        for label, label_candidates in self.candidates.items():
            for candidate in label_candidates:
                if candidate.source_element is element and candidate.is_winner:
                    return label
        return None

    def get_elements_by_label(self, label: str) -> List[Element]:
        """Get all elements with the given label.

        Args:
            label: The label to search for

        Returns:
            List of elements with that label
        """
        label_candidates = self.candidates.get(label, [])
        return [c.source_element for c in label_candidates if c.is_winner]

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
            Dictionary mapping elements to score objects for that label
        """
        label_candidates = self.candidates.get(label, [])
        return {c.source_element: c.score_details for c in label_candidates}

    def has_label(self, label: str) -> bool:
        """Check if any elements have been assigned the given label.

        Args:
            label: The label to check for

        Returns:
            True if at least one element has this label, False otherwise
        """
        label_candidates = self.candidates.get(label, [])
        return any(c.is_winner for c in label_candidates)

    def get_best_candidate(self, label: str) -> "Optional[Candidate]":
        """Get the winning candidate for a label.

        Args:
            label: The label to get the best candidate for

        Returns:
            The candidate with the highest score that successfully constructed,
            or None if no valid candidates exist
        """
        label_candidates = self.candidates.get(label, [])
        valid = [c for c in label_candidates if c.constructed is not None]
        return max(valid, key=lambda c: c.score) if valid else None

    def get_alternative_candidates(
        self, label: str, exclude_winner: bool = True
    ) -> List[Candidate]:
        """Get alternative candidates for a label (for UI/re-evaluation).

        Args:
            label: The label to get alternatives for
            exclude_winner: If True, exclude the winning candidate

        Returns:
            List of candidates sorted by score (highest first)
        """
        label_candidates = self.candidates.get(label, [])
        if exclude_winner:
            winner_elems = self.get_elements_by_label(label)
            if winner_elems:
                winner_id = id(winner_elems[0])
                label_candidates = [
                    c for c in label_candidates if id(c.source_element) != winner_id
                ]
        return sorted(label_candidates, key=lambda c: c.score, reverse=True)


@dataclass
class ClassificationHints:
    """Hints to guide the classification process."""

    pass
