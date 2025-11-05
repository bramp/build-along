"""
Data classes for the classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union

from dataclass_wizard import JSONPyWizard

from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_elements import Element

if TYPE_CHECKING:
    from build_a_long.pdf_extract.extractor.extractor import PageData
    from build_a_long.pdf_extract.extractor.lego_page_elements import LegoPageElement


# Score key can be either a single Element or a tuple of Elements (for pairings)
ScoreKey = Union[Element, tuple[Element, ...]]


@dataclass
class RemovalReason(JSONPyWizard):
    """Tracks why an element was removed during classification."""

    reason_type: str
    """Type of removal: 'child_bbox' or 'similar_bbox'"""

    target_element: Element
    """The element that caused this removal"""


@dataclass
class Candidate(JSONPyWizard):
    """A candidate element with its score and constructed LegoElement.

    Represents a single element that was considered for a particular label,
    including its score, the constructed LegoPageElement (if successful),
    and information about why it succeeded or failed.

    This enables:
    - Re-evaluation with hints (exclude specific candidates)
    - Debugging (see all candidates and why they won/lost)
    - UI support (show users alternatives)
    """

    bbox: BBox
    """The bounding box for this candidate (from source_element or constructed)"""

    label: str
    """The label this candidate would have (e.g., 'page_number')"""

    # TODO Maybe score is redudant with score_details?
    score: float
    """Combined score (0.0-1.0)"""

    score_details: Any
    """The detailed score object (e.g., _PageNumberScore)"""

    constructed: LegoPageElement | None
    """The constructed LegoElement if parsing succeeded, None if failed"""

    source_element: Element | None = None
    """The raw element that was scored (None for synthetic elements like Step)"""

    failure_reason: str | None = None
    """Why construction failed, if it did"""

    is_winner: bool = False
    """Whether this candidate was selected as the winner.
    
    This field is set by mark_winner() and is used for:
    - Querying winners (get_label, get_elements_by_label, has_label)
    - Synthetic candidates (which have no source_element and can't be tracked in _element_winners)
    - JSON serialization and golden file comparisons
    
    Note: For candidates with source_element, this is redundant with _element_winners,
    but provides convenient access and handles synthetic candidates.
    """


@dataclass
class ClassifierConfig(JSONPyWizard):
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
class ClassificationResult(JSONPyWizard):
    """Represents the outcome of a single classification run.

    This class encapsulates the results of element classification, including
    labels, scores, and removal information. The candidates field is now the
    primary source of truth for classification results, containing all scored
    elements, their constructed LegoPageElements, and winner information.

    ClassificationResult is passed through the classifier pipeline, with each
    classifier adding its candidates and marking winners. This allows later
    classifiers to query the current state and make decisions based on earlier
    results.

    External code should use the accessor methods rather than accessing internal
    fields directly to maintain encapsulation.
    """

    page_data: PageData
    """The original page data being classified"""

    _warnings: list[str] = field(default_factory=list)

    _removal_reasons: dict[int, RemovalReason] = field(default_factory=dict)
    """Maps element IDs (element.id, not id(element)) to the reason they were removed.
    
    Keys are element IDs (int) instead of Element objects to ensure JSON serializability
    and consistency with _constructed_elements.
    """

    _constructed_elements: dict[int, LegoPageElement] = field(default_factory=dict)
    """Maps source element IDs to their constructed LegoPageElements.
    
    Only contains elements that were successfully labeled and constructed.
    The builder should use these pre-constructed elements rather than
    re-parsing the source elements.
    
    Keys are element IDs (int) instead of Element objects to ensure JSON serializability.
    """

    _candidates: dict[str, list[Candidate]] = field(default_factory=dict)
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

    _element_winners: dict[int, tuple[str, Candidate]] = field(default_factory=dict)
    """Maps element IDs to their winning (label, candidate) tuple.
    
    Ensures each element has at most one winning candidate across all labels.
    Keys are element IDs (int) for JSON serializability.
    """

    # Legacy: Persisted relations discovered during classification
    # TODO: Migrate to candidates pattern
    part_image_pairs: list[tuple[Any, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate PageData elements have unique IDs (if present).

        Elements may have None IDs, but elements with IDs must have unique IDs.
        Note: Only elements with IDs can be tracked in _constructed_elements and
        _removal_reasons (which require element.id as keys for JSON serializability).
        """
        # Validate unique IDs (ignoring None values)
        element_ids = [e.id for e in self.page_data.elements if e.id is not None]
        if len(element_ids) != len(set(element_ids)):
            duplicates = [id_ for id_ in element_ids if element_ids.count(id_) > 1]
            raise ValueError(
                f"PageData elements must have unique IDs. Found duplicates: {set(duplicates)}"
            )

    def _validate_element_in_page_data(
        self, element: Element | None, param_name: str = "element"
    ) -> None:
        """Validate that an element is in PageData.

        Args:
            element: The element to validate (None is allowed and skips validation)
            param_name: Name of the parameter being validated (for error messages)

        Raises:
            ValueError: If element is not None and not in PageData.elements
        """
        if element is not None and element not in self.page_data.elements:
            raise ValueError(
                f"{param_name} must be in PageData.elements. Element: {element}"
            )

    @property
    def elements(self) -> list[Element]:
        """Get the elements from the page data.

        Returns:
            List of elements from the page data
        """
        return self.page_data.elements

    def add_warning(self, warning: str) -> None:
        """Add a warning message to the classification result.

        Args:
            warning: The warning message to add
        """
        self._warnings.append(warning)

    def get_warnings(self) -> list[str]:
        """Get all warnings generated during classification.

        Returns:
            List of warning messages
        """
        return self._warnings.copy()

    def get_constructed_element(self, element: Element) -> LegoPageElement | None:
        """Get the constructed LegoPageElement for a source element.

        Args:
            element: The source element

        Returns:
            The constructed LegoPageElement if it exists, None otherwise
        """
        return self._constructed_elements.get(element.id)

    # TODO maybe add a parameter to fitler out winners/non-winners
    def get_candidates(self, label: str) -> list[Candidate]:
        """Get all candidates for a specific label.

        Args:
            label: The label to get candidates for

        Returns:
            List of candidates for that label (returns copy to prevent external modification)
        """
        return self._candidates.get(label, []).copy()

    def get_all_candidates(self) -> dict[str, list[Candidate]]:
        """Get all candidates across all labels.

        Returns:
            Dictionary mapping labels to their candidates (returns deep copy)
        """
        return {label: cands.copy() for label, cands in self._candidates.items()}

    def add_candidate(self, label: str, candidate: Candidate) -> None:
        """Add a single candidate for a specific label.

        Args:
            label: The label this candidate is for
            candidate: The candidate to add

        Raises:
            ValueError: If candidate has a source_element that is not in PageData
        """
        self._validate_element_in_page_data(
            candidate.source_element, "candidate.source_element"
        )

        if label not in self._candidates:
            self._candidates[label] = []
        self._candidates[label].append(candidate)

    def mark_winner(
        self,
        candidate: Candidate,
        constructed: LegoPageElement,
    ) -> None:
        """Mark a candidate as the winner and update tracking dicts.

        Args:
            candidate: The candidate to mark as winner
            constructed: The constructed LegoPageElement

        Raises:
            ValueError: If candidate has a source_element that is not in PageData
            ValueError: If this element already has a winner candidate
        """
        self._validate_element_in_page_data(
            candidate.source_element, "candidate.source_element"
        )

        # Check if this element already has a winner
        if candidate.source_element is not None:
            element_id = candidate.source_element.id
            if element_id in self._element_winners:
                existing_label, existing_candidate = self._element_winners[element_id]
                raise ValueError(
                    f"Element {element_id} already has a winner candidate for label "
                    f"'{existing_label}'. Cannot mark as winner for label '{candidate.label}'. "
                    f"Each element can have at most one winner candidate."
                )

        candidate.is_winner = True
        # Store the constructed element for this source element
        if candidate.source_element is not None:
            self._constructed_elements[candidate.source_element.id] = constructed
            self._element_winners[candidate.source_element.id] = (
                candidate.label,
                candidate,
            )

    def mark_removed(self, element: Element, reason: RemovalReason) -> None:
        """Mark an element as removed with the given reason.

        Args:
            element: The element to mark as removed
            reason: The reason for removal

        Raises:
            ValueError: If element is not in PageData
        """
        self._validate_element_in_page_data(element, "element")
        self._removal_reasons[element.id] = reason

    # TODO Consider removing this method.
    def get_labeled_elements(self) -> dict[Element, str]:
        """Get a dictionary of all labeled elements.

        Returns:
            Dictionary mapping elements to their labels (excludes synthetic candidates)
        """
        labeled: dict[Element, str] = {}
        for label, label_candidates in self._candidates.items():
            for candidate in label_candidates:
                if candidate.is_winner and candidate.source_element is not None:
                    labeled[candidate.source_element] = label
        return labeled

    def get_label(self, element: Element) -> str | None:
        """Get the label for an element from this classification result.

        Args:
            element: The element to get the label for

        Returns:
            The label string if found, None otherwise
        """
        # Search through all candidates to find the winning label for this element
        for label, label_candidates in self._candidates.items():
            for candidate in label_candidates:
                if candidate.source_element is element and candidate.is_winner:
                    return label
        return None

    def get_elements_by_label(self, label: str) -> list[Element]:
        """Get all elements with the given label.

        Args:
            label: The label to search for

        Returns:
            List of elements with that label (excludes synthetic candidates without source_element)
        """
        label_candidates = self._candidates.get(label, [])
        return [
            c.source_element
            for c in label_candidates
            if c.is_winner and c.source_element is not None
        ]

    def is_removed(self, element: Element) -> bool:
        """Check if an element has been marked for removal.

        Args:
            element: The element to check

        Returns:
            True if the element is marked for removal, False otherwise
        """
        return element.id in self._removal_reasons

    def get_removal_reason(self, element: Element) -> RemovalReason | None:
        """Get the reason why an element was removed.

        Args:
            element: The element to get the removal reason for

        Returns:
            The RemovalReason if the element was removed, None otherwise
        """
        return self._removal_reasons.get(element.id)

    def get_scores_for_label(self, label: str) -> dict[ScoreKey, Any]:
        """Get all scores for a specific label.

        Args:
            label: The label to get scores for

        Returns:
            Dictionary mapping elements to score objects for that label
            (excludes synthetic candidates without source_element)
        """
        label_candidates = self._candidates.get(label, [])
        return {
            c.source_element: c.score_details
            for c in label_candidates
            if c.source_element is not None
        }

    def has_label(self, label: str) -> bool:
        """Check if any elements have been assigned the given label.

        Args:
            label: The label to check for

        Returns:
            True if at least one element has this label, False otherwise
        """
        label_candidates = self._candidates.get(label, [])
        return any(c.is_winner for c in label_candidates)

    def get_best_candidate(self, label: str) -> Candidate | None:
        """Get the winning candidate for a label.

        Args:
            label: The label to get the best candidate for

        Returns:
            The candidate with the highest score that successfully constructed,
            or None if no valid candidates exist
        """
        label_candidates = self._candidates.get(label, [])
        valid = [c for c in label_candidates if c.constructed is not None]
        return max(valid, key=lambda c: c.score) if valid else None

    def get_alternative_candidates(
        self, label: str, exclude_winner: bool = True
    ) -> list[Candidate]:
        """Get alternative candidates for a label (for UI/re-evaluation).

        Args:
            label: The label to get alternatives for
            exclude_winner: If True, exclude the winning candidate

        Returns:
            List of candidates sorted by score (highest first)
        """
        label_candidates = self._candidates.get(label, [])
        if exclude_winner:
            winner_elems = self.get_elements_by_label(label)
            if winner_elems:
                winner_id = id(winner_elems[0])
                label_candidates = [
                    c for c in label_candidates if id(c.source_element) != winner_id
                ]
        return sorted(label_candidates, key=lambda c: c.score, reverse=True)


@dataclass
class ClassificationHints(JSONPyWizard):
    """Hints to guide the classification process."""

    pass
