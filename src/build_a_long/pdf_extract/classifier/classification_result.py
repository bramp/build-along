"""
Data classes for the classifier.
"""

from __future__ import annotations

from typing import Annotated, Any, cast

from annotated_types import Ge, Le
from pydantic import BaseModel, Field, model_validator

from build_a_long.pdf_extract.classifier.font_size_hints import FontSizeHints
from build_a_long.pdf_extract.classifier.text_histogram import TextHistogram
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.extractor import PageData
from build_a_long.pdf_extract.extractor.lego_page_elements import LegoPageElements, Page
from build_a_long.pdf_extract.extractor.page_blocks import Blocks

# Score key can be either a single Block or a tuple of Blocks (for pairings)
ScoreKey = Blocks | tuple[Blocks, ...]

# Weight value constrained to [0.0, 1.0] range
Weight = Annotated[float, Ge(0), Le(1)]


# TODO Make this JSON serializable
class BatchClassificationResult(BaseModel):
    """Results from classifying multiple pages together.

    This class holds both the per-page classification results and the
    global text histogram computed across all pages.
    """

    results: list[ClassificationResult]
    """Per-page classification results, one for each input page"""

    histogram: TextHistogram
    """Global text histogram computed across all pages"""


# TODO Change this to be frozen
class RemovalReason(BaseModel):
    """Tracks why a block was removed during classification."""

    reason_type: str
    """Type of removal: 'duplicate_bbox', 'child_bbox', or 'similar_bbox'"""

    # TODO Should this be updated to the Candidate that caused the removal?
    target_block: Blocks
    """The block that caused this removal"""


# TODO Change this to be frozen
class Candidate(BaseModel):
    """A candidate block with its score and constructed LegoElement.

    Represents a single block that was considered for a particular label,
    including its score, the constructed LegoPageElement (if successful),
    and information about why it succeeded or failed.

    This enables:
    - Re-evaluation with hints (exclude specific candidates)
    - Debugging (see all candidates and why they won/lost)
    - UI support (show users alternatives)
    """

    bbox: BBox
    """The bounding box for this candidate (from source_block or constructed)"""

    label: str
    """The label this candidate would have (e.g., 'page_number')"""

    # TODO Maybe score is redudant with score_details?
    score: float
    """Combined score (0.0-1.0)"""

    score_details: Any
    """The detailed score object (e.g., _PageNumberScore)"""

    constructed: LegoPageElements | None
    """The constructed LegoElement if parsing succeeded, None if failed"""

    source_block: Blocks | None = None
    """The raw element that was scored (None for synthetic elements like Step)"""

    failure_reason: str | None = None
    """Why construction failed, if it did"""


class ClassifierConfig(BaseModel):
    """Configuration for the classifier."""

    # TODO Consistenctly use this, or give it a name more descriptive of where
    # it's used
    min_confidence_threshold: Weight = 0.6

    page_number_text_weight: Weight = 0.7
    page_number_position_weight: Weight = 0.3
    page_number_position_scale: float = 50.0
    page_number_page_value_weight: Weight = 1.0
    page_number_font_size_weight: Weight = 0.1

    step_number_text_weight: Weight = 0.7
    step_number_font_size_weight: Weight = 0.3

    part_count_text_weight: Weight = 0.7
    part_count_font_size_weight: Weight = 0.3

    parts_list_max_area_ratio: float = Field(default=0.75, ge=0.0, le=1.0)
    """Maximum ratio of page area a parts list can occupy (0.0-1.0).
    
    Drawings larger than this fraction of the page are rejected as they're
    likely the entire page background rather than actual parts lists.
    Default is 0.75 (75% of page area).
    """

    font_size_hints: FontSizeHints = Field(default_factory=FontSizeHints.empty)
    """Font size hints derived from analyzing all pages"""


class ClassificationResult(BaseModel):
    """Result of classifying a single page.

    This class stores both the results and intermediate artifacts for a page
    classification. It provides structured access to:
    - Labels assigned to blocks
    - LegoPageElements constructed from blocks
    - Removal reasons for filtered blocks
    - All candidates considered (including rejected ones)

    The use of dictionaries keyed by block IDs (int) instead of Block objects
    ensures JSON serializability and consistent equality semantics.

    # TODO: Consider refactoring to separate DAO (Data Access Object) representation
    # from the business logic. The public fields below are used for serialization
    # but external code should prefer using the accessor methods to maintain
    # encapsulation and allow future refactoring.

    External code should use the accessor methods rather than accessing these
    fields directly to maintain encapsulation.
    """

    page_data: PageData
    """The original page data being classified"""

    warnings: list[str] = Field(default_factory=list)
    """Warning messages generated during classification.
    
    Public for serialization. Prefer using add_warning() and get_warnings() methods.
    """

    removal_reasons: dict[int, RemovalReason] = Field(default_factory=dict)
    """Maps block IDs (block.id, not id(block)) to the reason they were removed.
    
    Keys are block IDs (int) instead of Block objects to ensure JSON serializability
    and consistency with constructed_elements.
    
    Public for serialization. Prefer using accessor methods.
    """

    candidates: dict[str, list[Candidate]] = Field(default_factory=dict)
    """Maps label names to lists of all candidates considered for that label.
    
    Each candidate includes:
    - The source element
    - Its score and score details
    - The constructed LegoPageElement (if successful)
    - Failure reason (if construction failed)
    
    This enables:
    - Re-evaluation with hints (exclude specific candidates)
    - Debugging (see why each candidate won/lost)
    - UI support (show users alternatives)
    
    Public for serialization. Prefer using get_* accessor methods.
    """

    @model_validator(mode="after")
    def validate_unique_block_ids(self) -> ClassificationResult:
        """Validate that all block IDs in page_data are unique.

        Blocks must have unique IDs.
        Note: Blocks with IDs can be tracked in removal_reasons
        (which require block.id as keys for JSON serializability).
        """
        # Validate unique IDs
        block_ids = [b.id for b in self.page_data.blocks]
        if len(block_ids) != len(set(block_ids)):
            duplicates = [id_ for id_ in block_ids if block_ids.count(id_) > 1]
            raise ValueError(
                f"PageData blocks must have unique IDs. "
                f"Found duplicates: {set(duplicates)}"
            )
        return self

    def _validate_block_in_page_data(
        self, block: Blocks | None, param_name: str = "block"
    ) -> None:
        """Validate that a block is in PageData.

        Args:
            block: The block to validate (None is allowed and skips validation)
            param_name: Name of the parameter being validated (for error messages)

        Raises:
            ValueError: If block is not None and not in PageData.blocks
        """
        if block is not None and block not in self.page_data.blocks:
            raise ValueError(f"{param_name} must be in PageData.blocks. Block: {block}")

    @property
    def blocks(self) -> list[Blocks]:
        """Get the blocks from the page data.

        Returns:
            List of blocks from the page data
        """
        return self.page_data.blocks

    @property
    def page(self) -> Page | None:
        """Returns the Page object built from this classification result."""
        pages = self.get_winners_by_score("page", Page, max_count=1)
        return pages[0] if pages else None

    def add_warning(self, warning: str) -> None:
        """Add a warning message to the classification result.

        Args:
            warning: The warning message to add
        """
        self.warnings.append(warning)

    def get_warnings(self) -> list[str]:
        """Get all warnings generated during classification.

        Returns:
            List of warning messages
        """
        return self.warnings.copy()

    # TODO Reconsider the methods below - some may be redundant.

    def get_candidates(self, label: str) -> list[Candidate]:
        """Get all candidates for a specific label.

        Args:
            label: The label to get candidates for

        Returns:
            List of candidates for that label (returns copy to prevent
            external modification)
        """
        return self.candidates.get(label, []).copy()

    def get_winners_by_score[T: LegoPageElements](
        self, label: str, element_type: type[T], max_count: int | None = None
    ) -> list[T]:
        """Get the best candidates for a specific label by score.

        Selects candidates by:
        - Successfully constructed (constructed is not None)
        - Highest score
        - Match the specified element type

        Args:
            label: The label to get winners for (e.g., "page_number", "step")
            element_type: The type of element to filter for (e.g., PageNumber)
            max_count: Maximum number of winners to return (None = all valid)

        Returns:
            List of constructed elements of the specified type, sorted by score
            (highest first)

        Raises:
            AssertionError: If element_type doesn't match the actual
                constructed type
        """
        # Get all candidates and filter for successful construction
        valid_candidates = [
            c for c in self.get_candidates(label) if c.constructed is not None
        ]

        # Validate types
        for candidate in valid_candidates:
            assert isinstance(candidate.constructed, element_type), (
                f"Type mismatch for label '{label}': requested "
                f"{element_type.__name__} but got "
                f"{type(candidate.constructed).__name__}. "
                f"This indicates a programming error in the caller."
            )

        # Sort by score (highest first)
        valid_candidates.sort(key=lambda c: c.score, reverse=True)

        # Apply max_count if specified
        if max_count is not None:
            valid_candidates = valid_candidates[:max_count]

        # Extract constructed elements
        return [cast(T, c.constructed) for c in valid_candidates]

    def get_all_candidates(self) -> dict[str, list[Candidate]]:
        """Get all candidates across all labels.

        Returns:
            Dictionary mapping labels to their candidates (returns copy to
            prevent external modification)
        """
        return {label: cands for label, cands in self.candidates.items()}

    def count_successful_candidates(self, label: str) -> int:
        """Count how many candidates were successfully constructed for a label.

        Test helper method that counts candidates where construction succeeded.

        Args:
            label: The label to count successful candidates for

        Returns:
            Count of successfully constructed candidates
        """
        return sum(1 for c in self.get_candidates(label) if c.constructed is not None)

    def get_all_candidates_for_block(self, block: Blocks) -> list[Candidate]:
        """Get all candidates for a block across all labels.

        Searches across all labels to find candidates that used the given block
        as their source. For finding a candidate with a specific label, use
        get_candidate_for_block() instead.

        Args:
            block: The block to find candidates for

        Returns:
            List of all candidates across all labels with this block as source_block
        """
        results = []
        for candidates in self.candidates.values():
            for candidate in candidates:
                if candidate.source_block is block:
                    results.append(candidate)
        return results

    def get_candidate_for_block(self, block: Blocks, label: str) -> Candidate | None:
        """Get the candidate for a specific block with a specific label.

        Helper method for testing - returns the single candidate for the given
        block and label combination. Returns None if no such candidate exists.

        Args:
            block: The block to find the candidate for
            label: The label to search within

        Returns:
            The candidate if found, None otherwise

        Raises:
            ValueError: If multiple candidates exist for this block/label pair
        """
        candidates = [c for c in self.get_candidates(label) if c.source_block is block]

        if len(candidates) == 0:
            return None

        if len(candidates) == 1:
            return candidates[0]

        raise ValueError(
            f"Multiple candidates found for block {block.id} "
            f"with label '{label}'. Expected at most one."
        )

    def get_label(self, block: Blocks) -> str | None:
        """Get the label for a block from its successfully constructed candidate.

        Returns the label of the first successfully constructed candidate for
        the given block, or None if no successfully constructed candidate exists.

        Args:
            block: The block to get the label for

        Returns:
            The label string if a successfully constructed candidate exists,
            None otherwise
        """
        for candidate in self.get_all_candidates_for_block(block):
            if candidate.constructed is not None:
                return candidate.label
        return None

    def add_candidate(self, label: str, candidate: Candidate) -> None:
        """Add a single candidate for a specific label.

        Args:
            label: The label this candidate is for
            candidate: The candidate to add

        Raises:
            ValueError: If candidate has a source_block that is not in PageData
        """
        self._validate_block_in_page_data(
            candidate.source_block, "candidate.source_block"
        )

        if label not in self.candidates:
            self.candidates[label] = []
        self.candidates[label].append(candidate)

    def mark_removed(self, block: Blocks, reason: RemovalReason) -> None:
        """Mark a block as removed with the given reason.

        Args:
            block: The block to mark as removed
            reason: The reason for removal

        Raises:
            ValueError: If block is not in PageData
        """
        self._validate_block_in_page_data(block, "block")
        self.removal_reasons[block.id] = reason

    def is_removed(self, block: Blocks) -> bool:
        """Check if a block has been marked for removal.

        Args:
            block: The block to check

        Returns:
            True if the block is marked for removal, False otherwise
        """
        return block.id in self.removal_reasons

    def get_removal_reason(self, block: Blocks) -> RemovalReason | None:
        """Get the reason why a block was removed.

        Args:
            block: The block to get the removal reason for

        Returns:
            The RemovalReason if the block was removed, None otherwise
        """
        return self.removal_reasons.get(block.id)
