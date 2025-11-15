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
from build_a_long.pdf_extract.extractor.lego_page_elements import LegoPageElement, Page
from build_a_long.pdf_extract.extractor.page_blocks import Block

# Score key can be either a single Block or a tuple of Blocks (for pairings)
ScoreKey = Block | tuple[Block, ...]

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


class RemovalReason(BaseModel):
    """Tracks why a block was removed during classification."""

    reason_type: str
    """Type of removal: 'duplicate_bbox', 'child_bbox', or 'similar_bbox'"""

    # TODO Should this be updated to the Candidate that caused the removal?
    target_block: Block
    """The block that caused this removal"""


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

    constructed: LegoPageElement | None
    """The constructed LegoElement if parsing succeeded, None if failed"""

    source_block: Block | None = None
    """The raw element that was scored (None for synthetic elements like Step)"""

    failure_reason: str | None = None
    """Why construction failed, if it did"""

    is_winner: bool = False
    """Whether this candidate was selected as the winner.
    
    This field is set by mark_winner() and is used for:
    - Querying winners (get_label, get_blocks_by_label, has_label)
    - Synthetic candidates (which have no source_block and can't be tracked
      in _block_winners)
    - JSON serialization and golden file comparisons
    
    Note: For candidates with source_block, this is redundant with
    _block_winners, but provides convenient access and handles synthetic
    candidates.
    """


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
    - Whether it was the winner
    
    This enables:
    - Re-evaluation with hints (exclude specific candidates)
    - Debugging (see why each candidate won/lost)
    - UI support (show users alternatives)
    
    Public for serialization. Prefer using get_* accessor methods.
    """

    block_winners: dict[int, tuple[str, Candidate]] = Field(default_factory=dict)
    """Maps block IDs to their winning (label, candidate) tuple.
    
    Ensures each block has at most one winning candidate across all labels.
    Keys are block IDs (int) for JSON serializability.
    
    Public for serialization. Prefer using get_label() and related methods.
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
        self, block: Block | None, param_name: str = "block"
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
    def blocks(self) -> list[Block]:
        """Get the blocks from the page data.

        Returns:
            List of blocks from the page data
        """
        return self.page_data.blocks

    @property
    def page(self) -> Page | None:
        """Returns the Page object built from this classification result."""
        pages = self.get_candidates("page")
        # Filter for winner candidates
        winner_pages = [c for c in pages if c.is_winner and c.constructed is not None]
        assert len(winner_pages) <= 1, (
            "There should be no more than one winning Page candidate."
        )
        return cast(Page, winner_pages[0].constructed) if winner_pages else None

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

    def get_candidates(self, label: str) -> list[Candidate]:
        """Get all candidates for a specific label.

        Args:
            label: The label to get candidates for

        Returns:
            List of candidates for that label (returns copy to prevent
            external modification)
        """
        return self.candidates.get(label, []).copy()

    def get_winners[T: LegoPageElement](
        self, label: str, element_type: type[T]
    ) -> list[T]:
        """Get winning candidates for a specific label with type safety.

        This is a convenience method that filters candidates to only those that:
        - Are marked as winners (is_winner=True)
        - Have been successfully constructed (constructed is not None)
        - Match the specified element type

        Args:
            label: The label to get winners for (e.g., "page_number", "step")
            element_type: The type of element to filter for (e.g., PageNumber, Step)

        Returns:
            List of constructed elements of the specified type

        Raises:
            AssertionError: If a winning candidate has None constructed (invalid state)
            AssertionError: If element_type doesn't match the actual constructed type
        """
        winners = []
        for candidate in self.get_candidates(label):
            if not candidate.is_winner:
                continue

            # Invariant check: winners must have constructed elements
            assert candidate.constructed is not None, (
                f"Winner candidate for label '{label}' has None "
                f"constructed. This is an invalid state - winners must "
                f"have constructed elements."
            )

            # Type safety check: verify constructed matches requested type
            assert isinstance(candidate.constructed, element_type), (
                f"Type mismatch for label '{label}': requested "
                f"{element_type.__name__} but got "
                f"{type(candidate.constructed).__name__}. "
                f"This indicates a programming error in the caller."
            )

            if candidate.constructed is not None and isinstance(
                candidate.constructed, element_type
            ):
                winners.append(cast(T, candidate.constructed))

        return winners

    def get_all_candidates(self) -> dict[str, list[Candidate]]:
        """Get all candidates across all labels.

        Returns:
            Dictionary mapping labels to their candidates (returns deep copy)
        """
        return {label: cands.copy() for label, cands in self.candidates.items()}

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
            ValueError: If candidate has a source_block that is not in PageData
            ValueError: If this block already has a winner candidate
        """
        self._validate_block_in_page_data(
            candidate.source_block, "candidate.source_block"
        )

        # Check if this block already has a winner
        if candidate.source_block is not None:
            block_id = candidate.source_block.id
            if block_id in self.block_winners:
                existing_label, existing_candidate = self.block_winners[block_id]
                raise ValueError(
                    f"Block {block_id} already has a winner candidate for "
                    f"label '{existing_label}'. Cannot mark as winner for "
                    f"label '{candidate.label}'. Each block can have at most "
                    f"one winner candidate."
                )

        candidate.is_winner = True
        # Track the winner for this block
        if candidate.source_block is not None:
            self.block_winners[candidate.source_block.id] = (
                candidate.label,
                candidate,
            )

    def mark_removed(self, block: Block, reason: RemovalReason) -> None:
        """Mark a block as removed with the given reason.

        Args:
            block: The block to mark as removed
            reason: The reason for removal

        Raises:
            ValueError: If block is not in PageData
        """
        self._validate_block_in_page_data(block, "block")
        self.removal_reasons[block.id] = reason

    def get_label(self, block: Block) -> str | None:
        """Get the label for a block from this classification result.

        Args:
            block: The block to get the label for

        Returns:
            The label string if found, None otherwise
        """
        # Use O(1) lookup via block_winners
        if block.id in self.block_winners:
            label, _ = self.block_winners[block.id]
            return label
        return None

    def get_winner_candidate(self, block: Block) -> Candidate | None:
        """Get the winning candidate for a block.

        Provides O(1) lookup of the winner candidate and its constructed element.

        Args:
            block: The block to get the winner for

        Returns:
            The winning Candidate if found, None otherwise
        """
        if block.id in self.block_winners:
            _, candidate = self.block_winners[block.id]
            return candidate
        return None

    def get_blocks_by_label(self, label: str) -> list[Block]:
        """Get all blocks with the given label.

        Args:
            label: The label to search for

        Returns:
            List of blocks with that label. For constructed blocks (e.g., Part),
            returns the constructed object; for regular blocks, returns source_block.
        """
        label_candidates = self.candidates.get(label, [])
        blocks = []
        for c in label_candidates:
            if c.is_winner:
                # Prefer source_block, fall back to constructed for synthetic blocks
                if c.source_block is not None:
                    blocks.append(c.source_block)
                elif c.constructed is not None:
                    blocks.append(c.constructed)
        return blocks

    def is_removed(self, block: Block) -> bool:
        """Check if a block has been marked for removal.

        Args:
            block: The block to check

        Returns:
            True if the block is marked for removal, False otherwise
        """
        return block.id in self.removal_reasons

    def get_removal_reason(self, block: Block) -> RemovalReason | None:
        """Get the reason why a block was removed.

        Args:
            block: The block to get the removal reason for

        Returns:
            The RemovalReason if the block was removed, None otherwise
        """
        return self.removal_reasons.get(block.id)

    def has_label(self, label: str) -> bool:
        """Check if any elements have been assigned the given label.

        Args:
            label: The label to check for

        Returns:
            True if at least one element has this label, False otherwise
        """
        label_candidates = self.candidates.get(label, [])
        return any(c.is_winner for c in label_candidates)
