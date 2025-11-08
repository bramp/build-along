"""
Data classes for the classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dataclass_wizard import JSONPyWizard

from build_a_long.pdf_extract.classifier.font_size_hints import FontSizeHints
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.extractor import PageData
from build_a_long.pdf_extract.extractor.lego_page_elements import LegoPageElement
from build_a_long.pdf_extract.extractor.page_blocks import Block

if TYPE_CHECKING:
    from build_a_long.pdf_extract.classifier.text_histogram import TextHistogram

# Score key can be either a single Block or a tuple of Blocks (for pairings)
ScoreKey = Block | tuple[Block, ...]


# TODO Make this JSON serializable
@dataclass
class BatchClassificationResult(JSONPyWizard):
    """Results from classifying multiple pages together.

    This class holds both the per-page classification results and the
    global text histogram computed across all pages.
    """

    results: list[ClassificationResult]
    """Per-page classification results, one for each input page"""

    histogram: TextHistogram
    """Global text histogram computed across all pages"""


@dataclass
class RemovalReason(JSONPyWizard):
    """Tracks why a block was removed during classification."""

    reason_type: str
    """Type of removal: 'child_bbox' or 'similar_bbox'"""

    target_block: Block
    """The block that caused this removal"""


@dataclass
class Candidate(JSONPyWizard):
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

    font_size_hints: FontSizeHints | None = None
    """Font size hints derived from analyzing all pages"""

    def __post_init__(self) -> None:
        for key, weight in self.__dict__.items():
            if key == "font_size_hints":
                continue
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
    """Maps block IDs (block.id, not id(block)) to the reason they were removed.
    
    Keys are block IDs (int) instead of Block objects to ensure JSON serializability
    and consistency with _constructed_elements.
    """

    _constructed_elements: dict[int, LegoPageElement] = field(default_factory=dict)
    """Maps source block IDs to their constructed LegoPageElements.
    
    Only contains elements that were successfully labeled and constructed.
    The builder should use these pre-constructed elements rather than
    re-parsing the source blocks.
    
    Keys are block IDs (int) instead of Block objects to ensure JSON serializability.
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

    _block_winners: dict[int, tuple[str, Candidate]] = field(default_factory=dict)
    """Maps block IDs to their winning (label, candidate) tuple.
    
    Ensures each block has at most one winning candidate across all labels.
    Keys are block IDs (int) for JSON serializability.
    """

    def __post_init__(self) -> None:
        """Validate PageData blocks have unique IDs (if present).

        Blocks may have None IDs, but blocks with IDs must have unique IDs.
        Note: Only blocks with IDs can be tracked in _constructed_elements and
        _removal_reasons (which require block.id as keys for JSON serializability).
        """
        # Validate unique IDs (ignoring None values)
        block_ids = [e.id for e in self.page_data.blocks if e.id is not None]
        if len(block_ids) != len(set(block_ids)):
            duplicates = [id_ for id_ in block_ids if block_ids.count(id_) > 1]
            raise ValueError(
                f"PageData blocks must have unique IDs. "
                f"Found duplicates: {set(duplicates)}"
            )

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

    def get_constructed_element(self, block: Block) -> LegoPageElement | None:
        """Get the constructed LegoPageElement for a source block.

        Args:
            block: The source block

        Returns:
            The constructed LegoPageElement if it exists, None otherwise
        """
        return self._constructed_elements.get(block.id)

    # TODO maybe add a parameter to fitler out winners/non-winners
    def get_candidates(self, label: str) -> list[Candidate]:
        """Get all candidates for a specific label.

        Args:
            label: The label to get candidates for

        Returns:
            List of candidates for that label (returns copy to prevent
            external modification)
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
            ValueError: If candidate has a source_block that is not in PageData
        """
        self._validate_block_in_page_data(
            candidate.source_block, "candidate.source_block"
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
            ValueError: If candidate has a source_block that is not in PageData
            ValueError: If this block already has a winner candidate
        """
        self._validate_block_in_page_data(
            candidate.source_block, "candidate.source_block"
        )

        # Check if this block already has a winner
        if candidate.source_block is not None:
            block_id = candidate.source_block.id
            if block_id in self._block_winners:
                existing_label, existing_candidate = self._block_winners[block_id]
                raise ValueError(
                    f"Block {block_id} already has a winner candidate for "
                    f"label '{existing_label}'. Cannot mark as winner for "
                    f"label '{candidate.label}'. Each block can have at most "
                    f"one winner candidate."
                )

        candidate.is_winner = True
        # Store the constructed element for this source element
        if candidate.source_block is not None:
            self._constructed_elements[candidate.source_block.id] = constructed
            self._block_winners[candidate.source_block.id] = (
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
        self._removal_reasons[block.id] = reason

    # TODO Consider removing this method.
    def get_labeled_blocks(self) -> dict[Block, str]:
        """Get a dictionary of all labeled blocks.

        Returns:
            Dictionary mapping blocks to their labels (excludes synthetic candidates)
        """
        labeled: dict[Block, str] = {}
        for label, label_candidates in self._candidates.items():
            for candidate in label_candidates:
                if candidate.is_winner and candidate.source_block is not None:
                    labeled[candidate.source_block] = label
        return labeled

    def get_label(self, block: Block) -> str | None:
        """Get the label for a block from this classification result.

        Args:
            block: The block to get the label for

        Returns:
            The label string if found, None otherwise
        """
        # Search through all candidates to find the winning label for this block
        for label, label_candidates in self._candidates.items():
            for candidate in label_candidates:
                if candidate.source_block is block and candidate.is_winner:
                    return label
        return None

    def get_blocks_by_label(self, label: str) -> list[Block]:
        """Get all blocks with the given label.

        Args:
            label: The label to search for

        Returns:
            List of blocks with that label. For constructed blocks (e.g., Part),
            returns the constructed object; for regular blocks, returns source_block.
        """
        label_candidates = self._candidates.get(label, [])
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
        return block.id in self._removal_reasons

    def get_removal_reason(self, block: Block) -> RemovalReason | None:
        """Get the reason why a block was removed.

        Args:
            block: The block to get the removal reason for

        Returns:
            The RemovalReason if the block was removed, None otherwise
        """
        return self._removal_reasons.get(block.id)

    def get_scores_for_label(self, label: str) -> dict[ScoreKey, Any]:
        """Get all scores for a specific label.

        Args:
            label: The label to get scores for

        Returns:
            Dictionary mapping elements to score objects for that label
            (excludes synthetic candidates without source_block)
        """
        label_candidates = self._candidates.get(label, [])
        return {
            c.source_block: c.score_details
            for c in label_candidates
            if c.source_block is not None
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
            winner_blocks = self.get_blocks_by_label(label)
            if winner_blocks:
                winner_id = id(winner_blocks[0])
                label_candidates = [
                    c for c in label_candidates if id(c.source_block) != winner_id
                ]
        return sorted(label_candidates, key=lambda c: c.score, reverse=True)

    def get_part_image_pairs(self) -> list[tuple[Block, Block]]:
        """Get part_count and part_image element pairs from winning candidates.

        This derives the pairs from the part_image candidates' score_details,
        which contain the relationship between part_count text and image elements.

        Returns:
            List of (part_count, image) tuples for all winning part_image candidates
        """
        pairs: list[tuple[Block, Block]] = []
        for candidate in self.get_candidates("part_image"):
            if candidate.is_winner and candidate.score_details:
                # score_details is a _PartImageScore with part_count and image fields
                score = candidate.score_details
                if hasattr(score, "part_count") and hasattr(score, "image"):
                    pairs.append((score.part_count, score.image))
        return pairs
