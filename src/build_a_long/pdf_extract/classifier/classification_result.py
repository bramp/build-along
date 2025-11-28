"""
Data classes for the classifier.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, cast

from annotated_types import Ge, Le
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from build_a_long.pdf_extract.classifier.font_size_hints import FontSizeHints
from build_a_long.pdf_extract.classifier.pages.page_hint_collection import (
    PageHintCollection,
)
from build_a_long.pdf_extract.classifier.text_histogram import TextHistogram
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.extractor import PageData
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    LegoPageElements,
    Page,
)
from build_a_long.pdf_extract.extractor.page_blocks import Blocks

if TYPE_CHECKING:
    from build_a_long.pdf_extract.classifier.label_classifier import LabelClassifier

# Score key can be either a single Block or a tuple of Blocks (for pairings)
ScoreKey = Blocks | tuple[Blocks, ...]

# Weight value constrained to [0.0, 1.0] range
Weight = Annotated[float, Ge(0), Le(1)]


class Score(BaseModel):
    """Abstract base class for score_details objects.

    All score_details stored in Candidate objects must inherit from this class.
    This ensures a consistent interface for accessing the final score value.

    The score() method MUST return a value in the range [0.0, 1.0] where:
    - 0.0 indicates lowest confidence/worst match
    - 1.0 indicates highest confidence/best match

    Example implementations:
        class _PageNumberScore(Score):
            text_score: float
            position_score: float

            def score(self) -> Weight:
                # Returns normalized score in range [0.0, 1.0]
                return (self.text_score + self.position_score) / 2.0
    """

    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def score(self) -> Weight:
        """Return the final computed score value.

        Returns:
            Weight: A score value in the range [0.0, 1.0] where 0.0 is the
                lowest confidence and 1.0 is the highest confidence.
        """
        ...


@dataclass(frozen=True)
class _BuildSnapshot:
    """Snapshot of candidate and consumed block state for rollback.

    This is used to implement transactional semantics in build():
    if a classifier build fails, we can restore the state as if
    the build never started.
    """

    # Map candidate id -> (constructed value, failure_reason)
    candidate_states: dict[int, tuple[LegoPageElements | None, str | None]]
    # Set of consumed block IDs
    consumed_blocks: set[int]


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

    score_details: Score
    """The detailed score object inheriting from Score (e.g., _PageNumberScore)"""

    constructed: LegoPageElements | None = None
    """The constructed LegoElement if parsing succeeded, None if failed"""

    source_blocks: list[Blocks] = []
    """The raw elements that were scored (empty for synthetic elements like Step).
    
    Multiple source blocks indicate the candidate was derived from multiple inputs.
    For example, a PieceLength is derived from both a Text block (the number) and
    a Drawing block (the circle diagram).
    """

    failure_reason: str | None = None
    """Why construction failed, if it did"""

    @property
    def is_valid(self) -> bool:
        """Check if this candidate is valid (constructed and no failure).

        A valid candidate has been successfully constructed and has no failure reason.
        Use this to filter candidates when working with dependencies.

        Returns:
            True if candidate.constructed is not None and failure_reason is None
        """
        return self.constructed is not None and self.failure_reason is None


class ClassifierConfig(BaseModel):
    """Configuration for the classifier.

    Naming Conventions
    ------------------
    All classifier-specific settings should be prefixed with the label name:

    - `{label}_min_score`: Minimum score threshold. Candidates scoring below
      this value are not created (to reduce debug spam). Default: 0.5
    - `{label}_*_weight`: Weights for different scoring components
    - `{label}_*`: Other label-specific configuration

    Example: For a "page_number" label:
    - page_number_min_score
    - page_number_text_weight
    - page_number_position_weight
    """

    # TODO Consistenctly use this, or give it a name more descriptive of where
    # it's used
    min_confidence_threshold: Weight = 0.6

    # Page number classifier settings
    page_number_min_score: Weight = 0.5
    page_number_text_weight: Weight = 0.7
    page_number_position_weight: Weight = 0.3
    page_number_position_scale: float = 50.0
    page_number_page_value_weight: Weight = 1.0
    page_number_font_size_weight: Weight = 0.1

    # Step number classifier settings
    step_number_min_score: Weight = 0.5
    step_number_text_weight: Weight = 0.7
    step_number_font_size_weight: Weight = 0.3

    # Part count classifier settings
    part_count_min_score: Weight = 0.5
    part_count_text_weight: Weight = 0.7
    part_count_font_size_weight: Weight = 0.3

    # Part number classifier settings
    part_number_min_score: Weight = 0.5

    # Parts list classifier settings
    parts_list_min_score: Weight = 0.5
    parts_list_max_area_ratio: float = Field(default=0.75, ge=0.0, le=1.0)
    """Maximum ratio of page area a parts list can occupy (0.0-1.0).
    
    Drawings larger than this fraction of the page are rejected as they're
    likely the entire page background rather than actual parts lists.
    Default is 0.75 (75% of page area).
    """

    font_size_hints: FontSizeHints = Field(default_factory=FontSizeHints.empty)
    """Font size hints derived from analyzing all pages"""

    page_hints: PageHintCollection = Field(default_factory=PageHintCollection.empty)
    """Page type hints derived from analyzing all pages"""


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

    # TODO Do we need this field? Can we remove it?
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

    _classifiers: dict[str, LabelClassifier] = PrivateAttr(default_factory=dict)
    _consumed_blocks: set[int] = PrivateAttr(default_factory=set)

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

    def _register_classifier(self, label: str, classifier: LabelClassifier) -> None:
        """Register a classifier for a specific label.

        This is called automatically by LabelClassifier.score() and should not
        be called directly by external code.
        """
        self._classifiers[label] = classifier

    def build(self, candidate: Candidate) -> LegoPageElements:
        """Construct a candidate using the registered classifier.

        This is the entry point for top-down construction. If the build fails,
        all changes to candidate states and consumed blocks are automatically
        rolled back, ensuring transactional semantics.
        """
        if candidate.constructed:
            return candidate.constructed

        if candidate.failure_reason:
            raise ValueError(f"Candidate failed: {candidate.failure_reason}")

        # Check if any source block is already consumed
        # TODO Do we need the following? As _fail_conflicting_candidates should
        # be setting failure reasons already.
        for block in candidate.source_blocks:
            if block.id in self._consumed_blocks:
                # Find who consumed it (for better error message)
                # This is expensive but only happens on failure
                winner_label = "unknown"
                for _label, cat_candidates in self.candidates.items():
                    for c in cat_candidates:
                        if c.constructed and any(
                            b.id == block.id for b in c.source_blocks
                        ):
                            winner_label = _label
                            break

                failure_msg = f"Block {block.id} already consumed by '{winner_label}'"
                candidate.failure_reason = failure_msg
                raise ValueError(failure_msg)

        classifier = self._classifiers.get(candidate.label)
        if not classifier:
            raise ValueError(f"No classifier registered for label '{candidate.label}'")

        # Take snapshot before building for automatic rollback on failure
        snapshot = self._take_snapshot()

        try:
            element = classifier.build(candidate, self)
            candidate.constructed = element

            # Mark blocks as consumed
            for block in candidate.source_blocks:
                self._consumed_blocks.add(block.id)

            # Fail other candidates that use these blocks
            self._fail_conflicting_candidates(candidate)

            return element
        except Exception:
            # Rollback all changes made during this build
            self._restore_snapshot(snapshot)
            raise

    def _take_snapshot(self) -> _BuildSnapshot:
        """Take a snapshot of all candidate states and consumed blocks."""
        candidate_states = {}
        for candidates in self.candidates.values():
            for c in candidates:
                candidate_states[id(c)] = (c.constructed, c.failure_reason)

        return _BuildSnapshot(
            candidate_states=candidate_states,
            consumed_blocks=self._consumed_blocks.copy(),
        )

    def _restore_snapshot(self, snapshot: _BuildSnapshot) -> None:
        """Restore candidate states and consumed blocks from a snapshot."""
        # Restore candidate states
        for candidates in self.candidates.values():
            for c in candidates:
                cid = id(c)
                if cid in snapshot.candidate_states:
                    c.constructed, c.failure_reason = snapshot.candidate_states[cid]

        # Restore consumed blocks
        self._consumed_blocks = snapshot.consumed_blocks.copy()

    def _fail_conflicting_candidates(self, winner: Candidate) -> None:
        """Mark other candidates sharing blocks with winner as failed."""
        winner_block_ids = {b.id for b in winner.source_blocks}

        for _label, candidates in self.candidates.items():
            for candidate in candidates:
                if candidate is winner:
                    continue
                if candidate.failure_reason:
                    continue

                # Check for overlap
                for block in candidate.source_blocks:
                    if block.id in winner_block_ids:
                        candidate.failure_reason = (
                            f"Lost conflict to '{winner.label}' "
                            f"(score={winner.score:.3f})"
                        )
                        break

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
        page_candidates = self.get_scored_candidates("page", valid_only=True)
        if page_candidates:
            page = page_candidates[0].constructed
            assert isinstance(page, Page)
            return page
        return None

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

    def get_scored_candidates(
        self,
        label: str,
        min_score: float = 0.0,
        valid_only: bool = True,
        exclude_failed: bool = False,
    ) -> list[Candidate]:
        """Get candidates for a label that have been scored.

        **Use this method in score() when working with dependency classifiers.**

        This enforces the pattern of working with candidates (not constructed
        elements or raw blocks) when one classifier depends on another. The
        returned candidates are sorted by score (highest first).

        During score(), you should:
        1. Get parent candidates using this method
        2. Store references to parent candidates in your score_details
        3. In construct(), validate parent candidates before using their elements

        Example:
            # In PartsClassifier.score()
            part_count_candidates = result.get_scored_candidates("part_count")
            for pc_cand in part_count_candidates:
                # Store the CANDIDATE reference in score details
                score_details = _PartPairScore(
                    part_count_candidate=pc_cand,  # Not pc_cand.constructed!
                    image=img,
                )

            # Later in _construct_single()
            def _construct_single(self, candidate, result):
                pc_cand = candidate.score_details.part_count_candidate

                # Validate parent candidate is still valid
                if not pc_cand.is_valid:
                    raise ValueError(
                        f"Parent invalid: {pc_cand.failure_reason or 'not constructed'}"
                    )

                # Now safe to use the constructed element
                assert isinstance(pc_cand.constructed, PartCount)
                return Part(count=pc_cand.constructed, ...)

        Args:
            label: The label to get candidates for
            min_score: Optional minimum score threshold (default: 0.0)
            valid_only: If True (default), only return valid candidates
                (constructed and no failure). Set to False to get all scored
                candidates regardless of construction status.
            exclude_failed: If True, filter out candidates with failure_reason,
                even if valid_only is False. (default: False)

        Returns:
            List of scored candidates sorted by score (highest first).
            By default, only includes valid candidates (is_valid=True).
        """
        candidates = self.get_candidates(label)

        # Filter to candidates that have been scored
        scored = [c for c in candidates if c.score_details is not None]

        # Apply score threshold if specified
        if min_score > 0:
            scored = [c for c in scored if c.score >= min_score]

        # Filter to valid candidates if requested (default)
        if valid_only:
            scored = [c for c in scored if c.is_valid]
        elif exclude_failed:
            scored = [c for c in scored if c.failure_reason is None]

        # Sort by score descending
        # TODO add a tie breaker for determinism.
        scored.sort(key=lambda c: -c.score)

        return scored

    def get_winners_by_score[T: LegoPageElements](
        self, label: str, element_type: type[T], max_count: int | None = None
    ) -> list[T]:
        """Get the best candidates for a specific label by score.

        **DEPRECATED for use in score() methods.**

        This method returns constructed LegoPageElements, which encourages the
        anti-pattern of looking at constructed elements during the score() phase.

        - **In score()**: Use get_scored_candidates() instead to work with candidates
        - **In construct()**: It's OK to use this method when you need fully
          constructed dependency elements

        Prefer get_scored_candidates() in score() to maintain proper separation
        between the scoring and construction phases.

        Selects candidates by:
        - Successfully constructed (constructed is not None)
        - Match the specified element type
        - Sorted by score (highest first)

        Invariant: Each source block should have at most one successfully
        constructed candidate per label. This method validates that invariant.

        Args:
            label: The label to get winners for (e.g., "page_number", "step")
            element_type: The type of element to filter for (e.g., PageNumber)
            max_count: Maximum number of winners to return (None = all valid)

        Returns:
            List of constructed elements of the specified type, sorted by score
            (highest first)

        Raises:
            AssertionError: If element_type doesn't match the actual constructed type,
                or if multiple candidates exist for the same source block
        """
        # Get all candidates and filter for successful construction
        valid_candidates = [
            c for c in self.get_candidates(label) if c.constructed is not None
        ]

        # Validate that each source block has at most one candidate for this label
        # (candidates without source blocks are synthetic and can have duplicates)
        seen_blocks: set[int] = set()
        for candidate in valid_candidates:
            assert isinstance(candidate.constructed, element_type), (
                f"Type mismatch for label '{label}': requested "
                f"{element_type.__name__} but got "
                f"{type(candidate.constructed).__name__}. "
                f"This indicates a programming error in the caller."
            )

            for source_block in candidate.source_blocks:
                block_id = id(source_block)
                assert block_id not in seen_blocks, (
                    f"Multiple successfully constructed candidates found for "
                    f"label '{label}' with the same source block id:{block_id}. "
                    f"This indicates a programming error in the classifier. "
                    f"Source block: {source_block}"
                )
                seen_blocks.add(block_id)

        # Sort by score (highest first), then by source block ID for determinism
        # when scores are equal
        valid_candidates.sort(
            key=lambda c: (
                -c.score,  # Negative for descending order
                # TODO Fix this, so it's deterministic.
                c.source_blocks[0].id if c.source_blocks else 0,  # Tie-breaker
            )
        )

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
            List of all candidates across all labels with this block in source_blocks
        """
        results = []
        for candidates in self.candidates.values():
            for candidate in candidates:
                if block in candidate.source_blocks:
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
        candidates = [c for c in self.get_candidates(label) if block in c.source_blocks]

        if len(candidates) == 0:
            return None

        if len(candidates) == 1:
            return candidates[0]

        raise ValueError(
            f"Multiple candidates found for block {block.id} "
            f"with label '{label}'. Expected at most one."
        )

    def get_best_candidate(self, block: Blocks) -> Candidate | None:
        """Get the highest-scoring successfully constructed candidate for a block.

        When a block has candidates for multiple labels, this returns the one
        with the highest score. This is the "winning" candidate for reporting
        and output purposes.

        Args:
            block: The block to get the best candidate for

        Returns:
            The highest-scoring successfully constructed candidate, or None
            if no successfully constructed candidate exists
        """
        candidates = self.get_all_candidates_for_block(block)
        valid_candidates = [c for c in candidates if c.constructed is not None]

        if not valid_candidates:
            return None

        # Return the highest-scoring candidate
        return max(valid_candidates, key=lambda c: c.score)

    # TODO I think this API is broken - there can be multiple labels per block,
    # but we only return one here.
    def get_label(self, block: Blocks) -> str | None:
        """Get the label for a block from its highest-scoring constructed candidate.

        Returns the label of the successfully constructed candidate with the
        highest score for the given block, or None if no successfully
        constructed candidate exists.

        This is a convenience method equivalent to:
            candidate = result.get_best_candidate(block)
            return candidate.label if candidate else None

        Args:
            block: The block to get the label for

        Returns:
            The label string of the highest-scoring constructed candidate,
            None otherwise
        """
        best_candidate = self.get_best_candidate(block)
        return best_candidate.label if best_candidate else None

    def add_candidate(self, candidate: Candidate) -> None:
        """Add a single candidate.

        The label is extracted from candidate.label.

        Args:
            candidate: The candidate to add

        Raises:
            ValueError: If candidate has source_blocks that are not in PageData
        """
        for source_block in candidate.source_blocks:
            self._validate_block_in_page_data(source_block, "candidate.source_blocks")

        label = candidate.label
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
