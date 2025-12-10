"""ClassificationResult class for single page classification."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.removal_reason import RemovalReason
from build_a_long.pdf_extract.extractor.extractor import PageData
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    LegoPageElements,
    Page,
)
from build_a_long.pdf_extract.extractor.page_blocks import Blocks

if TYPE_CHECKING:
    from build_a_long.pdf_extract.classifier.label_classifier import LabelClassifier

log = logging.getLogger(__name__)

# Score key can be either a single Block or a tuple of Blocks (for pairings)
ScoreKey = Blocks | tuple[Blocks, ...]


class CandidateFailedError(Exception):
    """Raised when a candidate cannot be built due to a failure.

    This exception carries information about which candidate failed,
    allowing callers to potentially create replacement candidates and retry.
    """

    def __init__(self, candidate: Candidate, message: str):
        super().__init__(message)
        self.candidate = candidate


class _BuildSnapshot(BaseModel):
    """Snapshot of candidate and consumed block state for rollback.

    This is used to implement transactional semantics in build():
    if a classifier build fails, we can restore the state as if
    the build never started.
    """

    model_config = {"frozen": True}

    # Map candidate id -> (constructed value, failure_reason)
    candidate_states: dict[int, tuple[LegoPageElements | None, str | None]]
    # Set of consumed block IDs
    consumed_blocks: set[int]


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

    skipped_reason: str | None = None
    """If set, classification was skipped for this page.
    
    This is used for pages that cannot be reasonably classified, such as:
    - Pages with too many blocks (e.g., >1000 vector drawings)
    - Info/inventory pages where each character is a separate vector
    
    When set, most classification results will be empty.
    """

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

    def build_all_for_label(self, label: str) -> list[LegoPageElements]:
        """Build all candidates for a label using the registered classifier's build_all.

        This delegates to the classifier's build_all() method, allowing classifiers
        to implement custom coordination logic (e.g., Hungarian matching) before
        building individual candidates.

        Args:
            label: The label to build all candidates for

        Returns:
            List of successfully constructed LegoPageElements

        Raises:
            ValueError: If no classifier is registered for the label
        """
        classifier = self._classifiers.get(label)
        if not classifier:
            raise ValueError(f"No classifier registered for label '{label}'")

        return classifier.build_all(self)

    def build(self, candidate: Candidate, **kwargs: Any) -> LegoPageElements:
        """Construct a candidate using the registered classifier.

        This is the entry point for top-down construction. If the build fails,
        all changes to candidate states and consumed blocks are automatically
        rolled back, ensuring transactional semantics.

        If a nested candidate fails due to conflicts, this method will attempt
        to create replacement candidates and retry the build.

        Args:
            candidate: The candidate to construct
            **kwargs: Additional keyword arguments passed to the classifier's
                build method. For example, DiagramClassifier accepts
                constraint_bbox to limit clustering.
        """
        if candidate.constructed:
            return candidate.constructed

        if candidate.failure_reason:
            raise CandidateFailedError(
                candidate, f"Candidate failed: {candidate.failure_reason}"
            )

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
                raise CandidateFailedError(candidate, failure_msg)

        classifier = self._classifiers.get(candidate.label)
        if not classifier:
            raise ValueError(f"No classifier registered for label '{candidate.label}'")

        # Take snapshot before building for automatic rollback on failure
        snapshot = self._take_snapshot()

        try:
            element = classifier.build(candidate, self, **kwargs)
            candidate.constructed = element

            # Sync candidate bbox with constructed element's bbox.
            # The constructed element may have a different bbox (e.g., Step's
            # bbox includes diagram which is only determined at build time).
            if candidate.bbox != element.bbox:
                log.debug(
                    "[build] Updating candidate bbox from %s to %s - This indicate the bbox changed between score and build, and may indicate a classification bug",
                    candidate.bbox,
                    element.bbox,
                )
            candidate.bbox = element.bbox

            # Mark blocks as consumed
            log.debug(
                "[build] Marking %d blocks as consumed for '%s' at %s: %s",
                len(candidate.source_blocks),
                candidate.label,
                candidate.bbox,
                [b.id for b in candidate.source_blocks],
            )
            for block in candidate.source_blocks:
                self._consumed_blocks.add(block.id)

            # Fail other candidates that use these blocks
            self._fail_conflicting_candidates(candidate)

            return element
        except CandidateFailedError as e:
            # A nested candidate failed - rollback and check if we can retry
            self._restore_snapshot(snapshot)

            # If the failed candidate has a "Replaced by reduced candidate" reason,
            # we may be able to find the replacement and the caller can retry
            failed_candidate = e.candidate
            if (
                failed_candidate.failure_reason
                and "Replaced by reduced candidate" in failed_candidate.failure_reason
            ):
                # The failed candidate was replaced - caller should retry with
                # new candidates available
                log.debug(
                    "[build] Nested candidate %s (%s) was replaced, "
                    "propagating for retry",
                    failed_candidate.label,
                    failed_candidate.bbox,
                )
            raise
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
        """Mark other candidates sharing blocks with winner as failed.

        For candidates that support re-scoring, we try to create a reduced
        version without the conflicting blocks before failing them entirely.
        """
        winner_block_ids = {b.id for b in winner.source_blocks}

        if not winner_block_ids:
            return

        for label, candidates in self.candidates.items():
            for candidate in candidates:
                if candidate is winner:
                    continue
                if candidate.failure_reason:
                    continue

                # Check for overlap
                conflicting_block_ids = {
                    b.id for b in candidate.source_blocks if b.id in winner_block_ids
                }

                if not conflicting_block_ids:
                    continue

                # Fall back to failing the candidate
                candidate_block_ids = [b.id for b in candidate.source_blocks]
                candidate.failure_reason = (
                    f"Lost conflict to '{winner.label}' at {winner.bbox} "
                    f"(winner_blocks={sorted(winner_block_ids)}, "
                    f"candidate_blocks={candidate_block_ids}, "
                    f"conflicting={sorted(conflicting_block_ids)})"
                )

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

    # TODO Reconsider the removal API below - do we need it? We have been
    # capturing all blocks by a element.
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
