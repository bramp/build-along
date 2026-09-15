"""
Dependency-aware page interpretation search with backtracking.

This module finds the best interpretation of a page by searching over
block assignments while respecting element dependencies. Unlike greedy
classification, this approach:

1. Assigns blocks to primitive element types (PartImage, PartCount, etc.)
2. Automatically forms composites when dependencies are satisfied
3. Rescores affected elements after each assignment
4. Backtracks when hitting dead ends

Key Concepts
------------
- **Primitives**: Elements directly derived from blocks (PartImage, PartCount,
  StepNumber, ArrowHead, DiagramBlock, etc.)
- **Composites**: Elements formed from primitives (Part = PartImage + PartCount,
  Arrow = ArrowHead + ArrowShaft, Step = StepNumber + PartsList + Diagram)
- **Dependencies**: Required and optional primitives for each composite

The Problem
-----------
Some elements can't be scored until their dependencies are resolved:
- A Part requires both a PartImage and PartCount
- "2x" could be PartCount (if near a PartImage) or StepCount (if in a substep)
- The correct interpretation depends on what OTHER blocks are assigned

The Solution
------------
Search with incremental composite formation:
1. Pick an unassigned block (most constrained first)
2. Try each valid primitive type for this block
3. After each assignment, check if any composite is now complete
4. Score newly-completed composites
5. Prune branches that can't beat the best solution
6. Backtrack to try alternatives

Usage
-----
    searcher = DependencyAwareSearcher(page_data)
    result = searcher.search()
    # result.composites contains the formed elements
    # result.assignments maps block_id -> primitive type
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol

from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.extractor import PageData
from build_a_long.pdf_extract.extractor.page_blocks import Blocks

log = logging.getLogger(__name__)


# =============================================================================
# Element Types and Dependencies
# =============================================================================


class PrimitiveType(Enum):
    """Primitive element types - directly derived from blocks.

    These are the atomic types that blocks can be assigned to.
    Composites are formed from combinations of primitives.
    """

    # Parts-related
    PART_IMAGE = auto()
    PART_COUNT = auto()
    PART_NUMBER = auto()
    PIECE_LENGTH = auto()
    SHINE = auto()

    # Step-related
    STEP_NUMBER = auto()
    STEP_COUNT = auto()  # "2x" in substep context

    # Arrow parts
    ARROW_HEAD = auto()
    ARROW_SHAFT = auto()

    # Diagram blocks
    DIAGRAM_BLOCK = auto()

    # Other
    PAGE_NUMBER = auto()
    PROGRESS_BAR = auto()
    BAG_NUMBER = auto()
    OPEN_BAG = auto()
    ROTATION_SYMBOL = auto()

    # Special: block is not classified
    UNASSIGNED = auto()
    BACKGROUND = auto()  # Filtered out


class CompositeType(Enum):
    """Composite element types - formed from primitives."""

    PART = auto()  # PartImage + PartCount + optional(PartNumber, PieceLength)
    ARROW = auto()  # ArrowHead + ArrowShaft
    DIAGRAM = auto()  # Multiple DiagramBlocks (contiguous)
    PARTS_LIST = auto()  # Multiple Parts in a grid
    SUBSTEP = auto()  # Diagram + optional(StepCount, Arrow)
    STEP = auto()  # StepNumber + PartsList + Diagram
    PAGE = auto()  # Everything together


@dataclass(frozen=True)
class ElementDependency:
    """Defines what primitives a composite requires.

    Attributes:
        composite_type: The type of composite element
        required: Primitive types that MUST all be present
        optional: Primitive types that CAN be present (boost score)
        spatial_constraint: How primitives must be spatially related
    """

    composite_type: CompositeType
    required: frozenset[PrimitiveType]
    optional: frozenset[PrimitiveType]
    spatial_constraint: str  # "adjacent", "nearby", "contains", "contiguous"


# Define the dependency graph
DEPENDENCIES: dict[CompositeType, ElementDependency] = {
    CompositeType.PART: ElementDependency(
        composite_type=CompositeType.PART,
        required=frozenset({PrimitiveType.PART_IMAGE, PrimitiveType.PART_COUNT}),
        optional=frozenset({PrimitiveType.PART_NUMBER, PrimitiveType.PIECE_LENGTH}),
        spatial_constraint="adjacent",  # PartCount below PartImage
    ),
    CompositeType.ARROW: ElementDependency(
        composite_type=CompositeType.ARROW,
        required=frozenset({PrimitiveType.ARROW_HEAD}),
        optional=frozenset({PrimitiveType.ARROW_SHAFT}),
        spatial_constraint="connected",  # Head at end of shaft
    ),
    CompositeType.DIAGRAM: ElementDependency(
        composite_type=CompositeType.DIAGRAM,
        required=frozenset(),  # Just needs contiguous blocks
        optional=frozenset({PrimitiveType.DIAGRAM_BLOCK}),
        spatial_constraint="contiguous",
    ),
    CompositeType.SUBSTEP: ElementDependency(
        composite_type=CompositeType.SUBSTEP,
        required=frozenset(),  # Needs a diagram area
        optional=frozenset({PrimitiveType.STEP_COUNT}),
        spatial_constraint="contains",
    ),
    CompositeType.STEP: ElementDependency(
        composite_type=CompositeType.STEP,
        required=frozenset({PrimitiveType.STEP_NUMBER}),
        optional=frozenset(),  # PartsList and Diagram are composites
        spatial_constraint="contains",
    ),
}


# =============================================================================
# Scoring Protocol
# =============================================================================


class PrimitiveScorer(Protocol):
    """Protocol for scoring block -> primitive assignments."""

    def score(self, block: Blocks, primitive_type: PrimitiveType) -> float:
        """Score how well a block fits a primitive type.

        Returns 0.0 if invalid, higher values for better fit.
        """
        ...


class CompositeScorer(Protocol):
    """Protocol for scoring composite elements."""

    def score(
        self,
        composite_type: CompositeType,
        primitives: list[tuple[Blocks, PrimitiveType]],
    ) -> float:
        """Score a composite formed from the given primitives.

        Returns 0.0 if invalid composite, higher values for better fit.
        """
        ...


# =============================================================================
# Primitive Hypothesis
# =============================================================================


@dataclass
class PrimitiveHypothesis:
    """A possible primitive type assignment for a block.

    Represents: "block X could be primitive type Y with base score Z"
    """

    block_id: int
    primitive_type: PrimitiveType
    score: float
    bbox: BBox


# =============================================================================
# Composite Candidate
# =============================================================================


@dataclass
class CompositeCandidate:
    """A potential composite element waiting for primitives.

    Tracks which primitives have been assigned and determines
    when the composite is complete.
    """

    composite_type: CompositeType
    expected_block_ids: set[int]
    """Block IDs that would form this composite if assigned correctly."""

    assigned_primitives: dict[int, PrimitiveType] = field(default_factory=dict)
    """Maps block_id -> assigned primitive type for blocks in this candidate."""

    score: float = 0.0
    """Score computed when complete (0.0 while incomplete)."""

    is_complete: bool = False

    def add_primitive(self, block_id: int, ptype: PrimitiveType) -> bool:
        """Add a primitive assignment. Returns True if this affects us."""
        if block_id not in self.expected_block_ids:
            return False

        self.assigned_primitives[block_id] = ptype
        self._check_completeness()
        return True

    def remove_primitive(self, block_id: int) -> bool:
        """Remove a primitive assignment. Returns True if this affects us."""
        if block_id not in self.assigned_primitives:
            return False

        del self.assigned_primitives[block_id]
        self.is_complete = False
        self.score = 0.0
        return True

    def _check_completeness(self) -> None:
        """Check if all required primitives are present."""
        dep = DEPENDENCIES.get(self.composite_type)
        if dep is None:
            return

        # Get the primitive types we have
        assigned_types = set(self.assigned_primitives.values())

        # Check if all required types are present
        if dep.required.issubset(assigned_types):
            self.is_complete = True
        else:
            self.is_complete = False

    def get_missing_required(self) -> set[PrimitiveType]:
        """Get required primitive types that are still missing."""
        dep = DEPENDENCIES.get(self.composite_type)
        if dep is None:
            return set()

        assigned_types = set(self.assigned_primitives.values())
        return set(dep.required - assigned_types)


# =============================================================================
# Formed Composite
# =============================================================================


@dataclass
class FormedComposite:
    """A successfully formed composite element."""

    composite_type: CompositeType
    primitives: dict[int, PrimitiveType]
    """Maps block_id -> primitive type for all blocks in this composite."""

    score: float
    bbox: BBox
    """Bounding box encompassing all primitive bboxes."""


# =============================================================================
# Search State
# =============================================================================


@dataclass
class SearchState:
    """The state of the search at any point.

    This is an immutable snapshot that can be copied for branching.
    """

    # Block assignments
    assignments: dict[int, PrimitiveType] = field(default_factory=dict)
    """Maps block_id -> assigned primitive type."""

    # Formed composites
    composites: list[FormedComposite] = field(default_factory=list)

    # Pending composite candidates
    pending_candidates: list[CompositeCandidate] = field(default_factory=list)

    # Blocks that are part of a composite (can't be reassigned)
    claimed_blocks: set[int] = field(default_factory=set)

    # Orphan tracking (primitives not in any composite)
    orphan_blocks: set[int] = field(default_factory=set)

    # Score
    composite_score: float = 0.0
    """Sum of scores from formed composites."""

    orphan_penalty: float = 0.0
    """Penalty for orphaned primitives."""

    @property
    def total_score(self) -> float:
        """Total score for this state."""
        return self.composite_score - self.orphan_penalty

    def copy(self) -> SearchState:
        """Create a deep copy of this state."""
        return SearchState(
            assignments=self.assignments.copy(),
            composites=list(self.composites),
            pending_candidates=[deepcopy(c) for c in self.pending_candidates],
            claimed_blocks=self.claimed_blocks.copy(),
            orphan_blocks=self.orphan_blocks.copy(),
            composite_score=self.composite_score,
            orphan_penalty=self.orphan_penalty,
        )


# =============================================================================
# Search Result
# =============================================================================


@dataclass
class SearchResult:
    """Result of the dependency-aware search."""

    state: SearchState
    """The best state found."""

    states_explored: int
    """Number of states evaluated during search."""

    pruned_count: int
    """Number of branches pruned."""

    backtrack_count: int
    """Number of times backtracking occurred."""


# =============================================================================
# Default Scorers
# =============================================================================


class DefaultPrimitiveScorer:
    """Default implementation of primitive scoring.

    This is a placeholder - in practice, this would use the existing
    classifier scoring logic.
    """

    def __init__(self, page_data: PageData) -> None:
        self._page_data = page_data

    def score(self, block: Blocks, primitive_type: PrimitiveType) -> float:
        """Score how well a block fits a primitive type."""
        # Placeholder - returns 0.5 for all valid assignments
        # Real implementation would call into existing classifiers
        return 0.5


class DefaultCompositeScorer:
    """Default implementation of composite scoring."""

    def __init__(self, page_data: PageData) -> None:
        self._page_data = page_data
        self._block_index = {b.id: b for b in page_data.blocks}

    def score(
        self,
        composite_type: CompositeType,
        primitives: list[tuple[Blocks, PrimitiveType]],
    ) -> float:
        """Score a composite formed from the given primitives."""
        if composite_type == CompositeType.PART:
            return self._score_part(primitives)
        elif composite_type == CompositeType.ARROW:
            return self._score_arrow(primitives)
        elif composite_type == CompositeType.DIAGRAM:
            return self._score_diagram(primitives)
        else:
            # Default scoring
            return 0.5 * len(primitives)

    def _score_part(self, primitives: list[tuple[Blocks, PrimitiveType]]) -> float:
        """Score a Part composite."""
        has_image = any(p[1] == PrimitiveType.PART_IMAGE for p in primitives)
        has_count = any(p[1] == PrimitiveType.PART_COUNT for p in primitives)

        if not (has_image and has_count):
            return 0.0

        # Base score for having required components
        score = 1.0

        # Bonus for spatial relationship (count below image)
        image_block = next(
            (b for b, t in primitives if t == PrimitiveType.PART_IMAGE), None
        )
        count_block = next(
            (b for b, t in primitives if t == PrimitiveType.PART_COUNT), None
        )

        if image_block and count_block:
            # Check if count is below image
            if count_block.bbox.y0 >= image_block.bbox.y1 - 5:
                score += 0.3  # Good spatial relationship

            # Check horizontal alignment
            img_cx = (image_block.bbox.x0 + image_block.bbox.x1) / 2
            cnt_cx = (count_block.bbox.x0 + count_block.bbox.x1) / 2
            if abs(img_cx - cnt_cx) < 20:
                score += 0.2  # Well aligned

        return score

    def _score_arrow(self, primitives: list[tuple[Blocks, PrimitiveType]]) -> float:
        """Score an Arrow composite."""
        has_head = any(p[1] == PrimitiveType.ARROW_HEAD for p in primitives)

        if not has_head:
            return 0.0

        score = 0.8  # Base score for having head

        # Bonus for having shaft
        has_shaft = any(p[1] == PrimitiveType.ARROW_SHAFT for p in primitives)
        if has_shaft:
            score += 0.4

        return score

    def _score_diagram(self, primitives: list[tuple[Blocks, PrimitiveType]]) -> float:
        """Score a Diagram composite."""
        diagram_blocks = [b for b, t in primitives if t == PrimitiveType.DIAGRAM_BLOCK]

        if not diagram_blocks:
            return 0.0

        # Score based on number of contiguous blocks
        # More blocks = higher score (up to a point)
        num_blocks = len(diagram_blocks)
        base_score = min(num_blocks * 0.1, 2.0)

        # Check contiguity
        if num_blocks > 1 and self._are_contiguous(diagram_blocks):
            base_score += 0.5

        return base_score

    def _are_contiguous(self, blocks: list[Blocks]) -> bool:
        """Check if blocks form a contiguous region."""
        if len(blocks) <= 1:
            return True

        # Simple check: all blocks should be within reasonable distance
        # A more sophisticated check would use union-find with overlap detection
        for i, b1 in enumerate(blocks):
            has_neighbor = False
            for j, b2 in enumerate(blocks):
                if i == j:
                    continue
                # Check if bboxes are close (within 10 units)
                gap_x = max(0, max(b1.bbox.x0 - b2.bbox.x1, b2.bbox.x0 - b1.bbox.x1))
                gap_y = max(0, max(b1.bbox.y0 - b2.bbox.y1, b2.bbox.y0 - b1.bbox.y1))
                if gap_x < 10 and gap_y < 10:
                    has_neighbor = True
                    break
            if not has_neighbor:
                return False

        return True


# =============================================================================
# Dependency-Aware Searcher
# =============================================================================


class DependencyAwareSearcher:
    """Search for the best page interpretation respecting element dependencies.

    This searcher:
    1. Takes hypotheses for each block (possible primitive types with scores)
    2. Searches over assignments using backtracking with pruning
    3. Forms composites when dependencies are satisfied
    4. Returns the highest-scoring complete interpretation
    """

    # Search parameters (class defaults, can be overridden per-instance)
    ORPHAN_PENALTY: float = 0.3  # Penalty per orphaned primitive

    def __init__(
        self,
        page_data: PageData,
        primitive_scorer: PrimitiveScorer | None = None,
        composite_scorer: CompositeScorer | None = None,
        max_states: int = 100_000,
    ) -> None:
        """Initialize the searcher.

        Args:
            page_data: The page data with blocks to classify
            primitive_scorer: Optional custom primitive scorer
            composite_scorer: Optional custom composite scorer
            max_states: Maximum states to explore before giving up
        """
        self._page_data = page_data
        self._block_index = {b.id: b for b in page_data.blocks}

        self._primitive_scorer = primitive_scorer or DefaultPrimitiveScorer(page_data)
        self._composite_scorer = composite_scorer or DefaultCompositeScorer(page_data)
        self._max_states = max_states

        # Search statistics
        self._states_explored = 0
        self._pruned_count = 0
        self._backtrack_count = 0

        # Best solution found
        self._best_state: SearchState | None = None
        self._best_score: float = float("-inf")

    def search(
        self,
        hypotheses: dict[int, list[PrimitiveHypothesis]],
        composite_candidates: list[CompositeCandidate] | None = None,
    ) -> SearchResult:
        """Search for the best page interpretation.

        Args:
            hypotheses: Maps block_id -> list of possible primitive assignments
            composite_candidates: Optional pre-computed composite candidates.
                If None, will be inferred from hypotheses.

        Returns:
            SearchResult with the best interpretation found.
        """
        self._states_explored = 0
        self._pruned_count = 0
        self._backtrack_count = 0
        self._best_state = None
        self._best_score = float("-inf")

        # Separate blocks into fixed (only one hypothesis) and ambiguous
        fixed_assignments: dict[int, PrimitiveType] = {}
        ambiguous_blocks: list[int] = []
        ambiguous_hypotheses: dict[int, list[PrimitiveHypothesis]] = {}

        for block_id, block_hyps in hypotheses.items():
            if not block_hyps:
                continue

            if len(block_hyps) == 1:
                # Only one option - fixed
                fixed_assignments[block_id] = block_hyps[0].primitive_type
            else:
                # Multiple options - ambiguous
                ambiguous_blocks.append(block_id)
                # Sort by score descending for better pruning
                ambiguous_hypotheses[block_id] = sorted(
                    block_hyps, key=lambda h: -h.score
                )

        log.info(
            "[search] Fixed: %d blocks, Ambiguous: %d blocks",
            len(fixed_assignments),
            len(ambiguous_blocks),
        )

        # Initialize state with fixed assignments
        initial_state = SearchState(assignments=fixed_assignments.copy())

        # Add composite candidates
        if composite_candidates:
            initial_state.pending_candidates = [
                deepcopy(c) for c in composite_candidates
            ]
            # Update candidates with fixed assignments
            for block_id, ptype in fixed_assignments.items():
                for candidate in initial_state.pending_candidates:
                    candidate.add_primitive(block_id, ptype)

        # Check if any composites are already complete from fixed assignments
        initial_state = self._form_complete_composites(initial_state)

        if not ambiguous_blocks:
            # No ambiguity - compute final score and return
            initial_state = self._compute_orphans(initial_state)
            return SearchResult(
                state=initial_state,
                states_explored=1,
                pruned_count=0,
                backtrack_count=0,
            )

        # Order ambiguous blocks by MRV heuristic (most constrained first)
        ambiguous_blocks = self._order_by_mrv(ambiguous_blocks, ambiguous_hypotheses)

        # Search using backtracking with pruning
        self._backtrack(
            state=initial_state,
            remaining_blocks=ambiguous_blocks,
            hypotheses=ambiguous_hypotheses,
        )

        if self._best_state is None:
            # No valid solution found - return initial state
            log.warning("[search] No valid solution found, returning initial state")
            self._best_state = initial_state

        return SearchResult(
            state=self._best_state,
            states_explored=self._states_explored,
            pruned_count=self._pruned_count,
            backtrack_count=self._backtrack_count,
        )

    def _backtrack(
        self,
        state: SearchState,
        remaining_blocks: list[int],
        hypotheses: dict[int, list[PrimitiveHypothesis]],
    ) -> None:
        """Recursive backtracking search.

        Args:
            state: Current search state
            remaining_blocks: Block IDs that still need assignment
            hypotheses: Maps block_id -> list of hypotheses
        """
        # Check limits
        if self._states_explored >= self._max_states:
            return

        self._states_explored += 1

        # Base case: all blocks assigned
        if not remaining_blocks:
            final_state = self._compute_orphans(state)
            if final_state.total_score > self._best_score:
                self._best_score = final_state.total_score
                self._best_state = final_state
            return

        # Pick next block to assign
        block_id = remaining_blocks[0]
        rest = remaining_blocks[1:]

        # Try each hypothesis for this block
        for hyp in hypotheses[block_id]:
            # Skip if this block is already claimed by a composite
            if block_id in state.claimed_blocks:
                continue

            # Make assignment
            new_state = self._apply_assignment(state, block_id, hyp.primitive_type)

            # Check if any composites are now complete
            new_state = self._form_complete_composites(new_state)

            # Compute optimistic upper bound
            upper_bound = self._compute_upper_bound(new_state, rest, hypotheses)

            # Pruning
            if upper_bound <= self._best_score:
                self._pruned_count += 1
                continue

            # Recurse
            self._backtrack(new_state, rest, hypotheses)

        # If we tried all options and none worked, that's backtracking
        self._backtrack_count += 1

    def _apply_assignment(
        self,
        state: SearchState,
        block_id: int,
        primitive_type: PrimitiveType,
    ) -> SearchState:
        """Apply a primitive assignment to the state.

        Returns a new state with the assignment applied.
        """
        new_state = state.copy()
        new_state.assignments[block_id] = primitive_type

        # Update pending composite candidates
        for candidate in new_state.pending_candidates:
            candidate.add_primitive(block_id, primitive_type)

        return new_state

    def _form_complete_composites(self, state: SearchState) -> SearchState:
        """Check for and form complete composites.

        When a composite candidate has all required primitives assigned,
        form the composite and update the state.
        """
        new_state = state.copy()
        still_pending: list[CompositeCandidate] = []

        for candidate in new_state.pending_candidates:
            if candidate.is_complete:
                # Form the composite
                composite = self._create_composite(candidate)
                if composite is not None:
                    new_state.composites.append(composite)
                    new_state.composite_score += composite.score
                    new_state.claimed_blocks.update(
                        candidate.assigned_primitives.keys()
                    )
                else:
                    # Failed to form (e.g., spatial constraints not met)
                    still_pending.append(candidate)
            else:
                still_pending.append(candidate)

        new_state.pending_candidates = still_pending
        return new_state

    def _create_composite(
        self, candidate: CompositeCandidate
    ) -> FormedComposite | None:
        """Create a FormedComposite from a complete candidate.

        Returns None if the composite is invalid (e.g., spatial constraints).
        """
        # Gather primitives
        primitives: list[tuple[Blocks, PrimitiveType]] = []
        for block_id, ptype in candidate.assigned_primitives.items():
            block = self._block_index.get(block_id)
            if block is None:
                return None
            primitives.append((block, ptype))

        if not primitives:
            return None

        # Score the composite
        score = self._composite_scorer.score(candidate.composite_type, primitives)
        if score <= 0:
            return None

        # Compute bounding box
        bboxes = [p[0].bbox for p in primitives]
        combined_bbox = bboxes[0]
        for bbox in bboxes[1:]:
            combined_bbox = combined_bbox.union(bbox)

        return FormedComposite(
            composite_type=candidate.composite_type,
            primitives=dict(candidate.assigned_primitives),
            score=score,
            bbox=combined_bbox,
        )

    def _compute_orphans(self, state: SearchState) -> SearchState:
        """Compute orphan penalty for primitives not in any composite."""
        new_state = state.copy()

        # Find blocks that are assigned but not claimed
        orphans: set[int] = set()
        for block_id, ptype in new_state.assignments.items():
            # Check if this primitive type should be part of a composite
            if (
                block_id not in new_state.claimed_blocks
                and self._should_be_in_composite(ptype)
            ):
                orphans.add(block_id)

        new_state.orphan_blocks = orphans
        new_state.orphan_penalty = len(orphans) * self.ORPHAN_PENALTY

        return new_state

    def _should_be_in_composite(self, ptype: PrimitiveType) -> bool:
        """Check if a primitive type should be part of a composite."""
        # These primitive types are expected to be part of composites
        composite_primitives = {
            PrimitiveType.PART_IMAGE,
            PrimitiveType.PART_COUNT,
            PrimitiveType.PART_NUMBER,
            PrimitiveType.ARROW_HEAD,
            PrimitiveType.ARROW_SHAFT,
            PrimitiveType.DIAGRAM_BLOCK,
            PrimitiveType.STEP_NUMBER,
        }
        return ptype in composite_primitives

    def _compute_upper_bound(
        self,
        state: SearchState,
        remaining_blocks: list[int],
        hypotheses: dict[int, list[PrimitiveHypothesis]],
    ) -> float:
        """Compute optimistic upper bound on score from this state.

        Assumes best-case for all remaining blocks.
        """
        bound = state.total_score

        for block_id in remaining_blocks:
            if block_id in state.claimed_blocks:
                continue

            block_hyps = hypotheses.get(block_id, [])
            if block_hyps:
                # Assume best hypothesis
                best_hyp = max(block_hyps, key=lambda h: h.score)
                # Optimistic: assume it will form a composite with bonus
                bound += best_hyp.score + 1.0

        return bound

    def _order_by_mrv(
        self,
        blocks: list[int],
        hypotheses: dict[int, list[PrimitiveHypothesis]],
    ) -> list[int]:
        """Order blocks by Most Remaining Values heuristic.

        Blocks with fewer valid options are tried first (fail-fast).
        """
        return sorted(blocks, key=lambda b: len(hypotheses.get(b, [])))


# =============================================================================
# Utility Functions
# =============================================================================


def create_part_candidate(
    part_image_block_id: int, part_count_block_id: int
) -> CompositeCandidate:
    """Create a composite candidate for a Part.

    Args:
        part_image_block_id: Block ID of the potential PartImage
        part_count_block_id: Block ID of the potential PartCount

    Returns:
        A CompositeCandidate that will be complete when both blocks
        are assigned to their respective primitive types.
    """
    return CompositeCandidate(
        composite_type=CompositeType.PART,
        expected_block_ids={part_image_block_id, part_count_block_id},
    )


def create_arrow_candidate(
    head_block_id: int, shaft_block_id: int | None = None
) -> CompositeCandidate:
    """Create a composite candidate for an Arrow.

    Args:
        head_block_id: Block ID of the ArrowHead
        shaft_block_id: Optional block ID of the ArrowShaft

    Returns:
        A CompositeCandidate for the arrow.
    """
    expected = {head_block_id}
    if shaft_block_id is not None:
        expected.add(shaft_block_id)

    return CompositeCandidate(
        composite_type=CompositeType.ARROW,
        expected_block_ids=expected,
    )


def create_diagram_candidate(block_ids: set[int]) -> CompositeCandidate:
    """Create a composite candidate for a Diagram.

    Args:
        block_ids: Block IDs that could form the diagram

    Returns:
        A CompositeCandidate for the diagram.
    """
    return CompositeCandidate(
        composite_type=CompositeType.DIAGRAM,
        expected_block_ids=block_ids,
    )
