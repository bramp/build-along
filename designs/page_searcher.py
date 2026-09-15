"""
Page interpretation search using branch-and-bound.

This module finds the best interpretation of a page by searching over
ambiguous block assignments. Instead of greedily committing to classifications,
we explore multiple hypotheses and pick the one with highest coherence score.

The Problem
-----------
Some blocks have multiple plausible interpretations:
- "2x" could be a PartCount (need 2 parts) or StepMultiplier (repeat 2 times)
- A small image could be a PartImage or part of a Diagram
- An arrow could belong to different Steps

The correct interpretation depends on context - what other blocks are nearby
and how they're classified.

The Solution
------------
1. Score all blocks to find ambiguous ones (multiple high-scoring labels)
2. Fix unambiguous blocks (only one plausible label)
3. Search over combinations of ambiguous assignments
4. Score each complete interpretation by coherence (mutual support)
5. Return the highest-scoring interpretation

Complexity
----------
With N ambiguous blocks and K labels each, worst case is K^N.
But with pruning (branch-and-bound) and beam search, practical complexity
is much lower. Typical pages have ~5-15 ambiguous blocks.

Usage
-----
    searcher = PageSearcher()
    interpretation = searcher.search(page_data, candidates)
    # interpretation.assignments maps block_id -> label
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.extractor import PageData
from build_a_long.pdf_extract.extractor.page_blocks import Blocks, Drawing, Image, Text

log = logging.getLogger(__name__)


@dataclass
class BlockHypothesis:
    """A possible interpretation for a single block."""

    block_id: int
    label: str
    score: float  # Base score for this interpretation (without context)


@dataclass
class PageInterpretation:
    """One possible interpretation of the entire page.

    This represents a (partial or complete) assignment of labels to blocks.
    """

    assignments: dict[int, str] = field(default_factory=dict)
    """Maps block_id -> label for decided blocks."""

    score: float = 0.0
    """Total coherence score for this interpretation."""

    def copy(self) -> PageInterpretation:
        """Create a copy of this interpretation."""
        return PageInterpretation(
            assignments=self.assignments.copy(),
            score=self.score,
        )

    def with_assignment(self, block_id: int, label: str) -> PageInterpretation:
        """Return a new interpretation with an additional assignment."""
        new = self.copy()
        new.assignments[block_id] = label
        return new


@dataclass
class SearchResult:
    """Result of searching for the best page interpretation."""

    interpretation: PageInterpretation
    """The best interpretation found."""

    fixed_count: int
    """Number of blocks with unambiguous labels."""

    ambiguous_count: int
    """Number of blocks that required search."""

    interpretations_evaluated: int
    """Total number of interpretations scored during search."""


class PageSearcher:
    """Find the best interpretation of a page using search.

    This class implements a search over possible block assignments,
    using branch-and-bound pruning and optional beam search for efficiency.
    """

    # Threshold for considering a block ambiguous
    # If second-best score is within this ratio of best, it's ambiguous
    AMBIGUITY_THRESHOLD = 0.7

    # Score difference below which two labels are considered "tied"
    SCORE_TIE_THRESHOLD = 0.1

    # Maximum number of ambiguous blocks to search over
    # Beyond this, we fall back to greedy for remaining blocks
    MAX_AMBIGUOUS_BLOCKS = 15

    # Beam width for beam search (0 = full branch-and-bound)
    BEAM_WIDTH = 50

    def __init__(self, page_data: PageData) -> None:
        """Initialize with page data for spatial queries."""
        self._page_data = page_data
        self._block_index: dict[int, Blocks] = {b.id: b for b in page_data.blocks}
        self._interpretations_evaluated = 0

    def search(
        self,
        hypotheses: dict[int, list[BlockHypothesis]],
    ) -> SearchResult:
        """Find the best page interpretation.

        Args:
            hypotheses: Maps block_id -> list of possible interpretations,
                       each with a base score.

        Returns:
            SearchResult with the best interpretation found.
        """
        self._interpretations_evaluated = 0

        # Separate ambiguous from fixed blocks
        fixed: dict[int, str] = {}
        ambiguous: list[int] = []
        ambiguous_hypotheses: dict[int, list[BlockHypothesis]] = {}

        for block_id, block_hyps in hypotheses.items():
            if not block_hyps:
                continue

            # Sort by score descending
            sorted_hyps = sorted(block_hyps, key=lambda h: -h.score)
            best_score = sorted_hyps[0].score

            if len(sorted_hyps) == 1:
                # Only one option - fixed
                fixed[block_id] = sorted_hyps[0].label
            elif (
                len(sorted_hyps) > 1
                and sorted_hyps[1].score >= best_score * self.AMBIGUITY_THRESHOLD
            ):
                # Second option is close enough - ambiguous
                ambiguous.append(block_id)
                # Keep only plausible hypotheses
                ambiguous_hypotheses[block_id] = [
                    h
                    for h in sorted_hyps
                    if h.score >= best_score * self.AMBIGUITY_THRESHOLD
                ]
            else:
                # Best is clearly better - fixed
                fixed[block_id] = sorted_hyps[0].label

        log.info(
            "[search] Fixed: %d blocks, Ambiguous: %d blocks",
            len(fixed),
            len(ambiguous),
        )

        # Limit ambiguous blocks to search
        if len(ambiguous) > self.MAX_AMBIGUOUS_BLOCKS:
            log.warning(
                "[search] Too many ambiguous blocks (%d), limiting to %d",
                len(ambiguous),
                self.MAX_AMBIGUOUS_BLOCKS,
            )
            # Sort by "most ambiguous" (smallest score gap) and take top N
            ambiguous.sort(
                key=lambda bid: ambiguous_hypotheses[bid][0].score
                - ambiguous_hypotheses[bid][-1].score
            )
            extra = ambiguous[self.MAX_AMBIGUOUS_BLOCKS :]
            ambiguous = ambiguous[: self.MAX_AMBIGUOUS_BLOCKS]

            # Fix the extra blocks to their best hypothesis
            for bid in extra:
                fixed[bid] = ambiguous_hypotheses[bid][0].label

        # Start with fixed assignments
        initial = PageInterpretation(assignments=fixed.copy())
        initial.score = self._compute_score(initial.assignments)

        if not ambiguous:
            # No ambiguity - return fixed assignments
            return SearchResult(
                interpretation=initial,
                fixed_count=len(fixed),
                ambiguous_count=0,
                interpretations_evaluated=1,
            )

        # Search over ambiguous blocks
        if self.BEAM_WIDTH > 0:
            best = self._beam_search(initial, ambiguous, ambiguous_hypotheses)
        else:
            best = self._branch_and_bound(
                initial, ambiguous, ambiguous_hypotheses, best_so_far=None
            )

        return SearchResult(
            interpretation=best,
            fixed_count=len(fixed),
            ambiguous_count=len(ambiguous),
            interpretations_evaluated=self._interpretations_evaluated,
        )

    def _beam_search(
        self,
        initial: PageInterpretation,
        ambiguous: list[int],
        hypotheses: dict[int, list[BlockHypothesis]],
    ) -> PageInterpretation:
        """Search using beam search - keep top K at each step."""
        beam = [initial]

        for block_id in ambiguous:
            next_beam: list[PageInterpretation] = []

            for interp in beam:
                for hyp in hypotheses[block_id]:
                    new_interp = interp.with_assignment(block_id, hyp.label)
                    new_interp.score = self._compute_score(new_interp.assignments)
                    self._interpretations_evaluated += 1
                    next_beam.append(new_interp)

            # Keep top K
            next_beam.sort(key=lambda i: -i.score)
            beam = next_beam[: self.BEAM_WIDTH]

            log.debug(
                "[search] After block %d: beam has %d interpretations, best score=%.3f",
                block_id,
                len(beam),
                beam[0].score if beam else 0,
            )

        return beam[0] if beam else initial

    def _branch_and_bound(
        self,
        current: PageInterpretation,
        remaining: list[int],
        hypotheses: dict[int, list[BlockHypothesis]],
        best_so_far: PageInterpretation | None,
    ) -> PageInterpretation:
        """Search using branch-and-bound with pruning."""
        if not remaining:
            self._interpretations_evaluated += 1
            return current

        block_id = remaining[0]
        rest = remaining[1:]

        best = best_so_far

        for hyp in hypotheses[block_id]:
            new_interp = current.with_assignment(block_id, hyp.label)
            new_interp.score = self._compute_score(new_interp.assignments)
            self._interpretations_evaluated += 1

            # Pruning: compute optimistic bound
            if best is not None:
                optimistic = new_interp.score + self._optimistic_bound(rest, hypotheses)
                if optimistic <= best.score:
                    continue  # Can't beat best, skip this branch

            # Recurse
            result = self._branch_and_bound(new_interp, rest, hypotheses, best)

            if best is None or result.score > best.score:
                best = result

        return best if best is not None else current

    def _optimistic_bound(
        self,
        remaining: list[int],
        hypotheses: dict[int, list[BlockHypothesis]],
    ) -> float:
        """Compute optimistic upper bound on score from remaining blocks.

        Assumes best-case context scores for all remaining blocks.
        """
        bound = 0.0
        for block_id in remaining:
            if block_id in hypotheses:
                # Assume best hypothesis wins and gets max context bonus
                best_base = max(h.score for h in hypotheses[block_id])
                bound += best_base + 0.5  # Optimistic context bonus
        return bound

    def _compute_score(self, assignments: dict[int, str]) -> float:
        """Compute coherence score for an interpretation.

        This is the key scoring function that rewards consistent interpretations.
        Higher scores indicate better coherence between assignments.
        """
        score = 0.0

        for block_id, label in assignments.items():
            block = self._block_index.get(block_id)
            if block is None:
                continue

            # Base score (could come from original hypothesis)
            score += self._base_label_score(block, label)

            # Context score - how well does this fit with neighbors?
            score += self._context_score(block, label, assignments)

        return score

    def _base_label_score(self, block: Blocks, label: str) -> float:
        """Base score for assigning a label to a block.

        This captures intrinsic fitness (e.g., "2x" pattern for part_count).
        """
        # For now, return a constant. In practice, this would use
        # the original classifier scores.
        return 0.5

    def _context_score(
        self,
        block: Blocks,
        label: str,
        assignments: dict[int, str],
    ) -> float:
        """Score how well this label fits given other assignments.

        This is where mutual support between related elements is computed.
        """
        score = 0.0

        if label == "part_count":
            score += self._score_part_count_context(block, assignments)
        elif label == "step_multiplier":
            score += self._score_step_multiplier_context(block, assignments)
        elif label == "part_image":
            score += self._score_part_image_context(block, assignments)
        elif label == "diagram":
            score += self._score_diagram_context(block, assignments)
        elif label == "arrow":
            score += self._score_arrow_context(block, assignments)

        return score

    def _score_part_count_context(
        self,
        block: Blocks,
        assignments: dict[int, str],
    ) -> float:
        """Context score for part_count label."""
        score = 0.0

        # Reward if there's a part_image below
        image_below = self._find_image_below(block.bbox, max_distance=30)
        if image_below:
            neighbor_label = assignments.get(image_below.id)
            if neighbor_label == "part_image":
                score += 0.5  # Strong mutual support
            elif neighbor_label == "diagram":
                score -= 0.3  # Penalty - image claimed by diagram
            elif neighbor_label is None:
                score += 0.1  # Slight bonus - potential part_image

        # Penalty if too close to a step_number
        step_nearby = self._find_text_nearby_matching(
            block.bbox, max_distance=50, pattern_labels={"step_number"}
        )
        if step_nearby:
            neighbor_label = assignments.get(step_nearby.id)
            if neighbor_label == "step_number":
                score -= 0.2  # Might be step_multiplier instead

        return score

    def _score_step_multiplier_context(
        self,
        block: Blocks,
        assignments: dict[int, str],
    ) -> float:
        """Context score for step_multiplier label."""
        score = 0.0

        # Reward if there's a step_number nearby
        step_nearby = self._find_text_nearby_matching(
            block.bbox, max_distance=80, pattern_labels={"step_number"}
        )
        if step_nearby:
            neighbor_label = assignments.get(step_nearby.id)
            if neighbor_label == "step_number":
                score += 0.4  # Good support for step_multiplier

        # Penalty if there's clearly a part_image below
        image_below = self._find_image_below(block.bbox, max_distance=30)
        if image_below:
            neighbor_label = assignments.get(image_below.id)
            if neighbor_label == "part_image":
                score -= 0.4  # Likely part_count instead

        return score

    def _score_part_image_context(
        self,
        block: Blocks,
        assignments: dict[int, str],
    ) -> float:
        """Context score for part_image label."""
        score = 0.0

        # Reward if there's a part_count above
        text_above = self._find_text_above(block.bbox, max_distance=30)
        if text_above:
            neighbor_label = assignments.get(text_above.id)
            if neighbor_label == "part_count":
                score += 0.5  # Strong mutual support
            elif neighbor_label == "step_multiplier":
                score -= 0.3  # Penalty - text is claimed as step_multiplier

        # Reward if near other part_images (in a parts list)
        nearby_images = self._find_images_nearby(block.bbox, max_distance=50)
        part_image_neighbors = sum(
            1 for img in nearby_images if assignments.get(img.id) == "part_image"
        )
        score += 0.1 * part_image_neighbors  # Bonus for clustering

        return score

    def _score_diagram_context(
        self,
        block: Blocks,
        assignments: dict[int, str],
    ) -> float:
        """Context score for diagram label."""
        score = 0.0

        # Reward if near other diagram blocks (clustering)
        nearby = self._find_images_nearby(block.bbox, max_distance=20)
        diagram_neighbors = sum(
            1 for img in nearby if assignments.get(img.id) == "diagram"
        )
        score += 0.15 * diagram_neighbors

        # Penalty if there's a part_count directly above (should be part_image)
        text_above = self._find_text_above(block.bbox, max_distance=30)
        if text_above:
            neighbor_label = assignments.get(text_above.id)
            if neighbor_label == "part_count":
                score -= 0.4  # This image should probably be part_image

        return score

    def _score_arrow_context(
        self,
        block: Blocks,
        assignments: dict[int, str],
    ) -> float:
        """Context score for arrow label."""
        score = 0.0

        # Reward if near a diagram
        nearby_images = self._find_images_nearby(block.bbox, max_distance=50)
        diagram_neighbors = sum(
            1 for img in nearby_images if assignments.get(img.id) == "diagram"
        )
        if diagram_neighbors > 0:
            score += 0.3  # Arrows belong near diagrams

        return score

    # === Spatial Query Helpers ===

    def _find_image_below(
        self,
        bbox: BBox,
        max_distance: float,
    ) -> Blocks | None:
        """Find an image directly below the given bbox."""
        best: Blocks | None = None
        best_distance = max_distance

        for block in self._page_data.blocks:
            if not isinstance(block, Image | Drawing):
                continue

            # Check if below (block's top is below bbox's bottom)
            if block.bbox.y0 < bbox.y1:
                continue

            # Check horizontal overlap
            if block.bbox.x1 < bbox.x0 or block.bbox.x0 > bbox.x1:
                continue

            distance = block.bbox.y0 - bbox.y1
            if distance < best_distance:
                best = block
                best_distance = distance

        return best

    def _find_text_above(
        self,
        bbox: BBox,
        max_distance: float,
    ) -> Blocks | None:
        """Find a text block directly above the given bbox."""
        best: Blocks | None = None
        best_distance = max_distance

        for block in self._page_data.blocks:
            if not isinstance(block, Text):
                continue

            # Check if above (block's bottom is above bbox's top)
            if block.bbox.y1 > bbox.y0:
                continue

            # Check horizontal overlap
            if block.bbox.x1 < bbox.x0 or block.bbox.x0 > bbox.x1:
                continue

            distance = bbox.y0 - block.bbox.y1
            if distance < best_distance:
                best = block
                best_distance = distance

        return best

    def _find_text_nearby_matching(
        self,
        bbox: BBox,
        max_distance: float,
        pattern_labels: set[str],
    ) -> Blocks | None:
        """Find a text block nearby that might match given patterns.

        Note: This is a simplified version. In practice, we'd check
        the actual text content or hypothesis labels.
        """
        best: Blocks | None = None
        best_distance = max_distance

        for block in self._page_data.blocks:
            if not isinstance(block, Text):
                continue

            # Simple distance check (center to center)
            dx = (block.bbox.x0 + block.bbox.x1) / 2 - (bbox.x0 + bbox.x1) / 2
            dy = (block.bbox.y0 + block.bbox.y1) / 2 - (bbox.y0 + bbox.y1) / 2
            distance = (dx * dx + dy * dy) ** 0.5

            if distance < best_distance:
                # In practice, check if this text could be one of pattern_labels
                best = block
                best_distance = distance

        return best

    def _find_images_nearby(
        self,
        bbox: BBox,
        max_distance: float,
    ) -> list[Blocks]:
        """Find all images within max_distance of the bbox."""
        results: list[Blocks] = []

        for block in self._page_data.blocks:
            if not isinstance(block, Image | Drawing):
                continue

            # Skip self
            if block.bbox == bbox:
                continue

            # Check distance (edge to edge)
            dx = max(0, max(bbox.x0 - block.bbox.x1, block.bbox.x0 - bbox.x1))
            dy = max(0, max(bbox.y0 - block.bbox.y1, block.bbox.y0 - bbox.y1))
            distance = (dx * dx + dy * dy) ** 0.5

            if distance <= max_distance:
                results.append(block)

        return results
