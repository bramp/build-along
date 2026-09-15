"""
Conflict resolution for classification candidates.

This module handles the case where multiple candidates want the same blocks.
Instead of greedy "first wins", we find the globally optimal assignment.

The Problem
-----------
Consider a page with blocks A, B, C where:
- Arrow1 wants block A (score 0.9)
- Arrow2 wants block B (score 0.85)
- Diagram1 wants blocks A, B, C (score 0.8)

Greedy approach builds Arrow1 first (highest score), then Arrow2, then Diagram1
fails because A and B are taken. But maybe the best solution is:
- Build Diagram1 with all three blocks (total value 0.8)
- Skip both arrows

Or maybe:
- Build Arrow1 with A (0.9)
- Build Diagram with B, C only (0.7)
- Total value: 1.6

The Solution
------------
1. Identify contested blocks (claimed by multiple candidates)
2. For "flexible" candidates (like Diagram), generate variants excluding
   contested blocks
3. Model as weighted independent set: find non-conflicting candidates with
   maximum total score

Key Insight: Most blocks are uncontested. We only enumerate variants for
the small number of contested blocks, avoiding exponential blowup.

Usage
-----
    resolver = ConflictResolver()

    # Add all candidates
    for candidate in all_candidates:
        resolver.add_candidate(candidate)

    # Register flexible scorers for candidates that can adapt
    resolver.register_flexible_scorer("diagram", diagram_scorer_fn)

    # Resolve conflicts
    winners = resolver.resolve()
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

from build_a_long.pdf_extract.classifier.candidate import Candidate

log = logging.getLogger(__name__)


@dataclass
class CandidateVariant:
    """A specific variant of a candidate with a fixed set of blocks.

    For fixed candidates (like Arrow), there's only one variant.
    For flexible candidates (like Diagram), we generate variants
    for different subsets of contested blocks.
    """

    candidate: Candidate
    """The original candidate this variant is based on."""

    block_ids: frozenset[int]
    """The specific blocks this variant uses."""

    score: float
    """Score for this specific variant (may differ from original if blocks removed)."""

    excluded_ids: frozenset[int] = field(default_factory=frozenset)
    """Block IDs explicitly excluded from this variant."""


# Type for a function that can re-score a candidate with different blocks
FlexibleScorer = Callable[[Candidate, frozenset[int]], float | None]


class ConflictResolver:
    """Resolves conflicts between candidates competing for the same blocks.

    This class implements a clean separation between:
    1. Collecting candidates
    2. Identifying conflicts
    3. Generating variants for flexible candidates
    4. Finding the optimal non-conflicting assignment

    Example:
        resolver = ConflictResolver()

        # Add candidates (typically done by classifiers)
        resolver.add_candidate(arrow_candidate)
        resolver.add_candidate(diagram_candidate)

        # Register flexible scorers
        def diagram_scorer(candidate, block_ids):
            if len(block_ids) < 2:
                return None  # Need at least 2 blocks
            return 0.8 * len(block_ids) / len(candidate.source_blocks)

        resolver.register_flexible_scorer("diagram", diagram_scorer)

        # Resolve and get winners
        winners = resolver.resolve()
    """

    def __init__(self) -> None:
        self._candidates: list[Candidate] = []
        self._flexible_scorers: dict[str, FlexibleScorer] = {}

    def add_candidate(self, candidate: Candidate) -> None:
        """Add a candidate to be considered during resolution."""
        self._candidates.append(candidate)

    def add_candidates(self, candidates: list[Candidate]) -> None:
        """Add multiple candidates."""
        self._candidates.extend(candidates)

    def register_flexible_scorer(self, label: str, scorer: FlexibleScorer) -> None:
        """Register a scorer for candidates that can adapt to different block sets.

        The scorer function takes:
        - candidate: The original candidate
        - block_ids: The set of block IDs to use (subset of original)

        Returns:
        - A score (float) if the candidate is valid with those blocks
        - None if the candidate is invalid without those blocks
        """
        self._flexible_scorers[label] = scorer

    def resolve(self) -> list[Candidate]:
        """Find the optimal non-conflicting set of candidates.

        Returns:
            List of winning candidates (no two share a block)
        """
        if not self._candidates:
            return []

        # Step 1: Find contested blocks
        contested = self._find_contested_blocks()

        if not contested:
            # No conflicts - all candidates win
            log.debug(
                "[resolver] No contested blocks, all %d candidates win",
                len(self._candidates),
            )
            return self._candidates.copy()

        log.debug("[resolver] Found %d contested blocks: %s", len(contested), contested)

        # Step 2: Generate variants
        variants = self._generate_variants(contested)

        log.debug(
            "[resolver] Generated %d variants from %d candidates",
            len(variants),
            len(self._candidates),
        )

        # Step 3: Find best non-conflicting assignment
        winners = self._find_best_assignment(variants)

        log.debug(
            "[resolver] Selected %d winners with total score %.3f",
            len(winners),
            sum(v.score for v in winners),
        )

        # Return the original candidates (not variants)
        # For variants with excluded blocks, we need to update the candidate
        result: list[Candidate] = []
        for v in winners:
            if v.excluded_ids:
                # This variant has reduced blocks - create updated candidate
                reduced_blocks = [
                    b for b in v.candidate.source_blocks if b.id not in v.excluded_ids
                ]
                # Create a new candidate with reduced blocks and updated score
                # We copy most fields but update source_blocks and score
                result.append(
                    Candidate(
                        bbox=v.candidate.bbox,  # Keep original bbox for now
                        label=v.candidate.label,
                        score=v.score,
                        score_details=v.candidate.score_details,
                        source_blocks=reduced_blocks,
                    )
                )
            else:
                result.append(v.candidate)
        return result

    def _find_contested_blocks(self) -> set[int]:
        """Find blocks claimed by more than one candidate."""
        block_claims: dict[int, list[Candidate]] = defaultdict(list)

        for candidate in self._candidates:
            for block in candidate.source_blocks:
                block_claims[block.id].append(candidate)

        return {
            block_id
            for block_id, claimants in block_claims.items()
            if len(claimants) > 1
        }

    def _generate_variants(self, contested: set[int]) -> list[CandidateVariant]:
        """Generate candidate variants based on contested blocks.

        For fixed candidates: one variant with original blocks
        For flexible candidates: variants for each subset of contested blocks
        """
        variants: list[CandidateVariant] = []

        for candidate in self._candidates:
            candidate_block_ids = frozenset(b.id for b in candidate.source_blocks)
            candidate_contested = candidate_block_ids & contested

            if not candidate_contested:
                # No contested blocks - single variant with all blocks
                variants.append(
                    CandidateVariant(
                        candidate=candidate,
                        block_ids=candidate_block_ids,
                        score=candidate.score,
                    )
                )
                continue

            scorer = self._flexible_scorers.get(candidate.label)

            if scorer is None:
                # Fixed candidate - single variant, may conflict
                variants.append(
                    CandidateVariant(
                        candidate=candidate,
                        block_ids=candidate_block_ids,
                        score=candidate.score,
                    )
                )
                continue

            # Flexible candidate - generate variants
            # We enumerate subsets of contested blocks to exclude
            for excluded in _powerset(candidate_contested):
                remaining = candidate_block_ids - excluded

                if not remaining:
                    continue

                # Ask scorer for this variant's score
                score = scorer(candidate, remaining)

                if score is None:
                    continue  # Invalid variant

                variants.append(
                    CandidateVariant(
                        candidate=candidate,
                        block_ids=remaining,
                        score=score,
                        excluded_ids=excluded,
                    )
                )

                log.debug(
                    "[resolver] Generated variant for %s: %d blocks, score=%.3f, "
                    "excluded=%s",
                    candidate.label,
                    len(remaining),
                    score,
                    excluded,
                )

        return variants

    def _find_best_assignment(
        self, variants: list[CandidateVariant]
    ) -> list[CandidateVariant]:
        """Find the maximum-weight independent set of variants.

        No two selected variants can share a block.

        For now, uses a greedy approximation (fast, usually good).
        Could be upgraded to exact algorithm for small conflict graphs.
        """
        # Sort by score descending
        sorted_variants = sorted(variants, key=lambda v: -v.score)

        selected: list[CandidateVariant] = []
        used_blocks: set[int] = set()
        used_candidates: set[int] = set()  # Track by id() to handle variants

        for variant in sorted_variants:
            # Skip if any block already used
            if variant.block_ids & used_blocks:
                continue

            # Skip if we already selected a variant of this candidate
            # (don't want both "Diagram with A,B,C" and "Diagram with B,C")
            candidate_id = id(variant.candidate)
            if candidate_id in used_candidates:
                continue

            # Select this variant
            selected.append(variant)
            used_blocks.update(variant.block_ids)
            used_candidates.add(candidate_id)

        return selected


def _powerset(s: frozenset[int]) -> list[frozenset[int]]:
    """Generate all subsets of a set, from empty to full.

    For small sets (which contested blocks should be), this is fine.
    """
    result: list[frozenset[int]] = [frozenset()]
    for elem in s:
        result = result + [subset | {elem} for subset in result]
    return result
