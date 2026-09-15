"""Tests for the conflict resolver."""

from build_a_long.pdf_extract.classifier.conflict_resolver import (
    ConflictResolver,
    _powerset,
)

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.score import Score
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Blocks, Drawing


class _SimpleScore(Score):
    """Simple score for testing."""

    value: float

    def score(self) -> float:
        return self.value


def _make_block(block_id: int) -> Drawing:
    """Create a test Drawing block with a specific ID."""
    return Drawing(
        id=block_id,
        bbox=BBox(x0=0, y0=0, x1=10, y1=10),
        items=(),
    )


def _make_candidate(
    label: str,
    score: float,
    block_ids: list[int],
) -> Candidate:
    """Create a test candidate with specific blocks."""
    blocks: list[Blocks] = [_make_block(bid) for bid in block_ids]
    return Candidate(
        bbox=BBox(x0=0, y0=0, x1=100, y1=100),
        label=label,
        score=score,
        score_details=_SimpleScore(value=score),
        source_blocks=blocks,
    )


class TestPowerset:
    """Tests for the powerset helper function."""

    def test_empty_set(self) -> None:
        result = _powerset(frozenset())
        assert result == [frozenset()]

    def test_single_element(self) -> None:
        result = _powerset(frozenset({1}))
        assert set(result) == {frozenset(), frozenset({1})}

    def test_two_elements(self) -> None:
        result = _powerset(frozenset({1, 2}))
        expected = {
            frozenset(),
            frozenset({1}),
            frozenset({2}),
            frozenset({1, 2}),
        }
        assert set(result) == expected

    def test_three_elements(self) -> None:
        result = _powerset(frozenset({1, 2, 3}))
        assert len(result) == 8  # 2^3


class TestConflictResolverNoConflicts:
    """Tests for cases with no conflicts."""

    def test_empty_candidates(self) -> None:
        resolver = ConflictResolver()
        winners = resolver.resolve()
        assert winners == []

    def test_single_candidate(self) -> None:
        resolver = ConflictResolver()
        candidate = _make_candidate("arrow", 0.9, [1])
        resolver.add_candidate(candidate)

        winners = resolver.resolve()

        assert len(winners) == 1
        assert winners[0] is candidate

    def test_multiple_non_conflicting(self) -> None:
        """Multiple candidates with different blocks all win."""
        resolver = ConflictResolver()
        c1 = _make_candidate("arrow", 0.9, [1])
        c2 = _make_candidate("arrow", 0.8, [2])
        c3 = _make_candidate("diagram", 0.7, [3, 4])

        resolver.add_candidate(c1)
        resolver.add_candidate(c2)
        resolver.add_candidate(c3)

        winners = resolver.resolve()

        assert len(winners) == 3
        assert c1 in winners
        assert c2 in winners
        assert c3 in winners


class TestConflictResolverWithConflicts:
    """Tests for conflict resolution."""

    def test_simple_conflict_higher_wins(self) -> None:
        """When two candidates want the same block, higher score wins."""
        resolver = ConflictResolver()
        c1 = _make_candidate("arrow", 0.9, [1])
        c2 = _make_candidate("arrow", 0.7, [1])

        resolver.add_candidate(c1)
        resolver.add_candidate(c2)

        winners = resolver.resolve()

        assert len(winners) == 1
        assert winners[0] is c1

    def test_conflict_total_score_matters(self) -> None:
        """Greedy picks highest individual score first.

        In this case:
        - c1 (0.9) uses block 1
        - c2 (0.8) uses blocks 1, 2

        Greedy picks c1 first, then c2 conflicts.
        Result: only c1 (total 0.9)

        Note: optimal would be c2 (0.8), but greedy doesn't find it.
        """
        resolver = ConflictResolver()
        c1 = _make_candidate("arrow", 0.9, [1])
        c2 = _make_candidate("diagram", 0.8, [1, 2])

        resolver.add_candidate(c1)
        resolver.add_candidate(c2)

        winners = resolver.resolve()

        # Greedy picks highest score first
        assert len(winners) == 1
        assert winners[0] is c1

    def test_partial_conflict(self) -> None:
        """When candidates partially overlap, higher wins."""
        resolver = ConflictResolver()
        c1 = _make_candidate("arrow", 0.9, [1])
        c2 = _make_candidate("diagram", 0.7, [1, 2, 3])
        c3 = _make_candidate("arrow", 0.8, [4])

        resolver.add_candidate(c1)
        resolver.add_candidate(c2)
        resolver.add_candidate(c3)

        winners = resolver.resolve()

        # c1 wins over c2 (block 1 conflict), c3 has no conflict
        assert len(winners) == 2
        assert c1 in winners
        assert c3 in winners
        assert c2 not in winners


class TestConflictResolverFlexible:
    """Tests for flexible candidate scoring."""

    def test_flexible_generates_variants(self) -> None:
        """Flexible candidates generate variants for contested blocks."""
        resolver = ConflictResolver()

        c_arrow = _make_candidate("arrow", 0.9, [1])
        c_diagram = _make_candidate("diagram", 0.8, [1, 2, 3])

        resolver.add_candidate(c_arrow)
        resolver.add_candidate(c_diagram)

        # Register flexible scorer for diagram
        def diagram_scorer(
            candidate: Candidate, block_ids: frozenset[int]
        ) -> float | None:
            # Score proportional to number of blocks
            if len(block_ids) < 2:
                return None  # Need at least 2 blocks
            return 0.8 * len(block_ids) / 3  # Scale by fraction of original

        resolver.register_flexible_scorer("diagram", diagram_scorer)

        winners = resolver.resolve()

        # Now we should get both:
        # - Arrow with block 1 (score 0.9)
        # - Diagram with blocks 2, 3 (score ~0.53)
        assert len(winners) == 2
        assert c_arrow in winners

        # Find the diagram winner
        diagram_winners = [w for w in winners if w.label == "diagram"]
        assert len(diagram_winners) == 1
        diagram_winner = diagram_winners[0]

        # It should have reduced blocks (only 2 and 3, not 1)
        diagram_block_ids = {b.id for b in diagram_winner.source_blocks}
        assert diagram_block_ids == {2, 3}

    def test_flexible_chooses_best_variant(self) -> None:
        """Resolver picks the best variant when multiple are possible."""
        resolver = ConflictResolver()

        # Two arrows competing for two different blocks in a diagram
        c_arrow1 = _make_candidate("arrow", 0.5, [1])
        c_arrow2 = _make_candidate("arrow", 0.5, [2])
        c_diagram = _make_candidate("diagram", 1.0, [1, 2, 3, 4])

        resolver.add_candidate(c_arrow1)
        resolver.add_candidate(c_arrow2)
        resolver.add_candidate(c_diagram)

        # Flexible scorer: need at least 3 blocks, score = 0.25 per block
        def diagram_scorer(
            candidate: Candidate, block_ids: frozenset[int]
        ) -> float | None:
            if len(block_ids) < 3:
                return None
            return 0.25 * len(block_ids)

        resolver.register_flexible_scorer("diagram", diagram_scorer)

        winners = resolver.resolve()

        # Diagram with all 4 blocks (score 1.0) beats arrows (0.5 + 0.5)
        # So diagram should win
        assert len(winners) == 1
        # The winner is the diagram (may be a new candidate if blocks were reduced)
        assert winners[0].label == "diagram"
        # Should have all 4 blocks since none were contested by higher-scoring winners
        assert len(winners[0].source_blocks) == 4

    def test_flexible_invalid_variant_skipped(self) -> None:
        """Variants that return None from scorer are skipped."""
        resolver = ConflictResolver()

        c_arrow = _make_candidate("arrow", 0.9, [1, 2])  # Takes both blocks
        c_diagram = _make_candidate("diagram", 0.8, [1, 2, 3])

        resolver.add_candidate(c_arrow)
        resolver.add_candidate(c_diagram)

        # Diagram needs all 3 blocks to be valid
        def diagram_scorer(
            candidate: Candidate, block_ids: frozenset[int]
        ) -> float | None:
            if len(block_ids) < 3:
                return None  # Invalid
            return 0.8

        resolver.register_flexible_scorer("diagram", diagram_scorer)

        winners = resolver.resolve()

        # Arrow wins (0.9), diagram can't form valid variant without blocks 1,2
        assert len(winners) == 1
        assert c_arrow in winners
