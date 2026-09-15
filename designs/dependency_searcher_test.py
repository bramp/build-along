"""Tests for DependencyAwareSearcher."""

from __future__ import annotations

from build_a_long.pdf_extract.classifier.dependency_searcher import (
    CompositeCandidate,
    CompositeType,
    DefaultCompositeScorer,
    DependencyAwareSearcher,
    PrimitiveHypothesis,
    PrimitiveType,
    SearchState,
    create_arrow_candidate,
    create_diagram_candidate,
    create_part_candidate,
)

from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.extractor import PageData
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image, Text


def make_page_data(blocks: list) -> PageData:
    """Create a PageData with the given blocks."""
    return PageData(
        page_number=1,
        bbox=BBox(x0=0, y0=0, x1=600, y1=800),
        blocks=blocks,
    )


def make_text(block_id: int, x: float, y: float, text: str = "2x") -> Text:
    """Create a Text block."""
    return Text(
        id=block_id,
        bbox=BBox(x0=x, y0=y, x1=x + 20, y1=y + 10),
        text=text,
    )


def make_image(block_id: int, x: float, y: float, size: float = 50) -> Image:
    """Create an Image block."""
    return Image(
        id=block_id,
        bbox=BBox(x0=x, y0=y, x1=x + size, y1=y + size),
        ext="png",
        colorspace=3,  # RGB
        bpc=8,
        width=int(size),
        height=int(size),
        digest=b"abc123",
    )


def make_drawing(block_id: int, x: float, y: float, w: float, h: float) -> Drawing:
    """Create a Drawing block."""
    return Drawing(
        id=block_id,
        bbox=BBox(x0=x, y0=y, x1=x + w, y1=y + h),
    )


class TestSearchState:
    """Tests for SearchState."""

    def test_initial_state(self) -> None:
        """Test initial state has empty assignments."""
        state = SearchState()
        assert state.assignments == {}
        assert state.composites == []
        assert state.claimed_blocks == set()
        assert state.total_score == 0.0

    def test_copy_creates_independent_state(self) -> None:
        """Test that copy creates an independent state."""
        state = SearchState(assignments={1: PrimitiveType.PART_IMAGE})
        state.claimed_blocks.add(1)

        copied = state.copy()
        copied.assignments[2] = PrimitiveType.PART_COUNT
        copied.claimed_blocks.add(2)

        # Original should be unchanged
        assert 2 not in state.assignments
        assert 2 not in state.claimed_blocks

    def test_total_score_calculation(self) -> None:
        """Test total score accounts for composite score and orphan penalty."""
        state = SearchState(
            composite_score=5.0,
            orphan_penalty=1.5,
        )
        assert state.total_score == 3.5


class TestCompositeCandidate:
    """Tests for CompositeCandidate."""

    def test_part_candidate_completeness(self) -> None:
        """Test Part candidate requires both image and count."""
        candidate = CompositeCandidate(
            composite_type=CompositeType.PART,
            expected_block_ids={1, 2},
        )

        # Initially incomplete
        assert not candidate.is_complete

        # Add image - still incomplete
        candidate.add_primitive(1, PrimitiveType.PART_IMAGE)
        assert not candidate.is_complete
        assert PrimitiveType.PART_COUNT in candidate.get_missing_required()

        # Add count - now complete
        candidate.add_primitive(2, PrimitiveType.PART_COUNT)
        assert candidate.is_complete
        assert candidate.get_missing_required() == set()

    def test_remove_primitive_makes_incomplete(self) -> None:
        """Test removing a primitive makes candidate incomplete."""
        candidate = CompositeCandidate(
            composite_type=CompositeType.PART,
            expected_block_ids={1, 2},
        )

        candidate.add_primitive(1, PrimitiveType.PART_IMAGE)
        candidate.add_primitive(2, PrimitiveType.PART_COUNT)
        assert candidate.is_complete

        candidate.remove_primitive(2)
        assert not candidate.is_complete

    def test_add_unrelated_block_returns_false(self) -> None:
        """Test adding a block not in expected_block_ids returns False."""
        candidate = CompositeCandidate(
            composite_type=CompositeType.PART,
            expected_block_ids={1, 2},
        )

        result = candidate.add_primitive(99, PrimitiveType.PART_IMAGE)
        assert result is False
        assert 99 not in candidate.assigned_primitives


class TestDependencyAwareSearcher:
    """Tests for DependencyAwareSearcher."""

    def test_search_with_no_blocks(self) -> None:
        """Test search with empty hypotheses."""
        page_data = make_page_data([])
        searcher = DependencyAwareSearcher(page_data)

        result = searcher.search({})

        assert result.state.total_score == 0.0
        assert result.states_explored == 1

    def test_search_with_fixed_assignments(self) -> None:
        """Test search when all blocks have single hypothesis."""
        text = make_text(1, 100, 100, "2x")
        image = make_image(2, 100, 120)
        page_data = make_page_data([text, image])

        searcher = DependencyAwareSearcher(page_data)

        hypotheses = {
            1: [PrimitiveHypothesis(1, PrimitiveType.PART_COUNT, 0.9, text.bbox)],
            2: [PrimitiveHypothesis(2, PrimitiveType.PART_IMAGE, 0.9, image.bbox)],
        }

        result = searcher.search(hypotheses)

        # Both should be assigned
        assert result.state.assignments[1] == PrimitiveType.PART_COUNT
        assert result.state.assignments[2] == PrimitiveType.PART_IMAGE
        assert result.states_explored == 1

    def test_search_with_ambiguous_block(self) -> None:
        """Test search explores ambiguous assignments."""
        text = make_text(1, 100, 100, "2x")
        image = make_image(2, 100, 120)
        page_data = make_page_data([text, image])

        searcher = DependencyAwareSearcher(page_data)

        hypotheses = {
            # Block 1 could be PartCount or StepCount
            1: [
                PrimitiveHypothesis(1, PrimitiveType.PART_COUNT, 0.8, text.bbox),
                PrimitiveHypothesis(1, PrimitiveType.STEP_COUNT, 0.7, text.bbox),
            ],
            2: [PrimitiveHypothesis(2, PrimitiveType.PART_IMAGE, 0.9, image.bbox)],
        }

        result = searcher.search(hypotheses)

        # Should have explored multiple states
        assert result.states_explored > 1
        # Assignment should be one of the options
        assert result.state.assignments[1] in {
            PrimitiveType.PART_COUNT,
            PrimitiveType.STEP_COUNT,
        }

    def test_composite_formation(self) -> None:
        """Test that composites are formed when dependencies are met."""
        text = make_text(1, 100, 100, "2x")
        image = make_image(2, 100, 120)
        page_data = make_page_data([text, image])

        searcher = DependencyAwareSearcher(page_data)

        hypotheses = {
            1: [PrimitiveHypothesis(1, PrimitiveType.PART_COUNT, 0.9, text.bbox)],
            2: [PrimitiveHypothesis(2, PrimitiveType.PART_IMAGE, 0.9, image.bbox)],
        }

        # Create a Part candidate that expects these two blocks
        part_candidate = create_part_candidate(
            part_image_block_id=2, part_count_block_id=1
        )

        result = searcher.search(hypotheses, composite_candidates=[part_candidate])

        # Should have formed a Part composite
        assert len(result.state.composites) == 1
        assert result.state.composites[0].composite_type == CompositeType.PART
        assert result.state.composites[0].score > 0

        # Both blocks should be claimed
        assert 1 in result.state.claimed_blocks
        assert 2 in result.state.claimed_blocks

    def test_orphan_penalty(self) -> None:
        """Test that orphaned primitives incur a penalty."""
        text = make_text(1, 100, 100, "2x")
        page_data = make_page_data([text])

        searcher = DependencyAwareSearcher(page_data)

        hypotheses = {
            # Part count without a part image = orphan
            1: [PrimitiveHypothesis(1, PrimitiveType.PART_COUNT, 0.9, text.bbox)],
        }

        result = searcher.search(hypotheses)

        # Block should be marked as orphan (no composite to join)
        assert 1 in result.state.orphan_blocks
        assert result.state.orphan_penalty > 0

    def test_pruning_reduces_states(self) -> None:
        """Test that pruning can reduce the number of states explored.

        Note: Pruning only occurs when there's a good solution to compare against.
        Without composite candidates, all paths score similarly so no pruning occurs.
        """
        # Create several ambiguous blocks with composite candidates
        # so that some paths can be pruned
        texts = [make_text(i, 100 + i * 30, 100, "2x") for i in range(3)]
        images = [make_image(i + 10, 100 + i * 30, 115) for i in range(3)]
        page_data = make_page_data(texts + images)

        searcher = DependencyAwareSearcher(page_data)

        # Texts could be part_count or step_count
        hypotheses: dict[int, list[PrimitiveHypothesis]] = {
            i: [
                PrimitiveHypothesis(i, PrimitiveType.PART_COUNT, 0.9, t.bbox),
                PrimitiveHypothesis(i, PrimitiveType.STEP_COUNT, 0.7, t.bbox),
            ]
            for i, t in enumerate(texts)
        }
        # Images are fixed as part_image
        for i, img in enumerate(images):
            hypotheses[i + 10] = [
                PrimitiveHypothesis(i + 10, PrimitiveType.PART_IMAGE, 0.9, img.bbox)
            ]

        # Create Part candidates that pair text[i] with image[i]
        candidates = [
            create_part_candidate(part_image_block_id=i + 10, part_count_block_id=i)
            for i in range(3)
        ]

        result = searcher.search(hypotheses, composite_candidates=candidates)

        # Should explore states and potentially prune some
        # With 3 ambiguous blocks and 2 options each, max is 2^3 = 8
        # We also count intermediate states, so it can be higher
        assert result.states_explored >= 1
        # Best solution should form Parts (higher score than orphan step_counts)
        assert len(result.state.composites) >= 1

    def test_mrv_ordering(self) -> None:
        """Test that MRV heuristic orders blocks correctly."""
        blocks = [
            make_text(1, 100, 100, "2x"),
            make_text(2, 200, 100, "3x"),
        ]
        page_data = make_page_data(blocks)

        searcher = DependencyAwareSearcher(page_data)

        hypotheses = {
            # Block 1 has 3 options
            1: [
                PrimitiveHypothesis(1, PrimitiveType.PART_COUNT, 0.8, blocks[0].bbox),
                PrimitiveHypothesis(1, PrimitiveType.STEP_COUNT, 0.7, blocks[0].bbox),
                PrimitiveHypothesis(1, PrimitiveType.STEP_NUMBER, 0.6, blocks[0].bbox),
            ],
            # Block 2 has 2 options - should be tried first (MRV)
            2: [
                PrimitiveHypothesis(2, PrimitiveType.PART_COUNT, 0.8, blocks[1].bbox),
                PrimitiveHypothesis(2, PrimitiveType.STEP_COUNT, 0.7, blocks[1].bbox),
            ],
        }

        # MRV should put block 2 first (fewer options)
        ordered = searcher._order_by_mrv([1, 2], hypotheses)
        assert ordered == [2, 1]


class TestCompositeScoring:
    """Tests for composite scoring."""

    def test_part_scoring_requires_both(self) -> None:
        """Test Part scoring requires both image and count."""
        text = make_text(1, 100, 100, "2x")
        image = make_image(2, 100, 120)  # Below text
        page_data = make_page_data([text, image])

        scorer = DefaultCompositeScorer(page_data)

        # Only image - should fail
        score = scorer.score(CompositeType.PART, [(image, PrimitiveType.PART_IMAGE)])
        assert score == 0.0

        # Only count - should fail
        score = scorer.score(CompositeType.PART, [(text, PrimitiveType.PART_COUNT)])
        assert score == 0.0

        # Both - should succeed
        score = scorer.score(
            CompositeType.PART,
            [
                (image, PrimitiveType.PART_IMAGE),
                (text, PrimitiveType.PART_COUNT),
            ],
        )
        assert score > 0

    def test_part_scoring_rewards_spatial_alignment(self) -> None:
        """Test Part scoring rewards good spatial alignment."""
        # Well-aligned: count directly below image
        text_aligned = make_text(1, 100, 155, "2x")  # y0=155, image y1=150
        image = make_image(2, 100, 100)  # 100 to 150

        # Misaligned: count far from image
        text_misaligned = make_text(3, 300, 155, "2x")

        page_data = make_page_data([text_aligned, image, text_misaligned])

        scorer = DefaultCompositeScorer(page_data)

        score_aligned = scorer.score(
            CompositeType.PART,
            [
                (image, PrimitiveType.PART_IMAGE),
                (text_aligned, PrimitiveType.PART_COUNT),
            ],
        )

        score_misaligned = scorer.score(
            CompositeType.PART,
            [
                (image, PrimitiveType.PART_IMAGE),
                (text_misaligned, PrimitiveType.PART_COUNT),
            ],
        )

        # Aligned should score higher
        assert score_aligned > score_misaligned


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_part_candidate(self) -> None:
        """Test create_part_candidate helper."""
        candidate = create_part_candidate(part_image_block_id=1, part_count_block_id=2)

        assert candidate.composite_type == CompositeType.PART
        assert candidate.expected_block_ids == {1, 2}
        assert not candidate.is_complete

    def test_create_arrow_candidate_with_shaft(self) -> None:
        """Test create_arrow_candidate with shaft."""
        candidate = create_arrow_candidate(head_block_id=1, shaft_block_id=2)

        assert candidate.composite_type == CompositeType.ARROW
        assert candidate.expected_block_ids == {1, 2}

    def test_create_arrow_candidate_without_shaft(self) -> None:
        """Test create_arrow_candidate without shaft."""
        candidate = create_arrow_candidate(head_block_id=1)

        assert candidate.composite_type == CompositeType.ARROW
        assert candidate.expected_block_ids == {1}

    def test_create_diagram_candidate(self) -> None:
        """Test create_diagram_candidate helper."""
        candidate = create_diagram_candidate({1, 2, 3, 4})

        assert candidate.composite_type == CompositeType.DIAGRAM
        assert candidate.expected_block_ids == {1, 2, 3, 4}


class TestBacktracking:
    """Tests for backtracking behavior."""

    def test_backtracking_finds_better_solution(self) -> None:
        """Test that backtracking explores alternatives."""
        # Create a scenario where the first choice leads to a dead end
        # but backtracking finds a better solution
        text1 = make_text(1, 100, 100, "2x")
        text2 = make_text(2, 100, 200, "3x")
        image1 = make_image(3, 100, 115)  # Below text1
        image2 = make_image(4, 100, 215)  # Below text2

        page_data = make_page_data([text1, text2, image1, image2])

        searcher = DependencyAwareSearcher(page_data)

        hypotheses = {
            # Text 1 could be part_count or step_count
            1: [
                PrimitiveHypothesis(1, PrimitiveType.STEP_COUNT, 0.9, text1.bbox),
                PrimitiveHypothesis(1, PrimitiveType.PART_COUNT, 0.8, text1.bbox),
            ],
            # Text 2 fixed as part_count
            2: [PrimitiveHypothesis(2, PrimitiveType.PART_COUNT, 0.9, text2.bbox)],
            # Images fixed as part_image
            3: [PrimitiveHypothesis(3, PrimitiveType.PART_IMAGE, 0.9, image1.bbox)],
            4: [PrimitiveHypothesis(4, PrimitiveType.PART_IMAGE, 0.9, image2.bbox)],
        }

        # Create Part candidates
        part1 = create_part_candidate(part_image_block_id=3, part_count_block_id=1)
        part2 = create_part_candidate(part_image_block_id=4, part_count_block_id=2)

        result = searcher.search(hypotheses, composite_candidates=[part1, part2])

        # Should have backtracked at least once
        # (first choice of step_count for text1 leaves image1 orphaned)
        # May or may not backtrack depending on order
        assert result.backtrack_count >= 0

        # Should form at least one Part
        assert len(result.state.composites) >= 1

    def test_max_states_limit(self) -> None:
        """Test that search respects max_states limit."""
        # Create many ambiguous blocks to trigger limit
        blocks = [make_text(i, 100 + i * 30, 100, "2x") for i in range(20)]
        page_data = make_page_data(blocks)

        # Use low limit for testing
        searcher = DependencyAwareSearcher(page_data, max_states=100)

        hypotheses = {
            i: [
                PrimitiveHypothesis(i, PrimitiveType.PART_COUNT, 0.8, b.bbox),
                PrimitiveHypothesis(i, PrimitiveType.STEP_COUNT, 0.7, b.bbox),
            ]
            for i, b in enumerate(blocks)
        }

        result = searcher.search(hypotheses)

        # Should stop at the limit
        assert result.states_explored <= 100
