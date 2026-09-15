"""Tests for the page searcher."""

from build_a_long.pdf_extract.classifier.page_searcher import (
    BlockHypothesis,
    PageInterpretation,
    PageSearcher,
)

from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.extractor import PageData
from build_a_long.pdf_extract.extractor.page_blocks import (
    Blocks,
    Drawing,
    Image,
    Text,
)


def _make_text(block_id: int, text: str, x: float, y: float) -> Text:
    """Create a test Text block."""
    return Text(
        id=block_id,
        bbox=BBox(x0=x, y0=y, x1=x + 20, y1=y + 10),
        text=text,
        font_name="Arial",
        font_size=10,
    )


def _make_image(block_id: int, x: float, y: float, size: float = 30) -> Image:
    """Create a test Image block."""
    return Image(
        id=block_id,
        bbox=BBox(x0=x, y0=y, x1=x + size, y1=y + size),
        width=int(size),
        height=int(size),
        xref=block_id,
    )


def _make_drawing(block_id: int, x: float, y: float, size: float = 10) -> Drawing:
    """Create a test Drawing block."""
    return Drawing(
        id=block_id,
        bbox=BBox(x0=x, y0=y, x1=x + size, y1=y + size),
        items=(),
    )


class TestPageInterpretation:
    """Tests for PageInterpretation dataclass."""

    def test_copy(self) -> None:
        interp = PageInterpretation(assignments={1: "a", 2: "b"}, score=1.5)
        copy = interp.copy()

        assert copy.assignments == {1: "a", 2: "b"}
        assert copy.score == 1.5

        # Modify original, copy should be unchanged
        interp.assignments[3] = "c"
        assert 3 not in copy.assignments

    def test_with_assignment(self) -> None:
        interp = PageInterpretation(assignments={1: "a"}, score=1.0)
        new_interp = interp.with_assignment(2, "b")

        # Original unchanged
        assert interp.assignments == {1: "a"}

        # New has both
        assert new_interp.assignments == {1: "a", 2: "b"}


class TestPageSearcherBasic:
    """Basic tests for PageSearcher."""

    def test_no_hypotheses(self) -> None:
        """Empty hypotheses returns empty interpretation."""
        page_data = PageData(
            page_number=1,
            bbox=BBox(x0=0, y0=0, x1=100, y1=100),
            blocks=[],
        )
        searcher = PageSearcher(page_data)
        result = searcher.search({})

        assert result.interpretation.assignments == {}
        assert result.fixed_count == 0
        assert result.ambiguous_count == 0

    def test_single_unambiguous_block(self) -> None:
        """Single hypothesis means fixed assignment."""
        text = _make_text(1, "2x", 10, 10)
        page_data = PageData(
            page_number=1,
            bbox=BBox(x0=0, y0=0, x1=100, y1=100),
            blocks=[text],
        )

        hypotheses = {
            1: [BlockHypothesis(block_id=1, label="part_count", score=0.9)],
        }

        searcher = PageSearcher(page_data)
        result = searcher.search(hypotheses)

        assert result.interpretation.assignments[1] == "part_count"
        assert result.fixed_count == 1
        assert result.ambiguous_count == 0

    def test_single_ambiguous_block(self) -> None:
        """Two close hypotheses means ambiguous - search needed."""
        text = _make_text(1, "2x", 10, 10)
        page_data = PageData(
            page_number=1,
            bbox=BBox(x0=0, y0=0, x1=100, y1=100),
            blocks=[text],
        )

        hypotheses = {
            1: [
                BlockHypothesis(block_id=1, label="part_count", score=0.9),
                BlockHypothesis(block_id=1, label="step_multiplier", score=0.85),
            ],
        }

        searcher = PageSearcher(page_data)
        result = searcher.search(hypotheses)

        # Should pick one (probably part_count due to higher base score)
        assert result.interpretation.assignments[1] in {"part_count", "step_multiplier"}
        assert result.fixed_count == 0
        assert result.ambiguous_count == 1

    def test_unambiguous_due_to_score_gap(self) -> None:
        """Large score gap means not ambiguous."""
        text = _make_text(1, "2x", 10, 10)
        page_data = PageData(
            page_number=1,
            bbox=BBox(x0=0, y0=0, x1=100, y1=100),
            blocks=[text],
        )

        hypotheses = {
            1: [
                BlockHypothesis(block_id=1, label="part_count", score=0.9),
                BlockHypothesis(block_id=1, label="step_multiplier", score=0.5),
            ],
        }

        searcher = PageSearcher(page_data)
        result = searcher.search(hypotheses)

        # Second option is below threshold (0.5 < 0.9 * 0.7 = 0.63)
        assert result.interpretation.assignments[1] == "part_count"
        assert result.fixed_count == 1
        assert result.ambiguous_count == 0


class TestPageSearcherContextScoring:
    """Tests for context-aware scoring."""

    def test_part_count_with_image_below(self) -> None:
        """part_count should be preferred when there's an image below."""
        text = _make_text(1, "2x", 10, 10)
        image = _make_image(2, 10, 25, size=30)  # Below text

        page_data = PageData(
            page_number=1,
            bbox=BBox(x0=0, y0=0, x1=100, y1=100),
            blocks=[text, image],
        )

        hypotheses = {
            1: [
                BlockHypothesis(block_id=1, label="part_count", score=0.8),
                BlockHypothesis(block_id=1, label="step_multiplier", score=0.8),
            ],
            2: [
                BlockHypothesis(block_id=2, label="part_image", score=0.8),
                BlockHypothesis(block_id=2, label="diagram", score=0.8),
            ],
        }

        searcher = PageSearcher(page_data)
        result = searcher.search(hypotheses)

        # Should pick part_count + part_image due to mutual support
        assert result.interpretation.assignments[1] == "part_count"
        assert result.interpretation.assignments[2] == "part_image"

    def test_step_multiplier_without_image(self) -> None:
        """step_multiplier should be preferred when no image nearby."""
        text = _make_text(1, "2x", 10, 10)
        step_num = _make_text(2, "1", 50, 10)  # Nearby text (step number)

        page_data = PageData(
            page_number=1,
            bbox=BBox(x0=0, y0=0, x1=100, y1=100),
            blocks=[text, step_num],
        )

        hypotheses = {
            1: [
                BlockHypothesis(block_id=1, label="part_count", score=0.75),
                BlockHypothesis(block_id=1, label="step_multiplier", score=0.8),
            ],
            2: [
                BlockHypothesis(block_id=2, label="step_number", score=0.9),
            ],
        }

        searcher = PageSearcher(page_data)
        result = searcher.search(hypotheses)

        # step_multiplier has higher base score and no image to support part_count
        assert result.interpretation.assignments[1] == "step_multiplier"
        assert result.interpretation.assignments[2] == "step_number"

    def test_mutual_exclusion(self) -> None:
        """Image can't be both part_image and diagram."""
        text = _make_text(1, "2x", 10, 10)
        image = _make_image(2, 10, 25, size=30)

        page_data = PageData(
            page_number=1,
            bbox=BBox(x0=0, y0=0, x1=100, y1=100),
            blocks=[text, image],
        )

        hypotheses = {
            1: [
                BlockHypothesis(block_id=1, label="part_count", score=0.8),
            ],
            2: [
                BlockHypothesis(block_id=2, label="part_image", score=0.7),
                BlockHypothesis(block_id=2, label="diagram", score=0.75),
            ],
        }

        searcher = PageSearcher(page_data)
        result = searcher.search(hypotheses)

        # With part_count above, part_image gets context bonus
        # Should beat diagram despite lower base score
        assert result.interpretation.assignments[2] == "part_image"


class TestPageSearcherPerformance:
    """Tests for search performance and pruning."""

    def test_many_ambiguous_blocks_completes(self) -> None:
        """Search should complete even with many ambiguous blocks."""
        blocks: list[Blocks] = [
            _make_text(i, f"{i}x", i * 30, 10) for i in range(1, 11)
        ]

        page_data = PageData(
            page_number=1,
            bbox=BBox(x0=0, y0=0, x1=500, y1=100),
            blocks=blocks,
        )

        # All blocks are ambiguous
        hypotheses = {
            b.id: [
                BlockHypothesis(block_id=b.id, label="part_count", score=0.8),
                BlockHypothesis(block_id=b.id, label="step_multiplier", score=0.75),
            ]
            for b in blocks
        }

        searcher = PageSearcher(page_data)
        result = searcher.search(hypotheses)

        # Should complete and assign all blocks
        assert len(result.interpretation.assignments) == 10
        assert result.ambiguous_count == 10

        # With beam search, shouldn't evaluate all 2^10 = 1024 possibilities
        # Beam width 50 means at most 10 * 2 * 50 = 1000 evaluations
        assert result.interpretations_evaluated < 2000

    def test_beam_search_finds_good_solution(self) -> None:
        """Beam search should find a reasonable (if not optimal) solution."""
        # Create a scenario where mutual support matters
        text1 = _make_text(1, "2x", 10, 10)
        image1 = _make_image(2, 10, 25, size=30)
        text2 = _make_text(3, "3x", 100, 10)
        image2 = _make_image(4, 100, 25, size=30)

        page_data = PageData(
            page_number=1,
            bbox=BBox(x0=0, y0=0, x1=200, y1=100),
            blocks=[text1, image1, text2, image2],
        )

        hypotheses = {
            1: [
                BlockHypothesis(block_id=1, label="part_count", score=0.8),
                BlockHypothesis(block_id=1, label="step_multiplier", score=0.75),
            ],
            2: [
                BlockHypothesis(block_id=2, label="part_image", score=0.8),
                BlockHypothesis(block_id=2, label="diagram", score=0.8),
            ],
            3: [
                BlockHypothesis(block_id=3, label="part_count", score=0.8),
                BlockHypothesis(block_id=3, label="step_multiplier", score=0.75),
            ],
            4: [
                BlockHypothesis(block_id=4, label="part_image", score=0.8),
                BlockHypothesis(block_id=4, label="diagram", score=0.8),
            ],
        }

        searcher = PageSearcher(page_data)
        result = searcher.search(hypotheses)

        # Both pairs should be part_count + part_image due to mutual support
        assert result.interpretation.assignments[1] == "part_count"
        assert result.interpretation.assignments[2] == "part_image"
        assert result.interpretation.assignments[3] == "part_count"
        assert result.interpretation.assignments[4] == "part_image"
