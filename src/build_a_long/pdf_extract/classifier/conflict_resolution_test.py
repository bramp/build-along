"""Tests for conflict resolution."""

from build_a_long.pdf_extract.classifier.classification_result import (
    Candidate,
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.conflict_resolution import (
    resolve_label_conflicts,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    BagNumber,
    PageNumber,
    PartCount,
    PieceLength,
    StepNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import Text


class TestConflictResolution:
    """Tests for label conflict resolution."""

    def test_no_conflicts(self) -> None:
        """Test that resolution doesn't affect non-conflicting candidates."""
        page_data = PageData(
            page_number=1,
            blocks=[
                Text(id=0, bbox=BBox(0, 0, 10, 10), text="1"),
                Text(id=1, bbox=BBox(20, 20, 30, 30), text="2"),
            ],
            bbox=BBox(0, 0, 100, 100),
        )

        result = ClassificationResult(page_data=page_data)

        # Add non-conflicting candidates (different blocks)
        block1 = page_data.blocks[0]
        block2 = page_data.blocks[1]

        candidate1 = Candidate(
            bbox=block1.bbox,
            label="page_number",
            score=0.9,
            score_details=None,
            constructed=PageNumber(value=1, bbox=block1.bbox),
            source_blocks=[block1],
            failure_reason=None,
        )

        candidate2 = Candidate(
            bbox=block2.bbox,
            label="step_number",
            score=0.8,
            score_details=None,
            constructed=StepNumber(value=2, bbox=block2.bbox),
            source_blocks=[block2],
            failure_reason=None,
        )

        result.add_candidate("page_number", candidate1)
        result.add_candidate("step_number", candidate2)

        # Resolve conflicts
        resolve_label_conflicts(result)

        # Both should remain successful (no conflicts)
        assert candidate1.failure_reason is None
        assert candidate2.failure_reason is None
        assert candidate1.constructed is not None
        assert candidate2.constructed is not None

    def test_page_number_beats_step_number(self) -> None:
        """Test that page_number wins over step_number for the same block."""
        page_data = PageData(
            page_number=1,
            blocks=[Text(id=0, bbox=BBox(0, 0, 10, 10), text="1")],
            bbox=BBox(0, 0, 100, 100),
        )

        result = ClassificationResult(page_data=page_data)
        block = page_data.blocks[0]

        # Add conflicting candidates (same block, different labels)
        page_num_candidate = Candidate(
            bbox=block.bbox,
            label="page_number",
            score=0.9,
            score_details=None,
            constructed=PageNumber(value=1, bbox=block.bbox),
            source_blocks=[block],
            failure_reason=None,
        )

        step_num_candidate = Candidate(
            bbox=block.bbox,
            label="step_number",
            score=0.8,
            score_details=None,
            constructed=StepNumber(value=1, bbox=block.bbox),
            source_blocks=[block],
            failure_reason=None,
        )

        result.add_candidate("page_number", page_num_candidate)
        result.add_candidate("step_number", step_num_candidate)

        # Resolve conflicts
        resolve_label_conflicts(result)

        # page_number should win
        assert page_num_candidate.failure_reason is None
        assert step_num_candidate.failure_reason is not None
        assert "page_number" in step_num_candidate.failure_reason

    def test_piece_length_beats_step_number(self) -> None:
        """Test that piece_length wins over step_number (circled numbers)."""
        page_data = PageData(
            page_number=1,
            blocks=[Text(id=0, bbox=BBox(0, 0, 10, 10), text="4")],
            bbox=BBox(0, 0, 100, 100),
        )

        result = ClassificationResult(page_data=page_data)
        block = page_data.blocks[0]

        # A "4" in a circle could be piece length or step number
        piece_length_candidate = Candidate(
            bbox=block.bbox,
            label="piece_length",
            score=0.85,
            score_details=None,
            constructed=PieceLength(value=4, bbox=block.bbox),
            source_blocks=[block],
            failure_reason=None,
        )

        step_num_candidate = Candidate(
            bbox=block.bbox,
            label="step_number",
            score=0.80,
            score_details=None,
            constructed=StepNumber(value=4, bbox=block.bbox),
            source_blocks=[block],
            failure_reason=None,
        )

        result.add_candidate("piece_length", piece_length_candidate)
        result.add_candidate("step_number", step_num_candidate)

        # Resolve conflicts
        resolve_label_conflicts(result)

        # piece_length should win
        assert piece_length_candidate.failure_reason is None
        assert step_num_candidate.failure_reason is not None
        assert "piece_length" in step_num_candidate.failure_reason

    def test_part_count_beats_step_number(self) -> None:
        """Test that part_count wins over step_number in parts lists."""
        page_data = PageData(
            page_number=1,
            blocks=[Text(id=0, bbox=BBox(0, 0, 10, 10), text="2x")],
            bbox=BBox(0, 0, 100, 100),
        )

        result = ClassificationResult(page_data=page_data)
        block = page_data.blocks[0]

        part_count_candidate = Candidate(
            bbox=block.bbox,
            label="part_count",
            score=0.9,
            score_details=None,
            constructed=PartCount(count=2, bbox=block.bbox),
            source_blocks=[block],
            failure_reason=None,
        )

        step_num_candidate = Candidate(
            bbox=block.bbox,
            label="step_number",
            score=0.7,
            score_details=None,
            constructed=StepNumber(value=2, bbox=block.bbox),
            source_blocks=[block],
            failure_reason=None,
        )

        result.add_candidate("part_count", part_count_candidate)
        result.add_candidate("step_number", step_num_candidate)

        # Resolve conflicts
        resolve_label_conflicts(result)

        # part_count should win
        assert part_count_candidate.failure_reason is None
        assert step_num_candidate.failure_reason is not None
        assert "part_count" in step_num_candidate.failure_reason

    def test_bag_number_beats_step_number(self) -> None:
        """Test that bag_number wins over step_number."""
        page_data = PageData(
            page_number=1,
            blocks=[Text(id=0, bbox=BBox(0, 0, 10, 10), text="BAG 2")],
            bbox=BBox(0, 0, 100, 100),
        )

        result = ClassificationResult(page_data=page_data)
        block = page_data.blocks[0]

        bag_num_candidate = Candidate(
            bbox=block.bbox,
            label="bag_number",
            score=0.95,
            score_details=None,
            constructed=BagNumber(value=2, bbox=block.bbox),
            source_blocks=[block],
            failure_reason=None,
        )

        step_num_candidate = Candidate(
            bbox=block.bbox,
            label="step_number",
            score=0.6,
            score_details=None,
            constructed=StepNumber(value=2, bbox=block.bbox),
            source_blocks=[block],
            failure_reason=None,
        )

        result.add_candidate("bag_number", bag_num_candidate)
        result.add_candidate("step_number", step_num_candidate)

        # Resolve conflicts
        resolve_label_conflicts(result)

        # bag_number should win
        assert bag_num_candidate.failure_reason is None
        assert step_num_candidate.failure_reason is not None
        assert "bag_number" in step_num_candidate.failure_reason

    def test_priority_fallback(self) -> None:
        """Test that priority system works when no specific rule applies."""
        page_data = PageData(
            page_number=1,
            blocks=[Text(id=0, bbox=BBox(0, 0, 10, 10), text="5")],
            bbox=BBox(0, 0, 100, 100),
        )

        result = ClassificationResult(page_data=page_data)
        block = page_data.blocks[0]

        # Two labels without a specific conflict rule
        # page_number has higher priority than bag_number
        page_num_candidate = Candidate(
            bbox=block.bbox,
            label="page_number",
            score=0.7,
            score_details=None,
            constructed=PageNumber(value=5, bbox=block.bbox),
            source_blocks=[block],
            failure_reason=None,
        )

        bag_num_candidate = Candidate(
            bbox=block.bbox,
            label="bag_number",
            score=0.8,  # Higher score but lower priority
            score_details=None,
            constructed=BagNumber(value=5, bbox=block.bbox),
            source_blocks=[block],
            failure_reason=None,
        )

        result.add_candidate("page_number", page_num_candidate)
        result.add_candidate("bag_number", bag_num_candidate)

        # Resolve conflicts
        resolve_label_conflicts(result)

        # page_number should win (higher priority)
        assert page_num_candidate.failure_reason is None
        assert bag_num_candidate.failure_reason is not None
        assert "priority" in bag_num_candidate.failure_reason

    def test_ignores_failed_candidates(self) -> None:
        """Test that already-failed candidates are not considered in conflicts."""
        page_data = PageData(
            page_number=1,
            blocks=[Text(id=0, bbox=BBox(0, 0, 10, 10), text="1")],
            bbox=BBox(0, 0, 100, 100),
        )

        result = ClassificationResult(page_data=page_data)
        block = page_data.blocks[0]

        # One successful candidate
        page_num_candidate = Candidate(
            bbox=block.bbox,
            label="page_number",
            score=0.9,
            score_details=None,
            constructed=PageNumber(value=1, bbox=block.bbox),
            source_blocks=[block],
            failure_reason=None,
        )

        # One already-failed candidate (constructed=None)
        step_num_candidate = Candidate(
            bbox=block.bbox,
            label="step_number",
            score=0.8,
            score_details=None,
            constructed=None,  # Already failed
            source_blocks=[block],
            failure_reason="Some earlier failure",
        )

        result.add_candidate("page_number", page_num_candidate)
        result.add_candidate("step_number", step_num_candidate)

        # Resolve conflicts
        resolve_label_conflicts(result)

        # page_number should remain successful
        # step_number should keep its original failure reason
        assert page_num_candidate.failure_reason is None
        assert step_num_candidate.failure_reason == "Some earlier failure"

    def test_multi_way_conflict(self) -> None:
        """Test resolution when more than 2 labels conflict."""
        page_data = PageData(
            page_number=1,
            blocks=[Text(id=0, bbox=BBox(0, 0, 10, 10), text="2")],
            bbox=BBox(0, 0, 100, 100),
        )

        result = ClassificationResult(page_data=page_data)
        block = page_data.blocks[0]

        # Three different labels for the same block
        page_num_candidate = Candidate(
            bbox=block.bbox,
            label="page_number",
            score=0.7,
            score_details=None,
            constructed=PageNumber(value=2, bbox=block.bbox),
            source_blocks=[block],
            failure_reason=None,
        )

        step_num_candidate = Candidate(
            bbox=block.bbox,
            label="step_number",
            score=0.8,
            score_details=None,
            constructed=StepNumber(value=2, bbox=block.bbox),
            source_blocks=[block],
            failure_reason=None,
        )

        part_count_candidate = Candidate(
            bbox=block.bbox,
            label="part_count",
            score=0.75,
            score_details=None,
            constructed=PartCount(count=2, bbox=block.bbox),
            source_blocks=[block],
            failure_reason=None,
        )

        result.add_candidate("page_number", page_num_candidate)
        result.add_candidate("step_number", step_num_candidate)
        result.add_candidate("part_count", part_count_candidate)

        # Resolve conflicts
        resolve_label_conflicts(result)

        # page_number should win (highest priority)
        assert page_num_candidate.failure_reason is None
        assert step_num_candidate.failure_reason is not None
        assert part_count_candidate.failure_reason is not None

    def test_multiple_blocks_with_conflicts(self) -> None:
        """Test that resolution handles multiple conflicting blocks independently."""
        page_data = PageData(
            page_number=1,
            blocks=[
                Text(id=0, bbox=BBox(0, 0, 10, 10), text="1"),
                Text(id=1, bbox=BBox(20, 20, 30, 30), text="2"),
            ],
            bbox=BBox(0, 0, 100, 100),
        )

        result = ClassificationResult(page_data=page_data)
        block1 = page_data.blocks[0]
        block2 = page_data.blocks[1]

        # Block 1: page_number vs step_number
        page_num_1 = Candidate(
            bbox=block1.bbox,
            label="page_number",
            score=0.9,
            score_details=None,
            constructed=PageNumber(value=1, bbox=block1.bbox),
            source_blocks=[block1],
            failure_reason=None,
        )
        step_num_1 = Candidate(
            bbox=block1.bbox,
            label="step_number",
            score=0.8,
            score_details=None,
            constructed=StepNumber(value=1, bbox=block1.bbox),
            source_blocks=[block1],
            failure_reason=None,
        )

        # Block 2: piece_length vs step_number
        piece_length_2 = Candidate(
            bbox=block2.bbox,
            label="piece_length",
            score=0.85,
            score_details=None,
            constructed=PieceLength(value=2, bbox=block2.bbox),
            source_blocks=[block2],
            failure_reason=None,
        )
        step_num_2 = Candidate(
            bbox=block2.bbox,
            label="step_number",
            score=0.75,
            score_details=None,
            constructed=StepNumber(value=2, bbox=block2.bbox),
            source_blocks=[block2],
            failure_reason=None,
        )

        result.add_candidate("page_number", page_num_1)
        result.add_candidate("step_number", step_num_1)
        result.add_candidate("piece_length", piece_length_2)
        result.add_candidate("step_number", step_num_2)

        # Resolve conflicts
        resolve_label_conflicts(result)

        # Block 1: page_number wins
        assert page_num_1.failure_reason is None
        assert step_num_1.failure_reason is not None

        # Block 2: piece_length wins
        assert piece_length_2.failure_reason is None
        assert step_num_2.failure_reason is not None
