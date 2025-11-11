"""Tests for the classification result data classes."""

import json

import pytest
from pydantic import ValidationError

from build_a_long.pdf_extract.classifier.classification_result import (
    Candidate,
    ClassificationResult,
    ClassifierConfig,
    RemovalReason,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    PageNumber,
    StepNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import Text


class TestClassifierConfig:
    """Tests for ClassifierConfig."""

    def test_negative_weight_raises(self) -> None:
        """Test that negative weights raise a ValidationError."""
        with pytest.raises(ValidationError, match=r"greater than or equal to 0"):
            ClassifierConfig(page_number_text_weight=-0.1)

    def test_weight_above_one_raises(self) -> None:
        """Test that weights above 1.0 raise a ValidationError."""
        with pytest.raises(ValidationError, match=r"less than or equal to 1"):
            ClassifierConfig(page_number_text_weight=1.5)

    def test_json_round_trip(self) -> None:
        """Test that ClassifierConfig can be serialized and deserialized."""
        config = ClassifierConfig(
            min_confidence_threshold=0.8,
            page_number_text_weight=0.6,
            page_number_position_weight=0.4,
        )

        # Serialize
        json_str = config.model_dump_json()

        # Deserialize
        config2: ClassifierConfig = ClassifierConfig.model_validate_json(json_str)

        # Verify all fields match
        assert config2.min_confidence_threshold == config.min_confidence_threshold
        assert config2.page_number_text_weight == config.page_number_text_weight
        assert config2.page_number_position_weight == config.page_number_position_weight
        assert config2.page_number_position_scale == config.page_number_position_scale


class TestClassificationResult:
    """Tests for ClassificationResult."""

    def test_add_and_get_warnings(self) -> None:
        """Test adding and retrieving warnings."""
        page_data = PageData(
            page_number=1,
            blocks=[],
            bbox=BBox(0, 0, 100, 100),
        )
        result = ClassificationResult(page_data=page_data)
        result.add_warning("Warning 1")
        result.add_warning("Warning 2")

        warnings = result.get_warnings()
        assert len(warnings) == 2
        assert "Warning 1" in warnings
        assert "Warning 2" in warnings

    def test_add_and_get_candidate(self) -> None:
        """Test adding and retrieving candidates."""
        block = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        page_data = PageData(
            page_number=1,
            blocks=[block],
            bbox=BBox(0, 0, 100, 100),
        )
        constructed = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        candidate = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=constructed,
            source_block=block,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate)

        candidates = result.get_candidates("page_number")
        assert len(candidates) == 1
        assert candidates[0].score == 0.95

    def test_mark_winner(self) -> None:
        """Test marking a candidate as winner."""
        block = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        page_data = PageData(
            page_number=1,
            blocks=[block],
            bbox=BBox(0, 0, 100, 100),
        )
        constructed = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        candidate = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=constructed,
            source_block=block,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate)
        result.mark_winner(candidate, constructed)

        assert candidate.is_winner is True
        assert result.get_constructed_element(block) is constructed
        assert result.has_label("page_number")

    def test_constructed_elements_dict(self) -> None:
        """Test the internal _constructed_elements dict."""
        block1 = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        block2 = Text(bbox=BBox(20, 20, 30, 30), text="Step 2", id=2)
        page_data = PageData(
            page_number=1,
            blocks=[block1, block2],
            bbox=BBox(0, 0, 100, 100),
        )

        constructed1 = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        constructed2 = StepNumber(bbox=BBox(20, 20, 30, 30), value=2)

        candidate1 = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=constructed1,
            source_block=block1,
        )
        candidate2 = Candidate(
            bbox=BBox(20, 20, 30, 30),
            label="step_number",
            score=0.90,
            score_details={},
            constructed=constructed2,
            source_block=block2,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate1)
        result.add_candidate("step_number", candidate2)
        result.mark_winner(candidate1, constructed1)
        result.mark_winner(candidate2, constructed2)

        # Access the internal dict directly (keyed by block ID)
        assert len(result.constructed_elements) == 2
        assert result.constructed_elements[1] is constructed1
        assert result.constructed_elements[2] is constructed2

    def test_get_label(self) -> None:
        """Test getting the label for a specific block."""
        block = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        page_data = PageData(
            page_number=1,
            blocks=[block],
            bbox=BBox(0, 0, 100, 100),
        )
        constructed = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        candidate = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=constructed,
            source_block=block,
            is_winner=True,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate)

        assert result.get_label(block) == "page_number"

    def test_get_blocks_by_label(self) -> None:
        """Test getting blocks by label."""
        block1 = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        block2 = Text(bbox=BBox(20, 20, 30, 30), text="5", id=2)
        page_data = PageData(
            page_number=1,
            blocks=[block1, block2],
            bbox=BBox(0, 0, 100, 100),
        )
        constructed1 = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        constructed2 = PageNumber(bbox=BBox(20, 20, 30, 30), value=5)

        candidate1 = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=constructed1,
            source_block=block1,
            is_winner=True,
        )
        candidate2 = Candidate(
            bbox=BBox(20, 20, 30, 30),
            label="page_number",
            score=0.90,
            score_details={},
            constructed=constructed2,
            source_block=block2,
            is_winner=True,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate1)
        result.add_candidate("page_number", candidate2)

        page_numbers = result.get_blocks_by_label("page_number")
        assert len(page_numbers) == 2
        assert block1 in page_numbers
        assert block2 in page_numbers

    def test_mark_and_check_removed(self) -> None:
        """Test marking blocks as removed and checking removal status."""
        block1 = Text(bbox=BBox(0, 0, 10, 10), text="test", id=1)
        block2 = Text(bbox=BBox(20, 20, 30, 30), text="target", id=2)
        page_data = PageData(
            page_number=1,
            blocks=[block1, block2],
            bbox=BBox(0, 0, 100, 100),
        )
        reason = RemovalReason(reason_type="child_bbox", target_block=block2)

        result = ClassificationResult(page_data=page_data)
        result.mark_removed(block1, reason)

        assert result.is_removed(block1) is True
        assert result.is_removed(block2) is False

        retrieved_reason = result.get_removal_reason(block1)
        assert retrieved_reason is not None
        assert retrieved_reason.reason_type == "child_bbox"
        assert retrieved_reason.target_block is block2

    def test_has_label(self) -> None:
        """Test checking if a label has been assigned."""
        block = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        page_data = PageData(
            page_number=1,
            blocks=[block],
            bbox=BBox(0, 0, 100, 100),
        )
        constructed = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        candidate = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=constructed,
            source_block=block,
            is_winner=True,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate)

        assert result.has_label("page_number") is True
        assert result.has_label("step_number") is False


class TestClassificationResultValidation:
    """Tests for ClassificationResult validation logic."""

    def test_post_init_validates_unique_block_ids(self) -> None:
        """Test that __post_init__ validates PageData blocks have unique IDs."""
        block1 = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        block2 = Text(bbox=BBox(20, 20, 30, 30), text="2", id=1)  # Duplicate ID!

        page_data = PageData(
            page_number=1,
            blocks=[block1, block2],
            bbox=BBox(0, 0, 100, 100),
        )

        with pytest.raises(
            ValueError, match=r"must have unique IDs.*duplicates.*\{1\}"
        ):
            ClassificationResult(page_data=page_data)

    def test_add_candidate_validates_source_block_in_page_data(self) -> None:
        """Test that add_candidate validates source_block is in PageData."""
        block1 = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        block2 = Text(bbox=BBox(20, 20, 30, 30), text="2", id=2)  # Not in PageData

        page_data = PageData(
            page_number=1,
            blocks=[block1],
            bbox=BBox(0, 0, 100, 100),
        )

        candidate = Candidate(
            bbox=BBox(20, 20, 30, 30),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=None,
            source_block=block2,  # Not in PageData!
        )

        result = ClassificationResult(page_data=page_data)
        with pytest.raises(ValueError, match="must be in PageData.blocks"):
            result.add_candidate("page_number", candidate)

    def test_add_candidate_allows_source_block_in_page_data(self) -> None:
        """Test that add_candidate succeeds when source_block is in PageData."""
        block = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)

        page_data = PageData(
            page_number=1,
            blocks=[block],
            bbox=BBox(0, 0, 100, 100),
        )

        candidate = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=None,
            source_block=block,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate)
        assert len(result.get_candidates("page_number")) == 1

    def test_add_candidate_allows_none_source_block(self) -> None:
        """Test that add_candidate allows None source_block (synthetic candidates)."""
        page_data = PageData(
            page_number=1,
            blocks=[],
            bbox=BBox(0, 0, 100, 100),
        )

        candidate = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="step",
            score=0.95,
            score_details={},
            constructed=None,
            source_block=None,  # Synthetic candidate
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("step", candidate)
        assert len(result.get_candidates("step")) == 1

    def test_mark_winner_validates_source_block_in_page_data(self) -> None:
        """Test that mark_winner validates source_block is in PageData."""
        block1 = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        block2 = Text(bbox=BBox(20, 20, 30, 30), text="2", id=2)  # Not in PageData

        page_data = PageData(
            page_number=1,
            blocks=[block1],
            bbox=BBox(0, 0, 100, 100),
        )

        constructed = PageNumber(bbox=BBox(20, 20, 30, 30), value=2)
        candidate = Candidate(
            bbox=BBox(20, 20, 30, 30),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=constructed,
            source_block=block2,  # Not in PageData!
        )

        result = ClassificationResult(page_data=page_data)
        with pytest.raises(ValueError, match="must be in PageData.blocks"):
            result.mark_winner(candidate, constructed)

    def test_mark_winner_allows_source_block_in_page_data(self) -> None:
        """Test that mark_winner succeeds when source_block is in PageData."""
        block = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)

        page_data = PageData(
            page_number=1,
            blocks=[block],
            bbox=BBox(0, 0, 100, 100),
        )

        constructed = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        candidate = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=constructed,
            source_block=block,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate)
        result.mark_winner(candidate, constructed)
        assert candidate.is_winner is True

    def test_mark_winner_allows_none_source_block(self) -> None:
        """Test that mark_winner allows None source_block (synthetic candidates)."""
        page_data = PageData(
            page_number=1,
            blocks=[],
            bbox=BBox(0, 0, 100, 100),
        )

        # Use a synthetic PageNumber without a source_block (e.g., derived from context)
        constructed = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        candidate = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=constructed,
            source_block=None,  # Synthetic candidate
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate)
        result.mark_winner(candidate, constructed)
        assert candidate.is_winner is True

    def test_mark_winner_rejects_duplicate_winners_for_same_block(self) -> None:
        """Test that mark_winner rejects marking the same block as winner twice."""
        block = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)

        page_data = PageData(
            page_number=1,
            blocks=[block],
            bbox=BBox(0, 0, 100, 100),
        )

        # First winner candidate
        constructed1 = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        candidate1 = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=constructed1,
            source_block=block,
        )

        # Second winner candidate with different label, same block
        constructed2 = StepNumber(bbox=BBox(0, 0, 10, 10), value=1)
        candidate2 = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="step_number",
            score=0.90,
            score_details={},
            constructed=constructed2,
            source_block=block,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate1)
        result.add_candidate("step_number", candidate2)

        # First mark_winner should succeed
        result.mark_winner(candidate1, constructed1)
        assert candidate1.is_winner is True

        # Second mark_winner should fail
        with pytest.raises(
            ValueError,
            match="Block 1 already has a winner candidate for label 'page_number'",
        ):
            result.mark_winner(candidate2, constructed2)

        # Verify only the first candidate is a winner
        assert candidate1.is_winner is True
        assert candidate2.is_winner is False

    def test_mark_removed_validates_block_in_page_data(self) -> None:
        """Test that mark_removed validates block is in PageData."""
        block1 = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        block2 = Text(bbox=BBox(20, 20, 30, 30), text="2", id=2)  # Not in PageData

        page_data = PageData(
            page_number=1,
            blocks=[block1],
            bbox=BBox(0, 0, 100, 100),
        )

        reason = RemovalReason(reason_type="child_bbox", target_block=block1)

        result = ClassificationResult(page_data=page_data)
        with pytest.raises(ValueError, match="must be in PageData.blocks"):
            result.mark_removed(block2, reason)

    def test_mark_removed_allows_block_in_page_data(self) -> None:
        """Test that mark_removed succeeds when block is in PageData."""
        block1 = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        block2 = Text(bbox=BBox(20, 20, 30, 30), text="2", id=2)

        page_data = PageData(
            page_number=1,
            blocks=[block1, block2],
            bbox=BBox(0, 0, 100, 100),
        )

        reason = RemovalReason(reason_type="child_bbox", target_block=block1)

        result = ClassificationResult(page_data=page_data)
        result.mark_removed(block2, reason)
        assert result.is_removed(block2)


class TestGetWinners:
    """Tests for the get_winners method."""

    def test_get_winners_filters_correctly(self) -> None:
        """Test that get_winners returns only winning candidates of the correct type."""
        block1 = Text(bbox=BBox(10, 10, 20, 20), text="1", id=1)
        block2 = Text(bbox=BBox(20, 20, 30, 30), text="2", id=2)
        block3 = Text(bbox=BBox(30, 30, 40, 40), text="3", id=3)

        page_data = PageData(
            page_number=1,
            blocks=[block1, block2, block3],
            bbox=BBox(0, 0, 100, 100),
        )

        result = ClassificationResult(page_data=page_data)

        # Add candidates
        page_num1 = PageNumber(bbox=block1.bbox, value=1)
        page_num2 = PageNumber(bbox=block2.bbox, value=2)
        step_num = StepNumber(bbox=block3.bbox, value=1)

        candidate1 = Candidate(
            bbox=block1.bbox,
            label="page_number",
            score=0.9,
            score_details={},
            constructed=page_num1,
            source_block=block1,
        )

        candidate2 = Candidate(
            bbox=block2.bbox,
            label="page_number",
            score=0.8,
            score_details={},
            constructed=page_num2,
            source_block=block2,
        )

        candidate3 = Candidate(
            bbox=block3.bbox,
            label="step_number",
            score=0.95,
            score_details={},
            constructed=step_num,
            source_block=block3,
        )

        result.add_candidate("page_number", candidate1)
        result.add_candidate("page_number", candidate2)
        result.add_candidate("step_number", candidate3)

        # Mark only first page_number and step_number as winners
        result.mark_winner(candidate1, page_num1)
        result.mark_winner(candidate3, step_num)

        # Get winners with type safety
        page_number_winners = result.get_winners("page_number", PageNumber)
        step_number_winners = result.get_winners("step_number", StepNumber)

        # Verify correct filtering
        assert len(page_number_winners) == 1
        assert page_number_winners[0] == page_num1
        assert page_number_winners[0].value == 1

        assert len(step_number_winners) == 1
        assert step_number_winners[0] == step_num
        assert step_number_winners[0].value == 1

        # Non-winners should not be returned
        assert page_num2 not in page_number_winners

    def test_get_winners_empty_list_when_no_winners(self) -> None:
        """Test that get_winners returns empty list when there are no winners."""
        block1 = Text(bbox=BBox(10, 10, 20, 20), text="1", id=1)

        page_data = PageData(
            page_number=1,
            blocks=[block1],
            bbox=BBox(0, 0, 100, 100),
        )

        result = ClassificationResult(page_data=page_data)

        # Add candidate but don't mark as winner
        page_num = PageNumber(bbox=block1.bbox, value=1)
        result.add_candidate(
            "page_number",
            Candidate(
                bbox=block1.bbox,
                label="page_number",
                score=0.9,
                score_details={},
                constructed=page_num,
                source_block=block1,
            ),
        )

        # Should return empty list
        winners = result.get_winners("page_number", PageNumber)
        assert winners == []

    def test_get_winners_empty_list_when_no_candidates(self) -> None:
        """Test that get_winners returns empty list when there are no candidates."""
        page_data = PageData(
            page_number=1,
            blocks=[],
            bbox=BBox(0, 0, 100, 100),
        )

        result = ClassificationResult(page_data=page_data)

        # Should return empty list when no candidates exist
        winners = result.get_winners("page_number", PageNumber)
        assert winners == []

    def test_get_winners_asserts_on_none_constructed(self) -> None:
        """Test that get_winners asserts when a winner has None constructed."""
        block1 = Text(bbox=BBox(10, 10, 20, 20), text="1", id=1)

        page_data = PageData(
            page_number=1,
            blocks=[block1],
            bbox=BBox(0, 0, 100, 100),
        )

        result = ClassificationResult(page_data=page_data)

        # Add a candidate with None constructed but mark as winner (invalid!)
        candidate = Candidate(
            bbox=block1.bbox,
            label="page_number",
            score=0.9,
            score_details={},
            constructed=None,  # Invalid for a winner!
            source_block=block1,
            is_winner=True,  # This combination is invalid
        )

        result.add_candidate("page_number", candidate)

        # Should assert because winner has None constructed
        with pytest.raises(AssertionError, match="has None constructed"):
            result.get_winners("page_number", PageNumber)

    def test_get_winners_asserts_on_type_mismatch(self) -> None:
        """Test that get_winners asserts when element_type doesn't match."""
        block1 = Text(bbox=BBox(10, 10, 20, 20), text="1", id=1)

        page_data = PageData(
            page_number=1,
            blocks=[block1],
            bbox=BBox(0, 0, 100, 100),
        )

        result = ClassificationResult(page_data=page_data)

        # Add a PageNumber candidate
        page_num = PageNumber(bbox=block1.bbox, value=1)
        candidate = Candidate(
            bbox=block1.bbox,
            label="page_number",
            score=0.9,
            score_details={},
            constructed=page_num,
            source_block=block1,
        )

        result.add_candidate("page_number", candidate)
        result.mark_winner(candidate, page_num)

        # Should assert when requesting wrong type (StepNumber instead of PageNumber)
        with pytest.raises(AssertionError, match="Type mismatch"):
            result.get_winners("page_number", StepNumber)
