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

    def test_get_labeled_blocks(self) -> None:
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
            is_winner=True,
        )
        candidate2 = Candidate(
            bbox=BBox(20, 20, 30, 30),
            label="step_number",
            score=0.90,
            score_details={},
            constructed=constructed2,
            source_block=block2,
            is_winner=True,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate1)
        result.add_candidate("step_number", candidate2)

        labeled = result.get_labeled_blocks()
        assert len(labeled) == 2
        assert labeled[block1] == "page_number"
        assert labeled[block2] == "step_number"

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

    def test_get_scores_for_label(self) -> None:
        """Test getting scores for a specific label."""
        block = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        page_data = PageData(
            page_number=1,
            blocks=[block],
            bbox=BBox(0, 0, 100, 100),
        )
        score_details = {"text": 1.0, "position": 0.9}
        constructed = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        candidate = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.95,
            score_details=score_details,
            constructed=constructed,
            source_block=block,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate)

        scores = result.get_scores_for_label("page_number")
        assert block in scores
        assert scores[block] == score_details

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

    def test_get_best_candidate(self) -> None:
        """Test getting the best candidate for a label."""
        block1 = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        block2 = Text(bbox=BBox(20, 20, 30, 30), text="2", id=2)
        page_data = PageData(
            page_number=1,
            blocks=[block1, block2],
            bbox=BBox(0, 0, 100, 100),
        )

        constructed1 = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        constructed2 = PageNumber(bbox=BBox(20, 20, 30, 30), value=2)

        candidate1 = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.80,
            score_details={},
            constructed=constructed1,
            source_block=block1,
        )
        candidate2 = Candidate(
            bbox=BBox(20, 20, 30, 30),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=constructed2,
            source_block=block2,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate1)
        result.add_candidate("page_number", candidate2)

        best = result.get_best_candidate("page_number")
        assert best is not None
        assert best.score == 0.95
        assert best.source_block is block2

    def test_get_best_candidate_excludes_failed(self) -> None:
        """Test that get_best_candidate excludes failed constructions."""
        block1 = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        block2 = Text(bbox=BBox(20, 20, 30, 30), text="invalid", id=2)
        page_data = PageData(
            page_number=1,
            blocks=[block1, block2],
            bbox=BBox(0, 0, 100, 100),
        )

        constructed1 = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)

        candidate1 = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.80,
            score_details={},
            constructed=constructed1,
            source_block=block1,
        )
        # Higher score but failed construction
        candidate2 = Candidate(
            bbox=BBox(20, 20, 30, 30),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=None,
            source_block=block2,
            failure_reason="Invalid format",
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate1)
        result.add_candidate("page_number", candidate2)

        best = result.get_best_candidate("page_number")
        assert best is not None
        assert best.score == 0.80
        assert best.source_block is block1

    def test_get_alternative_candidates(self) -> None:
        """Test getting alternative candidates excluding the winner."""
        block1 = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        block2 = Text(bbox=BBox(20, 20, 30, 30), text="2", id=2)
        block3 = Text(bbox=BBox(40, 40, 50, 50), text="3", id=3)
        page_data = PageData(
            page_number=1,
            blocks=[block1, block2, block3],
            bbox=BBox(0, 0, 100, 100),
        )

        constructed1 = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        constructed2 = PageNumber(bbox=BBox(20, 20, 30, 30), value=2)
        constructed3 = PageNumber(bbox=BBox(40, 40, 50, 50), value=3)

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
            score=0.80,
            score_details={},
            constructed=constructed2,
            source_block=block2,
        )
        candidate3 = Candidate(
            bbox=BBox(40, 40, 50, 50),
            label="page_number",
            score=0.70,
            score_details={},
            constructed=constructed3,
            source_block=block3,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate1)
        result.add_candidate("page_number", candidate2)
        result.add_candidate("page_number", candidate3)

        alternatives = result.get_alternative_candidates("page_number")
        assert len(alternatives) == 2
        # Should be sorted by score descending
        assert alternatives[0].score == 0.80
        assert alternatives[1].score == 0.70

    def test_constructed_elements_dict_uses_ids(self) -> None:
        """Test that _constructed_elements dict uses block IDs as keys."""
        # This is the key change to make ClassificationResult JSON serializable
        block = Text(bbox=BBox(0, 0, 10, 10), text="1", id=42)
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

        # Verify the internal _constructed_elements uses integer IDs as keys
        assert isinstance(result.constructed_elements, dict)
        assert 42 in result.constructed_elements
        assert result.constructed_elements[42] is constructed

        # Verify the public API still works correctly with block objects
        assert result.get_constructed_element(block) is constructed

    def test_internal_dict_is_json_serializable(self) -> None:
        """Test that the internal dict uses JSON-serializable keys.

        Note: Full ClassificationResult.model_dump_json() does not work due
        to the page_data field having a forward reference (PageData is in
        TYPE_CHECKING), but the _constructed_elements dict itself is now
        JSON-serializable since it uses int keys instead of block objects.
        """
        # Create a result with constructed blocks
        page_data = PageData(
            page_number=1,
            blocks=[],
            bbox=BBox(0, 0, 100, 100),
        )
        result = ClassificationResult(page_data=page_data)
        result.add_warning("Warning 1")
        result.add_warning("Warning 2")

        # Simulate what happens when blocks are marked as winners
        constructed1 = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        constructed2 = StepNumber(bbox=BBox(20, 20, 30, 30), value=2)

        # Manually populate the dict with integer keys (as mark_winner does)
        result.constructed_elements[42] = constructed1
        result.constructed_elements[99] = constructed2

        # Verify the dict itself is JSON serializable
        # This is the key improvement - before it used block objects as keys
        constructed_dict_json = json.dumps(
            {k: v.model_dump() for k, v in result.constructed_elements.items()}
        )

        assert constructed_dict_json is not None
        parsed = json.loads(constructed_dict_json)
        assert "42" in parsed  # JSON converts int keys to strings
        assert "99" in parsed
        assert parsed["42"]["value"] == 1
        assert parsed["99"]["value"] == 2


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
