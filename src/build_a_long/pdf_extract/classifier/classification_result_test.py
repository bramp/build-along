"""Tests for the classification result data classes."""

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
            source_blocks=[block],
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate)

        candidates = result.get_candidates("page_number")
        assert len(candidates) == 1
        assert candidates[0].score == 0.95

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
        """Test that add_candidate validates source_blocks are in PageData."""
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
            source_blocks=[block2],  # Not in PageData!
        )

        result = ClassificationResult(page_data=page_data)
        with pytest.raises(ValueError, match="must be in PageData.blocks"):
            result.add_candidate("page_number", candidate)

    def test_add_candidate_allows_source_block_in_page_data(self) -> None:
        """Test that add_candidate succeeds when source_blocks are in PageData."""
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
            source_blocks=[block],
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate)
        assert len(result.get_candidates("page_number")) == 1

    def test_add_candidate_allows_none_source_block(self) -> None:
        """Test that add_candidate allows empty source_blocks (synthetic candidates)."""
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
            source_blocks=[],  # Synthetic candidate
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("step", candidate)
        assert len(result.get_candidates("step")) == 1

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
