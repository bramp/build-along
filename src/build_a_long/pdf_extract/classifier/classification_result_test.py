"""Tests for the classification result data classes."""

import pytest
from pydantic import ValidationError

from build_a_long.pdf_extract.classifier import (
    Candidate,
    ClassificationResult,
    ClassifierConfig,
    RemovalReason,
)
from build_a_long.pdf_extract.classifier.config import PageNumberConfig
from build_a_long.pdf_extract.classifier.test_utils import PageBuilder, TestScore
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
            PageNumberConfig(text_weight=-0.1)

    def test_weight_above_one_raises(self) -> None:
        """Test that weights above 1.0 raise a ValidationError."""
        with pytest.raises(ValidationError, match=r"less than or equal to 1"):
            PageNumberConfig(text_weight=1.5)

    def test_json_round_trip(self) -> None:
        """Test that ClassifierConfig can be serialized and deserialized."""
        config = ClassifierConfig(
            min_confidence_threshold=0.8,
            page_number=PageNumberConfig(text_weight=0.6, position_weight=0.4),
        )

        # Serialize
        json_str = config.model_dump_json()

        # Deserialize
        config2: ClassifierConfig = ClassifierConfig.model_validate_json(json_str)

        # Verify all fields match
        assert config2.min_confidence_threshold == config.min_confidence_threshold
        assert config2.page_number.text_weight == config.page_number.text_weight
        assert config2.page_number.position_weight == config.page_number.position_weight
        assert config2.page_number.position_scale == config.page_number.position_scale


class TestClassificationResult:
    """Tests for ClassificationResult."""

    def test_add_and_get_candidate(self) -> None:
        """Test adding and retrieving candidates."""
        page = (
            PageBuilder(page_number=1, width=100, height=100)
            .add_text("1", 0, 0, 10, 10, id=1)
            .build()
        )
        block = page.blocks[0]

        constructed = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        candidate = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.95,
            score_details=TestScore(),
            constructed=constructed,
            source_blocks=[block],
        )

        result = ClassificationResult(page_data=page)
        result.add_candidate(candidate)

        candidates = result.get_candidates("page_number")
        assert len(candidates) == 1
        assert candidates[0].score == 0.95

    def test_mark_and_check_removed(self) -> None:
        """Test marking blocks as removed and checking removal status."""
        page = (
            PageBuilder(page_number=1, width=100, height=100)
            .add_text("test", 0, 0, 10, 10, id=1)
            .add_text("target", 20, 20, 10, 10, id=2)
            .build()
        )
        block1 = page.blocks[0]
        block2 = page.blocks[1]

        reason = RemovalReason(reason_type="child_bbox", target_block=block2)

        result = ClassificationResult(page_data=page)
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
        # PageBuilder handles IDs automatically if not provided,
        # but here we force duplicates
        page = (
            PageBuilder(page_number=1, width=100, height=100)
            .add_text("1", 0, 0, 10, 10, id=1)
            .add_text("2", 20, 20, 10, 10, id=1)  # Duplicate ID!
            .build()
        )

        with pytest.raises(
            ValueError, match=r"must have unique IDs.*duplicates.*\{1\}"
        ):
            ClassificationResult(page_data=page)

    def test_add_candidate_validates_source_block_in_page_data(self) -> None:
        """Test that add_candidate validates source_blocks are in PageData."""
        page = (
            PageBuilder(page_number=1, width=100, height=100)
            .add_text("1", 0, 0, 10, 10, id=1)
            .build()
        )
        # Create a block not in PageData
        block2 = Text(bbox=BBox(20, 20, 30, 30), text="2", id=2)

        candidate = Candidate(
            bbox=BBox(20, 20, 30, 30),
            label="page_number",
            score=0.95,
            score_details=TestScore(),
            source_blocks=[block2],  # Not in PageData!
        )

        result = ClassificationResult(page_data=page)
        with pytest.raises(ValueError, match="must be in PageData.blocks"):
            result.add_candidate(candidate)

    def test_add_candidate_allows_source_block_in_page_data(self) -> None:
        """Test that add_candidate succeeds when source_blocks are in PageData."""
        page = (
            PageBuilder(page_number=1, width=100, height=100)
            .add_text("1", 0, 0, 10, 10, id=1)
            .build()
        )
        block = page.blocks[0]

        candidate = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.95,
            score_details=TestScore(),
            source_blocks=[block],
        )

        result = ClassificationResult(page_data=page)
        result.add_candidate(candidate)
        assert len(result.get_candidates("page_number")) == 1

    def test_add_candidate_allows_none_source_block(self) -> None:
        """Test that add_candidate allows empty source_blocks (synthetic candidates)."""
        page = PageBuilder(page_number=1, width=100, height=100).build()

        candidate = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="step",
            score=0.95,
            score_details=TestScore(),
            source_blocks=[],  # Synthetic candidate
        )

        result = ClassificationResult(page_data=page)
        result.add_candidate(candidate)
        assert len(result.get_candidates("step")) == 1

    def test_mark_removed_validates_block_in_page_data(self) -> None:
        """Test that mark_removed validates block is in PageData."""
        page = (
            PageBuilder(page_number=1, width=100, height=100)
            .add_text("1", 0, 0, 10, 10, id=1)
            .build()
        )
        block1 = page.blocks[0]
        # Create a block not in PageData
        block2 = Text(bbox=BBox(20, 20, 30, 30), text="2", id=2)

        reason = RemovalReason(reason_type="child_bbox", target_block=block1)

        result = ClassificationResult(page_data=page)
        with pytest.raises(ValueError, match="must be in PageData.blocks"):
            result.mark_removed(block2, reason)

    def test_mark_removed_allows_block_in_page_data(self) -> None:
        """Test that mark_removed succeeds when block is in PageData."""
        page = (
            PageBuilder(page_number=1, width=100, height=100)
            .add_text("1", 0, 0, 10, 10, id=1)
            .add_text("2", 20, 20, 10, 10, id=2)
            .build()
        )
        block1 = page.blocks[0]
        block2 = page.blocks[1]

        reason = RemovalReason(reason_type="child_bbox", target_block=block1)

        result = ClassificationResult(page_data=page)
        result.mark_removed(block2, reason)
        assert result.is_removed(block2)


class TestGetScoredCandidates:
    """Tests for get_scored_candidates() API.

    These tests demonstrate the recommended pattern for classifiers that depend
    on other classifiers.
    """

    def test_returns_sorted_by_score(self) -> None:
        """Test that get_scored_candidates returns candidates sorted by score."""
        page = (
            PageBuilder(page_number=1, width=100, height=100)
            .add_text("42", 0, 0, 10, 10, id=1, font_size=12.0, font_name="Arial")
            .build()
        )
        text = page.blocks[0]

        result = ClassificationResult(page_data=page)

        # Add some candidates with different scores (all valid)
        result.add_candidate(
            Candidate(
                bbox=text.bbox,
                label="test_label",
                score=0.8,
                score_details=TestScore(value=0.8),
                constructed=PageNumber(bbox=text.bbox, value=1),
                source_blocks=[text],
            ),
        )
        result.add_candidate(
            Candidate(
                bbox=text.bbox,
                label="test_label",
                score=0.3,
                score_details=TestScore(value=0.3),
                constructed=PageNumber(bbox=text.bbox, value=2),
                source_blocks=[text],
            ),
        )
        result.add_candidate(
            Candidate(
                bbox=text.bbox,
                label="test_label",
                score=0.9,
                score_details=TestScore(value=0.9),
                constructed=PageNumber(bbox=text.bbox, value=3),
                source_blocks=[text],
            ),
        )

        # Get scored candidates
        candidates = result.get_scored_candidates("test_label")

        # Should be sorted by score descending
        assert len(candidates) == 3
        assert candidates[0].score == 0.9
        assert candidates[1].score == 0.8
        assert candidates[2].score == 0.3

    def test_filters_by_min_score(self) -> None:
        """Test that get_scored_candidates filters by minimum score."""
        page = (
            PageBuilder(page_number=1, width=100, height=100)
            .add_text("42", 0, 0, 10, 10, id=1, font_size=12.0, font_name="Arial")
            .build()
        )
        text = page.blocks[0]

        result = ClassificationResult(page_data=page)

        # Add candidates with different scores (all valid)
        result.add_candidate(
            Candidate(
                bbox=text.bbox,
                label="test_label",
                score=0.8,
                score_details=TestScore(value=0.8),
                constructed=PageNumber(bbox=text.bbox, value=1),
                source_blocks=[text],
            ),
        )
        result.add_candidate(
            Candidate(
                bbox=text.bbox,
                label="test_label",
                score=0.3,
                score_details=TestScore(value=0.3),
                constructed=PageNumber(bbox=text.bbox, value=2),
                source_blocks=[text],
            ),
        )

        # Get candidates with minimum score
        candidates = result.get_scored_candidates("test_label", min_score=0.5)

        # Should only include candidates with score >= 0.5
        assert len(candidates) == 1
        assert candidates[0].score == 0.8

    def test_excludes_unscored(self) -> None:
        """Test that get_scored_candidates excludes candidates without score_details."""
        page = (
            PageBuilder(page_number=1, width=100, height=100)
            .add_text("42", 0, 0, 10, 10, id=1, font_size=12.0, font_name="Arial")
            .build()
        )
        text = page.blocks[0]

        result = ClassificationResult(page_data=page)

        # Add a candidate with score_details
        result.add_candidate(
            Candidate(
                bbox=text.bbox,
                label="test_label",
                score=0.8,
                score_details=TestScore(value=0.8),
                constructed=PageNumber(bbox=text.bbox, value=1),
                source_blocks=[text],
            ),
        )

        result.add_candidate(
            Candidate(
                bbox=text.bbox,
                label="test_label",
                score=0.9,
                score_details=TestScore(value=0.9),
                constructed=PageNumber(bbox=text.bbox, value=2),
                source_blocks=[text],
            ),
        )

        # Get scored candidates
        candidates = result.get_scored_candidates("test_label")

        # Should include all valid candidates with score_details
        assert len(candidates) == 2

    def test_valid_only_parameter(self) -> None:
        """Test get_scored_candidates with valid_only parameter.

        By default, get_scored_candidates returns only valid candidates.
        Use valid_only=False to get all scored candidates including unconstructed ones.
        """
        page = (
            PageBuilder(page_number=1, width=100, height=100)
            .add_text("42", 0, 0, 10, 10, id=1, font_size=12.0, font_name="Arial")
            .build()
        )
        text = page.blocks[0]

        result = ClassificationResult(page_data=page)

        # Add a candidate that hasn't been constructed yet
        result.add_candidate(
            Candidate(
                bbox=text.bbox,
                label="test_label",
                score=0.8,
                score_details=TestScore(value=0.8),
                source_blocks=[text],
            ),
        )

        # Get scored candidates with valid_only=False
        candidates = result.get_scored_candidates("test_label", valid_only=False)

        # Should include the candidate even though it's not constructed
        assert len(candidates) == 1
        assert candidates[0].constructed is None

        # With default valid_only=True, should return empty list
        valid_candidates = result.get_scored_candidates("test_label")
        assert len(valid_candidates) == 0
