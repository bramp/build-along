"""Tests for the classification result data classes."""

import json

import pytest

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
from build_a_long.pdf_extract.extractor.page_elements import Text


class TestClassifierConfig:
    """Tests for ClassifierConfig."""

    def test_negative_weight_raises(self) -> None:
        """Test that negative weights raise a ValueError."""
        with pytest.raises(
            ValueError, match="All weights must be greater than or equal to 0"
        ):
            ClassifierConfig(page_number_text_weight=-0.1)

    def test_json_round_trip(self) -> None:
        """Test that ClassifierConfig can be serialized and deserialized."""
        config = ClassifierConfig(
            min_confidence_threshold=0.8,
            page_number_text_weight=0.6,
            page_number_position_weight=0.4,
        )

        # Serialize
        json_str = config.to_json()

        # Deserialize
        config2: ClassifierConfig = ClassifierConfig.from_json(json_str)  # type: ignore[assignment]

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
            elements=[],
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
        element = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        page_data = PageData(
            page_number=1,
            elements=[element],
            bbox=BBox(0, 0, 100, 100),
        )
        constructed = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        candidate = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=constructed,
            source_element=element,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate)

        candidates = result.get_candidates("page_number")
        assert len(candidates) == 1
        assert candidates[0].score == 0.95

    def test_mark_winner(self) -> None:
        """Test marking a candidate as winner."""
        element = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        page_data = PageData(
            page_number=1,
            elements=[element],
            bbox=BBox(0, 0, 100, 100),
        )
        constructed = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        candidate = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=constructed,
            source_element=element,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate)
        result.mark_winner(candidate, constructed)

        assert candidate.is_winner is True
        assert result.get_constructed_element(element) is constructed
        assert result.has_label("page_number")

    def test_constructed_elements_dict(self) -> None:
        """Test the internal _constructed_elements dict."""
        element1 = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        element2 = Text(bbox=BBox(20, 20, 30, 30), text="Step 2", id=2)
        page_data = PageData(
            page_number=1,
            elements=[element1, element2],
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
            source_element=element1,
        )
        candidate2 = Candidate(
            bbox=BBox(20, 20, 30, 30),
            label="step_number",
            score=0.90,
            score_details={},
            constructed=constructed2,
            source_element=element2,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate1)
        result.add_candidate("step_number", candidate2)
        result.mark_winner(candidate1, constructed1)
        result.mark_winner(candidate2, constructed2)

        # Access the internal dict directly (keyed by element ID)
        assert len(result._constructed_elements) == 2
        assert result._constructed_elements[1] is constructed1
        assert result._constructed_elements[2] is constructed2

    def test_get_labeled_elements(self) -> None:
        element1 = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        element2 = Text(bbox=BBox(20, 20, 30, 30), text="Step 2", id=2)
        page_data = PageData(
            page_number=1,
            elements=[element1, element2],
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
            source_element=element1,
            is_winner=True,
        )
        candidate2 = Candidate(
            bbox=BBox(20, 20, 30, 30),
            label="step_number",
            score=0.90,
            score_details={},
            constructed=constructed2,
            source_element=element2,
            is_winner=True,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate1)
        result.add_candidate("step_number", candidate2)

        labeled = result.get_labeled_elements()
        assert len(labeled) == 2
        assert labeled[element1] == "page_number"
        assert labeled[element2] == "step_number"

    def test_get_label(self) -> None:
        """Test getting the label for a specific element."""
        element = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        page_data = PageData(
            page_number=1,
            elements=[element],
            bbox=BBox(0, 0, 100, 100),
        )
        constructed = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        candidate = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=constructed,
            source_element=element,
            is_winner=True,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate)

        assert result.get_label(element) == "page_number"

    def test_get_elements_by_label(self) -> None:
        """Test getting elements by label."""
        element1 = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        element2 = Text(bbox=BBox(20, 20, 30, 30), text="5", id=2)
        page_data = PageData(
            page_number=1,
            elements=[element1, element2],
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
            source_element=element1,
            is_winner=True,
        )
        candidate2 = Candidate(
            bbox=BBox(20, 20, 30, 30),
            label="page_number",
            score=0.90,
            score_details={},
            constructed=constructed2,
            source_element=element2,
            is_winner=True,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate1)
        result.add_candidate("page_number", candidate2)

        page_numbers = result.get_elements_by_label("page_number")
        assert len(page_numbers) == 2
        assert element1 in page_numbers
        assert element2 in page_numbers

    def test_mark_and_check_removed(self) -> None:
        """Test marking elements as removed and checking removal status."""
        element1 = Text(bbox=BBox(0, 0, 10, 10), text="test", id=1)
        element2 = Text(bbox=BBox(20, 20, 30, 30), text="target", id=2)
        page_data = PageData(
            page_number=1,
            elements=[element1, element2],
            bbox=BBox(0, 0, 100, 100),
        )
        reason = RemovalReason(reason_type="child_bbox", target_element=element2)

        result = ClassificationResult(page_data=page_data)
        result.mark_removed(element1, reason)

        assert result.is_removed(element1) is True
        assert result.is_removed(element2) is False

        retrieved_reason = result.get_removal_reason(element1)
        assert retrieved_reason is not None
        assert retrieved_reason.reason_type == "child_bbox"
        assert retrieved_reason.target_element is element2

    def test_get_scores_for_label(self) -> None:
        """Test getting scores for a specific label."""
        element = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        page_data = PageData(
            page_number=1,
            elements=[element],
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
            source_element=element,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate)

        scores = result.get_scores_for_label("page_number")
        assert element in scores
        assert scores[element] == score_details

    def test_has_label(self) -> None:
        """Test checking if a label has been assigned."""
        element = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        page_data = PageData(
            page_number=1,
            elements=[element],
            bbox=BBox(0, 0, 100, 100),
        )
        constructed = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        candidate = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=constructed,
            source_element=element,
            is_winner=True,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate)

        assert result.has_label("page_number") is True
        assert result.has_label("step_number") is False

    def test_get_best_candidate(self) -> None:
        """Test getting the best candidate for a label."""
        element1 = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        element2 = Text(bbox=BBox(20, 20, 30, 30), text="2", id=2)
        page_data = PageData(
            page_number=1,
            elements=[element1, element2],
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
            source_element=element1,
        )
        candidate2 = Candidate(
            bbox=BBox(20, 20, 30, 30),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=constructed2,
            source_element=element2,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate1)
        result.add_candidate("page_number", candidate2)

        best = result.get_best_candidate("page_number")
        assert best is not None
        assert best.score == 0.95
        assert best.source_element is element2

    def test_get_best_candidate_excludes_failed(self) -> None:
        """Test that get_best_candidate excludes failed constructions."""
        element1 = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        element2 = Text(bbox=BBox(20, 20, 30, 30), text="invalid", id=2)
        page_data = PageData(
            page_number=1,
            elements=[element1, element2],
            bbox=BBox(0, 0, 100, 100),
        )

        constructed1 = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)

        candidate1 = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.80,
            score_details={},
            constructed=constructed1,
            source_element=element1,
        )
        # Higher score but failed construction
        candidate2 = Candidate(
            bbox=BBox(20, 20, 30, 30),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=None,
            source_element=element2,
            failure_reason="Invalid format",
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate1)
        result.add_candidate("page_number", candidate2)

        best = result.get_best_candidate("page_number")
        assert best is not None
        assert best.score == 0.80
        assert best.source_element is element1

    def test_get_alternative_candidates(self) -> None:
        """Test getting alternative candidates excluding the winner."""
        element1 = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        element2 = Text(bbox=BBox(20, 20, 30, 30), text="2", id=2)
        element3 = Text(bbox=BBox(40, 40, 50, 50), text="3", id=3)
        page_data = PageData(
            page_number=1,
            elements=[element1, element2, element3],
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
            source_element=element1,
            is_winner=True,
        )
        candidate2 = Candidate(
            bbox=BBox(20, 20, 30, 30),
            label="page_number",
            score=0.80,
            score_details={},
            constructed=constructed2,
            source_element=element2,
        )
        candidate3 = Candidate(
            bbox=BBox(40, 40, 50, 50),
            label="page_number",
            score=0.70,
            score_details={},
            constructed=constructed3,
            source_element=element3,
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
        """Test that _constructed_elements dict uses element IDs as keys."""
        # This is the key change to make ClassificationResult JSON serializable
        element = Text(bbox=BBox(0, 0, 10, 10), text="1", id=42)
        page_data = PageData(
            page_number=1,
            elements=[element],
            bbox=BBox(0, 0, 100, 100),
        )

        constructed = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        candidate = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=constructed,
            source_element=element,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate)
        result.mark_winner(candidate, constructed)

        # Verify the internal _constructed_elements uses integer IDs as keys
        assert isinstance(result._constructed_elements, dict)
        assert 42 in result._constructed_elements
        assert result._constructed_elements[42] is constructed

        # Verify the public API still works correctly with Element objects
        assert result.get_constructed_element(element) is constructed

    def test_internal_dict_is_json_serializable(self) -> None:
        """Test that the internal _constructed_elements dict uses JSON-serializable keys.

        Note: Full ClassificationResult.to_json() doesn't work due to the page_data
        field having a forward reference (PageData is in TYPE_CHECKING), but the
        _constructed_elements dict itself is now JSON-serializable since it uses
        int keys instead of Element objects.
        """
        # Create a result with constructed elements
        page_data = PageData(
            page_number=1,
            elements=[],
            bbox=BBox(0, 0, 100, 100),
        )
        result = ClassificationResult(page_data=page_data)
        result.add_warning("Warning 1")
        result.add_warning("Warning 2")

        # Simulate what happens when elements are marked as winners
        constructed1 = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        constructed2 = StepNumber(bbox=BBox(20, 20, 30, 30), value=2)

        # Manually populate the dict with integer keys (as mark_winner does)
        result._constructed_elements[42] = constructed1
        result._constructed_elements[99] = constructed2

        # Verify the dict itself is JSON serializable
        # This is the key improvement - before it used Element objects as keys
        constructed_dict_json = json.dumps(
            {k: v.to_dict() for k, v in result._constructed_elements.items()}
        )

        assert constructed_dict_json is not None
        parsed = json.loads(constructed_dict_json)
        assert "42" in parsed  # JSON converts int keys to strings
        assert "99" in parsed
        assert parsed["42"]["value"] == 1
        assert parsed["99"]["value"] == 2


class TestClassificationResultValidation:
    """Tests for ClassificationResult validation logic."""

    def test_post_init_validates_unique_element_ids(self) -> None:
        """Test that __post_init__ validates PageData elements have unique IDs."""
        element1 = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        element2 = Text(bbox=BBox(20, 20, 30, 30), text="2", id=1)  # Duplicate ID!

        page_data = PageData(
            page_number=1,
            elements=[element1, element2],
            bbox=BBox(0, 0, 100, 100),
        )

        with pytest.raises(
            ValueError, match=r"must have unique IDs.*duplicates.*\{1\}"
        ):
            ClassificationResult(page_data=page_data)

    def test_add_candidate_validates_source_element_in_page_data(self) -> None:
        """Test that add_candidate validates source_element is in PageData."""
        element1 = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        element2 = Text(bbox=BBox(20, 20, 30, 30), text="2", id=2)  # Not in PageData

        page_data = PageData(
            page_number=1,
            elements=[element1],
            bbox=BBox(0, 0, 100, 100),
        )

        candidate = Candidate(
            bbox=BBox(20, 20, 30, 30),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=None,
            source_element=element2,  # Not in PageData!
        )

        result = ClassificationResult(page_data=page_data)
        with pytest.raises(ValueError, match="must be in PageData.elements"):
            result.add_candidate("page_number", candidate)

    def test_add_candidate_allows_source_element_in_page_data(self) -> None:
        """Test that add_candidate succeeds when source_element is in PageData."""
        element = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)

        page_data = PageData(
            page_number=1,
            elements=[element],
            bbox=BBox(0, 0, 100, 100),
        )

        candidate = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=None,
            source_element=element,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate)
        assert len(result.get_candidates("page_number")) == 1

    def test_add_candidate_allows_none_source_element(self) -> None:
        """Test that add_candidate allows None source_element (synthetic candidates)."""
        page_data = PageData(
            page_number=1,
            elements=[],
            bbox=BBox(0, 0, 100, 100),
        )

        candidate = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="step",
            score=0.95,
            score_details={},
            constructed=None,
            source_element=None,  # Synthetic candidate
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("step", candidate)
        assert len(result.get_candidates("step")) == 1

    def test_mark_winner_validates_source_element_in_page_data(self) -> None:
        """Test that mark_winner validates source_element is in PageData."""
        element1 = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        element2 = Text(bbox=BBox(20, 20, 30, 30), text="2", id=2)  # Not in PageData

        page_data = PageData(
            page_number=1,
            elements=[element1],
            bbox=BBox(0, 0, 100, 100),
        )

        constructed = PageNumber(bbox=BBox(20, 20, 30, 30), value=2)
        candidate = Candidate(
            bbox=BBox(20, 20, 30, 30),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=constructed,
            source_element=element2,  # Not in PageData!
        )

        result = ClassificationResult(page_data=page_data)
        with pytest.raises(ValueError, match="must be in PageData.elements"):
            result.mark_winner(candidate, constructed)

    def test_mark_winner_allows_source_element_in_page_data(self) -> None:
        """Test that mark_winner succeeds when source_element is in PageData."""
        element = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)

        page_data = PageData(
            page_number=1,
            elements=[element],
            bbox=BBox(0, 0, 100, 100),
        )

        constructed = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        candidate = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=constructed,
            source_element=element,
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate)
        result.mark_winner(candidate, constructed)
        assert candidate.is_winner is True

    def test_mark_winner_allows_none_source_element(self) -> None:
        """Test that mark_winner allows None source_element (synthetic candidates)."""
        page_data = PageData(
            page_number=1,
            elements=[],
            bbox=BBox(0, 0, 100, 100),
        )

        # Use a synthetic PageNumber without a source_element (e.g., derived from context)
        constructed = PageNumber(bbox=BBox(0, 0, 10, 10), value=1)
        candidate = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="page_number",
            score=0.95,
            score_details={},
            constructed=constructed,
            source_element=None,  # Synthetic candidate
        )

        result = ClassificationResult(page_data=page_data)
        result.add_candidate("page_number", candidate)
        result.mark_winner(candidate, constructed)
        assert candidate.is_winner is True

    def test_mark_winner_rejects_duplicate_winners_for_same_element(self) -> None:
        """Test that mark_winner rejects marking the same element as winner twice."""
        element = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)

        page_data = PageData(
            page_number=1,
            elements=[element],
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
            source_element=element,
        )

        # Second winner candidate with different label, same element
        constructed2 = StepNumber(bbox=BBox(0, 0, 10, 10), value=1)
        candidate2 = Candidate(
            bbox=BBox(0, 0, 10, 10),
            label="step_number",
            score=0.90,
            score_details={},
            constructed=constructed2,
            source_element=element,
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
            match="Element 1 already has a winner candidate for label 'page_number'",
        ):
            result.mark_winner(candidate2, constructed2)

        # Verify only the first candidate is a winner
        assert candidate1.is_winner is True
        assert candidate2.is_winner is False

    def test_mark_removed_validates_element_in_page_data(self) -> None:
        """Test that mark_removed validates element is in PageData."""
        element1 = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        element2 = Text(bbox=BBox(20, 20, 30, 30), text="2", id=2)  # Not in PageData

        page_data = PageData(
            page_number=1,
            elements=[element1],
            bbox=BBox(0, 0, 100, 100),
        )

        reason = RemovalReason(reason_type="child_bbox", target_element=element1)

        result = ClassificationResult(page_data=page_data)
        with pytest.raises(ValueError, match="must be in PageData.elements"):
            result.mark_removed(element2, reason)

    def test_mark_removed_allows_element_in_page_data(self) -> None:
        """Test that mark_removed succeeds when element is in PageData."""
        element1 = Text(bbox=BBox(0, 0, 10, 10), text="1", id=1)
        element2 = Text(bbox=BBox(20, 20, 30, 30), text="2", id=2)

        page_data = PageData(
            page_number=1,
            elements=[element1, element2],
            bbox=BBox(0, 0, 100, 100),
        )

        reason = RemovalReason(reason_type="child_bbox", target_element=element1)

        result = ClassificationResult(page_data=page_data)
        result.mark_removed(element2, reason)
        assert result.is_removed(element2)
