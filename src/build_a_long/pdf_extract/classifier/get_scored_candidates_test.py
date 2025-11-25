"""
Tests for get_scored_candidates() API.

This test demonstrates the recommended pattern for classifiers that depend
on other classifiers.
"""

from build_a_long.pdf_extract.classifier.classification_result import (
    Candidate,
    ClassificationResult,
)
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.extractor import PageData
from build_a_long.pdf_extract.extractor.page_blocks import Text


def test_get_scored_candidates_returns_sorted_by_score() -> None:
    """Test that get_scored_candidates returns candidates sorted by score."""
    # Create a simple page with one text block
    text = Text(
        id=1, bbox=BBox(0, 0, 10, 10), text="42", font_size=12.0, font_name="Arial"
    )
    page_data = PageData(
        page_number=1,
        blocks=[text],
        bbox=BBox(0, 0, 100, 100),
    )

    result = ClassificationResult(page_data=page_data)

    # Add some candidates with different scores
    result.add_candidate(
        "test_label",
        Candidate(
            bbox=text.bbox,
            label="test_label",
            score=0.8,
            score_details={"detail": "high"},
            constructed=None,
            source_blocks=[text],
        ),
    )
    result.add_candidate(
        "test_label",
        Candidate(
            bbox=text.bbox,
            label="test_label",
            score=0.3,
            score_details={"detail": "low"},
            constructed=None,
            source_blocks=[text],
        ),
    )
    result.add_candidate(
        "test_label",
        Candidate(
            bbox=text.bbox,
            label="test_label",
            score=0.9,
            score_details={"detail": "highest"},
            constructed=None,
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


def test_get_scored_candidates_filters_by_min_score() -> None:
    """Test that get_scored_candidates filters by minimum score."""
    text = Text(
        id=1, bbox=BBox(0, 0, 10, 10), text="42", font_size=12.0, font_name="Arial"
    )
    page_data = PageData(
        page_number=1,
        blocks=[text],
        bbox=BBox(0, 0, 100, 100),
    )

    result = ClassificationResult(page_data=page_data)

    # Add candidates with different scores
    result.add_candidate(
        "test_label",
        Candidate(
            bbox=text.bbox,
            label="test_label",
            score=0.8,
            score_details={"detail": "high"},
            constructed=None,
            source_blocks=[text],
        ),
    )
    result.add_candidate(
        "test_label",
        Candidate(
            bbox=text.bbox,
            label="test_label",
            score=0.3,
            score_details={"detail": "low"},
            constructed=None,
            source_blocks=[text],
        ),
    )

    # Get candidates with minimum score
    candidates = result.get_scored_candidates("test_label", min_score=0.5)

    # Should only include candidates with score >= 0.5
    assert len(candidates) == 1
    assert candidates[0].score == 0.8


def test_get_scored_candidates_excludes_unscored() -> None:
    """Test that get_scored_candidates excludes candidates without score_details."""
    text = Text(
        id=1, bbox=BBox(0, 0, 10, 10), text="42", font_size=12.0, font_name="Arial"
    )
    page_data = PageData(
        page_number=1,
        blocks=[text],
        bbox=BBox(0, 0, 100, 100),
    )

    result = ClassificationResult(page_data=page_data)

    # Add a candidate with score_details
    result.add_candidate(
        "test_label",
        Candidate(
            bbox=text.bbox,
            label="test_label",
            score=0.8,
            score_details={"detail": "scored"},
            constructed=None,
            source_blocks=[text],
        ),
    )

    # Add a candidate without score_details (shouldn't happen, but test it)
    result.add_candidate(
        "test_label",
        Candidate(
            bbox=text.bbox,
            label="test_label",
            score=0.9,
            score_details=None,  # No score details
            constructed=None,
            source_blocks=[text],
        ),
    )

    # Get scored candidates
    candidates = result.get_scored_candidates("test_label")

    # Should only include candidates with score_details
    assert len(candidates) == 1
    assert candidates[0].score == 0.8


def test_get_scored_candidates_includes_unconstructed() -> None:
    """Test get_scored_candidates includes unconstructed candidates.

    This enforces the pattern that score() methods should work with candidates,
    not check whether they've been constructed.
    """
    text = Text(
        id=1, bbox=BBox(0, 0, 10, 10), text="42", font_size=12.0, font_name="Arial"
    )
    page_data = PageData(
        page_number=1,
        blocks=[text],
        bbox=BBox(0, 0, 100, 100),
    )

    result = ClassificationResult(page_data=page_data)

    # Add a candidate that hasn't been constructed yet
    result.add_candidate(
        "test_label",
        Candidate(
            bbox=text.bbox,
            label="test_label",
            score=0.8,
            score_details={"detail": "not constructed"},
            constructed=None,  # Not constructed yet
            source_blocks=[text],
        ),
    )

    # Get scored candidates
    candidates = result.get_scored_candidates("test_label")

    # Should include the candidate even though it's not constructed
    assert len(candidates) == 1
    assert candidates[0].constructed is None
