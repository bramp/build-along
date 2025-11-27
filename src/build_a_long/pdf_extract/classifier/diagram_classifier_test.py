"""Tests for DiagramClassifier."""

import pytest

from build_a_long.pdf_extract.classifier.classification_result import ClassifierConfig
from build_a_long.pdf_extract.classifier.diagram_classifier import DiagramClassifier
from build_a_long.pdf_extract.extractor.bbox import BBox


@pytest.fixture
def classifier() -> DiagramClassifier:
    return DiagramClassifier(config=ClassifierConfig())


def test_score_area(classifier: DiagramClassifier):
    """Test area scoring logic."""
    page_bbox = BBox(0, 0, 100, 100)  # 10,000 area

    # Too small (< 3% of page)
    tiny_bbox = BBox(0, 0, 10, 20)  # 200 area = 2%
    assert classifier._score_area(tiny_bbox, page_bbox) == 0.0

    # Minimum size (3% of page)
    min_bbox = BBox(0, 0, 10, 30)  # 300 area = 3%
    assert classifier._score_area(min_bbox, page_bbox) == 0.5

    # Good size (3-60% of page)
    medium_bbox = BBox(0, 0, 50, 50)  # 2,500 area = 25%
    assert classifier._score_area(medium_bbox, page_bbox) >= 1.0

    # Very large (> 60% of page)
    large_bbox = BBox(0, 0, 90, 90)  # 8,100 area = 81%
    score = classifier._score_area(large_bbox, page_bbox)
    assert 0.0 <= score < 0.5  # Should have reduced score


def test_score_position(classifier: DiagramClassifier):
    """Test position scoring logic."""
    page_bbox = BBox(0, 0, 100, 100)

    # Center position (should score 1.0)
    center_bbox = BBox(40, 40, 60, 60)
    center_score = classifier._score_position(center_bbox, page_bbox)
    assert center_score == 1.0

    # Left position within acceptable range (0.05-0.95)
    left_bbox = BBox(10, 40, 30, 60)  # center_x = 0.2
    left_score = classifier._score_position(left_bbox, page_bbox)
    assert left_score == 1.0  # Still in acceptable range

    # Far left at edge (gets penalized but not 0)
    far_left_bbox = BBox(0, 40, 2, 60)  # center_x = 0.01
    far_left_score = classifier._score_position(far_left_bbox, page_bbox)
    assert 0.0 < far_left_score < 1.0  # Penalized but not zero
    assert far_left_score < left_score
