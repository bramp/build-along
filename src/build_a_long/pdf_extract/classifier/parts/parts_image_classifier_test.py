"""Tests for PartsImageClassifier."""

import pytest

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.classifier.parts.parts_image_classifier import (
    PartsImageClassifier,
)
from build_a_long.pdf_extract.classifier.parts.shine_classifier import ShineClassifier
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import PartImage
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image


@pytest.fixture
def classifier() -> PartsImageClassifier:
    return PartsImageClassifier(config=ClassifierConfig())


def make_page_data(blocks: list, page_size: float = 1000.0) -> PageData:
    """Create a PageData with given blocks."""
    return PageData(
        page_number=1,
        bbox=BBox(x0=0, y0=0, x1=page_size, y1=page_size),
        blocks=blocks,
    )


class DummyScore(Score):
    def score(self) -> Weight:
        return 1.0


class TestPartsImageClassifier:
    """Tests for part image classification."""

    def test_finds_valid_part_image(self, classifier: PartsImageClassifier):
        """Test that a valid part image is found."""
        page_size = 1000.0
        # 10% of page size (100x100) - ideal size
        image = Image(id=1, bbox=BBox(100, 100, 200, 200))

        page = make_page_data([image], page_size=page_size)
        result = ClassificationResult(page_data=page)
        classifier._score(result)

        candidate = result.get_candidate_for_block(image, "part_image")
        assert candidate is not None
        assert candidate.score > 0.8  # Should be high score

        built = classifier.build(candidate, result)
        assert isinstance(built, PartImage)
        assert built.bbox == image.bbox
        assert built.shine is None

    def test_scores_too_small_image_low(self, classifier: PartsImageClassifier):
        """Test that too small images score low."""
        page_size = 1000.0
        # 1% of page size (10x10) - too small
        image = Image(id=1, bbox=BBox(100, 100, 110, 110))

        page = make_page_data([image], page_size=page_size)
        result = ClassificationResult(page_data=page)
        classifier._score(result)

        candidate = result.get_candidate_for_block(image, "part_image")
        assert candidate is not None
        # Score logic: ratio < 0.02 -> linear from 0 to 0.3. 0.01 -> 0.15
        assert candidate.score < 0.3

    def test_scores_too_large_image_low(self, classifier: PartsImageClassifier):
        """Test that too large images score low."""
        page_size = 1000.0
        # 50% of page size (500x500) - too large
        image = Image(id=1, bbox=BBox(0, 0, 500, 500))

        page = make_page_data([image], page_size=page_size)
        result = ClassificationResult(page_data=page)
        classifier._score(result)

        candidate = result.get_candidate_for_block(image, "part_image")
        assert candidate is not None
        # Score logic: ratio > 0.40 -> decreases from 0.3 to 0.1
        assert candidate.score < 0.4

    def test_builds_with_shine(self, classifier: PartsImageClassifier):
        """Test building part image with attached shine."""
        page_size = 1000.0
        # 100x100 image at (100,100) -> TR corner (200, 100)
        image = Image(id=1, bbox=BBox(100, 100, 200, 200))

        # Mock shine candidate
        # Shine at top-right corner of image: (190, 100, 210, 120) - overlaps image
        shine_bbox = BBox(190, 100, 210, 120)
        # We need a dummy drawing for source blocks
        shine_drawing = Drawing(id=99, bbox=shine_bbox)

        # Make sure shine is registered/mocked?
        # Shine discovery relies on "shine" candidates in result.
        # We need to manually add a shine candidate to result.
        page = make_page_data([image, shine_drawing], page_size=page_size)
        result = ClassificationResult(page_data=page)
        # Register ShineClassifier to handle shine candidates
        result._register_classifier("shine", ShineClassifier(config=ClassifierConfig()))

        shine_candidate = Candidate(
            bbox=shine_bbox,
            label="shine",
            score=1.0,
            score_details=DummyScore(),
            source_blocks=[shine_drawing],
        )
        result.add_candidate(shine_candidate)

        classifier._score(result)
        candidate = result.get_candidate_for_block(image, "part_image")
        assert candidate is not None

        built = classifier.build(candidate, result)
        assert isinstance(built, PartImage)
        assert built.shine is not None
        assert built.shine.bbox == shine_bbox
        # Bbox should be union of image and shine
        expected_bbox = image.bbox.union(shine_bbox)
        assert built.bbox == expected_bbox
