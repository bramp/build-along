"""Tests for PartsImageClassifier."""

import pytest

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.classifier.parts.parts_image_classifier import (
    PartImageScore,
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

        # Should have one candidate (without shine)
        all_candidates = list(result.get_scored_candidates("part_image"))
        assert len(all_candidates) == 1

        candidate = all_candidates[0]
        assert candidate.score > 0.8  # Should be high score

        built = classifier.build(candidate, result)
        assert isinstance(built, PartImage)
        assert built.bbox == image.bbox
        assert built.shine is None

    def test_scores_too_small_image_low(self, classifier: PartsImageClassifier):
        """Test that too small images are rejected."""
        page_size = 1000.0
        # 1% of page size (10x10) - too small
        image = Image(id=1, bbox=BBox(100, 100, 110, 110))

        page = make_page_data([image], page_size=page_size)
        result = ClassificationResult(page_data=page)
        classifier._score(result)

        # Should have no candidates
        all_candidates = list(result.get_scored_candidates("part_image"))
        assert len(all_candidates) == 0

    def test_scores_too_large_image_low(self, classifier: PartsImageClassifier):
        """Test that too large images are rejected."""
        page_size = 1000.0
        # 50% of page size (500x500) - too large
        image = Image(id=1, bbox=BBox(0, 0, 500, 500))

        page = make_page_data([image], page_size=page_size)
        result = ClassificationResult(page_data=page)
        classifier._score(result)

        # Should have no candidates
        all_candidates = list(result.get_scored_candidates("part_image"))
        assert len(all_candidates) == 0

    def test_creates_candidates_with_and_without_shine(
        self, classifier: PartsImageClassifier
    ):
        """Test that separate candidates are created with and without shine."""
        page_size = 1000.0
        # 100x100 image at (100,100) -> TR corner (200, 100)
        image = Image(id=1, bbox=BBox(100, 100, 200, 200))

        # Shine near top-right corner of image: (195, 97) center, distance ~5.8
        # Shine overlaps the image at top-right
        shine_bbox = BBox(190, 95, 200, 110)
        shine_drawing = Drawing(id=99, bbox=shine_bbox)

        page = make_page_data([image, shine_drawing], page_size=page_size)
        result = ClassificationResult(page_data=page)
        # Register ShineClassifier to handle shine candidates
        result._register_classifier("shine", ShineClassifier(config=ClassifierConfig()))

        # Add shine candidate
        shine_candidate = Candidate(
            bbox=shine_bbox,
            label="shine",
            score=1.0,
            score_details=DummyScore(),
            source_blocks=[shine_drawing],
        )
        result.add_candidate(shine_candidate)

        # Run classifier
        classifier._score(result)

        # Should have multiple candidates for this image:
        # 1. Without shine
        # 2. With shine
        all_candidates = list(result.get_scored_candidates("part_image"))
        assert len(all_candidates) == 2

        # Find the candidates
        without_shine = [
            c
            for c in all_candidates
            if isinstance(c.score_details, PartImageScore)
            and c.score_details.shine_candidate is None
        ]
        with_shine = [
            c
            for c in all_candidates
            if isinstance(c.score_details, PartImageScore)
            and c.score_details.shine_candidate is not None
        ]

        assert len(without_shine) == 1
        assert len(with_shine) == 1

        # Without shine should have original bbox
        assert without_shine[0].bbox == image.bbox

        # With shine should have expanded bbox
        expected_bbox = image.bbox.union(shine_bbox)
        assert with_shine[0].bbox == expected_bbox

    def test_builds_with_shine(self, classifier: PartsImageClassifier):
        """Test building part image with attached shine."""
        page_size = 1000.0
        # 100x100 image at (100,100) -> TR corner (200, 100)
        image = Image(id=1, bbox=BBox(100, 100, 200, 200))

        # Shine near top-right corner: center ~(195, 102.5), distance ~6.25 from TR
        shine_bbox = BBox(190, 95, 200, 110)
        shine_drawing = Drawing(id=99, bbox=shine_bbox)

        page = make_page_data([image, shine_drawing], page_size=page_size)
        result = ClassificationResult(page_data=page)
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

        # Find the candidate WITH shine
        all_candidates = list(result.get_scored_candidates("part_image"))
        with_shine_cand = next(
            c
            for c in all_candidates
            if isinstance(c.score_details, PartImageScore)
            and c.score_details.shine_candidate is not None
        )

        built = classifier.build(with_shine_cand, result)
        assert isinstance(built, PartImage)
        assert built.shine is not None
        assert built.shine.bbox == shine_bbox
        # Bbox should be union of image and shine
        expected_bbox = image.bbox.union(shine_bbox)
        assert built.bbox == expected_bbox

    def test_shine_not_overlapping_is_ignored(self, classifier: PartsImageClassifier):
        """Test that shines not overlapping the image are ignored."""
        page_size = 1000.0
        image = Image(id=1, bbox=BBox(100, 100, 200, 200))

        # Shine far away from image - no overlap
        shine_bbox = BBox(300, 300, 320, 320)
        shine_drawing = Drawing(id=99, bbox=shine_bbox)

        page = make_page_data([image, shine_drawing], page_size=page_size)
        result = ClassificationResult(page_data=page)
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

        # Should only have one candidate (without shine) since shine doesn't overlap
        all_candidates = list(result.get_scored_candidates("part_image"))
        assert len(all_candidates) == 1

        # The candidate should not have shine
        cand = all_candidates[0]
        assert isinstance(cand.score_details, PartImageScore)
        assert cand.score_details.shine_candidate is None

    def test_shine_too_far_from_corner_is_ignored(
        self, classifier: PartsImageClassifier
    ):
        """Test that shines overlapping but far from TR corner are ignored."""
        page_size = 1000.0
        # 100x100 image at (100,100) -> TR corner (200, 100)
        image = Image(id=1, bbox=BBox(100, 100, 200, 200))

        # Shine overlapping but at bottom-left of image (far from TR corner)
        # Distance from TR corner (200, 100) to center (107.5, 192.5) is ~127 units
        shine_bbox = BBox(100, 185, 115, 200)
        shine_drawing = Drawing(id=99, bbox=shine_bbox)

        page = make_page_data([image, shine_drawing], page_size=page_size)
        result = ClassificationResult(page_data=page)
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

        # Should only have one candidate (without shine) since shine is too far
        # from the TR corner (max distance is 10)
        all_candidates = list(result.get_scored_candidates("part_image"))
        assert len(all_candidates) == 1

        cand = all_candidates[0]
        assert isinstance(cand.score_details, PartImageScore)
        assert cand.score_details.shine_candidate is None
