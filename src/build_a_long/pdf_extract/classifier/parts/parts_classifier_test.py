"""Tests for the parts classifier (Part pairing logic)."""

from collections.abc import Callable

import pytest

from build_a_long.pdf_extract.classifier import (
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.classifier.conftest import CandidateFactory
from build_a_long.pdf_extract.classifier.parts.part_number_classifier import (
    PartNumberClassifier,
)
from build_a_long.pdf_extract.classifier.parts.parts_classifier import PartsClassifier
from build_a_long.pdf_extract.classifier.parts.parts_image_classifier import (
    PartsImageClassifier,
)
from build_a_long.pdf_extract.classifier.parts.piece_length_classifier import (
    PieceLengthClassifier,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Part,
)
from build_a_long.pdf_extract.extractor.page_blocks import Image, Text


@pytest.fixture
def classifier() -> PartsClassifier:
    return PartsClassifier(config=ClassifierConfig())


class TestPartsClassification:
    """Tests for Part assembly (pairing PartCount with Image)."""

    def test_duplicate_part_counts_only_match_once(
        self,
        classifier: PartsClassifier,
        candidate_factory: Callable[[ClassificationResult], CandidateFactory],
    ) -> None:
        """Test that duplicate part counts don't both pair with the same image.

        When two part count blocks have identical bounding boxes (e.g., drop
        shadows, overlapping detections), both create PartCount candidates,
        but only one should pair with any given image to create a Part.
        """
        page_bbox = BBox(0, 0, 100, 200)

        img1 = Image(id=0, bbox=BBox(10, 30, 20, 45))
        img2 = Image(id=1, bbox=BBox(30, 30, 40, 45))

        t1 = Text(id=2, bbox=BBox(10, 50, 20, 60), text="2x")
        t2 = Text(id=3, bbox=BBox(10, 50, 20, 60), text="2x")  # duplicate
        t3 = Text(id=4, bbox=BBox(30, 50, 40, 60), text="3x")

        page = PageData(
            page_number=1,
            blocks=[img1, img2, t1, t2, t3],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page)
        # Register classifiers so result.construct_candidate works
        PartsImageClassifier(classifier.config).score(result)
        PartNumberClassifier(classifier.config).score(result)
        PieceLengthClassifier(classifier.config).score(result)

        factory = candidate_factory(result)

        # Manually score part_count candidates
        factory.add_part_count(t1, score=1.0)
        factory.add_part_count(t2, score=0.9)  # Slightly lower score for duplicate
        factory.add_part_count(t3, score=1.0)

        classifier.score(result)

        # Now construct the Parts
        parts: list[Part] = []
        for part_candidate in result.get_candidates("part"):
            part = result.build(part_candidate)
            assert isinstance(part, Part)
            parts.append(part)

        # Verify that only 2 Parts are created (img1 pairs with one of t1/t2, img2 pairs with t3)
        assert len(parts) == 2

        # Verify the Part objects have the expected counts
        part_counts = sorted([p.count.count for p in parts])
        assert part_counts == [2, 3]

        # Verify that only 2 PartCounts were successfully constructed (one of t1/t2 lost conflict)
        assert result.count_successful_candidates("part_count") == 2

    def test_part_count_without_nearby_image(
        self,
        classifier: PartsClassifier,
        candidate_factory: Callable[[ClassificationResult], CandidateFactory],
    ) -> None:
        """Test that part counts are not paired if no images are above them."""
        page_bbox = BBox(0, 0, 200, 200)

        t1 = Text(id=0, bbox=BBox(10, 10, 20, 20), text="2x")
        img1 = Image(id=1, bbox=BBox(10, 50, 20, 65))  # Image below part count

        page = PageData(
            page_number=1,
            blocks=[img1, t1],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page)
        PartsImageClassifier(classifier.config).score(result)

        factory = candidate_factory(result)

        # Manually score part_count candidate
        factory.add_part_count(t1, score=1.0)

        classifier.score(result)

        # Parts should be created even though image isn't exactly left-aligned
        assert result.count_successful_candidates("part") == 0

        # The PartCount should exist but not be consumed
        pc_candidate = result.get_candidate_for_block(t1, "part_count")
        assert pc_candidate is not None
        assert result.build(pc_candidate) is not None  # Should be constructible

    def test_multiple_images_above_picks_closest(
        self,
        classifier: PartsClassifier,
        candidate_factory: Callable[[ClassificationResult], CandidateFactory],
    ) -> None:
        """Test that when multiple images are above a count, the closest is picked."""
        page_bbox = BBox(0, 0, 100, 200)

        img_far = Image(id=0, bbox=BBox(10, 20, 20, 35))
        img_near = Image(id=1, bbox=BBox(10, 40, 20, 55))
        t1 = Text(id=2, bbox=BBox(10, 60, 20, 70), text="2x")

        page = PageData(
            page_number=1,
            blocks=[img_far, img_near, t1],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page)
        PartsImageClassifier(classifier.config).score(result)

        factory = candidate_factory(result)

        # Manually score part_count candidate
        factory.add_part_count(t1, score=1.0)

        classifier.score(result)

        # Now construct the Parts
        parts: list[Part] = []
        for part_candidate in result.get_candidates("part"):
            part = result.build(part_candidate)
            assert isinstance(part, Part)
            parts.append(part)

        # Should create 1 Part
        assert len(parts) == 1
        part = parts[0]

        # The Part should use the closer image (img_near)
        assert part.diagram is not None
        assert part.diagram.bbox == img_near.bbox

        # The Part bbox should encompass both the count and the closer image
        assert part.bbox.y0 <= img_near.bbox.y0
        assert part.bbox.y1 >= t1.bbox.y1

    def test_horizontal_alignment_required(
        self,
        classifier: PartsClassifier,
        candidate_factory: Callable[[ClassificationResult], CandidateFactory],
    ) -> None:
        """Test that images must be roughly left-aligned with part counts."""
        page_bbox = BBox(0, 0, 200, 200)

        img1 = Image(id=0, bbox=BBox(10, 30, 20, 45))
        t1 = Text(id=1, bbox=BBox(150, 50, 160, 60), text="2x")  # Not aligned

        page = PageData(
            page_number=1,
            blocks=[img1, t1],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page)
        PartsImageClassifier(classifier.config).score(result)

        factory = candidate_factory(result)
        factory.add_part_count(t1, score=1.0)

        classifier.score(result)

        # No Part should be created (horizontal misalignment)
        assert result.count_successful_candidates("part") == 0

        # PartCount should still be constructible (not part of Part)
        pc_candidate = result.get_candidate_for_block(t1, "part_count")
        assert pc_candidate is not None
        assert result.build(pc_candidate) is not None

    def test_one_to_one_pairing_enforcement(
        self,
        classifier: PartsClassifier,
        candidate_factory: Callable[[ClassificationResult], CandidateFactory],
    ) -> None:
        """Test that one-to-one pairing is enforced (no image/count reuse)."""
        page_bbox = BBox(0, 0, 100, 200)

        img1 = Image(id=0, bbox=BBox(10, 30, 20, 45))
        t1 = Text(id=1, bbox=BBox(10, 50, 20, 60), text="2x")  # closer
        t2 = Text(id=2, bbox=BBox(10, 70, 20, 80), text="3x")  # farther

        page = PageData(
            page_number=1,
            blocks=[img1, t1, t2],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page)
        PartsImageClassifier(classifier.config).score(result)

        factory = candidate_factory(result)
        factory.add_part_count(t1, score=1.0)
        factory.add_part_count(
            t2, score=0.9
        )  # t2 is farther, so lower score if other factors were equal

        classifier.score(result)

        # Manually construct Parts
        parts: list[Part] = []
        for part_candidate in result.get_candidates("part"):
            part = result.build(part_candidate)
            assert isinstance(part, Part)
            parts.append(part)

        # Only 1 Part should be created (img1 pairs with t1, the closer one)
        assert len(parts) == 1
        assert parts[0].count.count == 2  # paired with t1 (the closer one)

        # Only 1 PartCount should be successfully constructed (t2 lost conflict, or simply not paired)
        assert result.count_successful_candidates("part_count") == 1
        unconstructed_pc_candidate = result.get_candidate_for_block(t2, "part_count")
        assert unconstructed_pc_candidate is not None
        assert unconstructed_pc_candidate.constructed is None
