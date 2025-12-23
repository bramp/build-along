"""Tests for the parts classifier (Part pairing logic)."""

from collections.abc import Callable
from typing import cast

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
from build_a_long.pdf_extract.classifier.test_utils import PageBuilder
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Part,
)
from build_a_long.pdf_extract.extractor.page_blocks import Image, Text


@pytest.fixture
def classifier() -> PartsClassifier:
    return PartsClassifier(config=ClassifierConfig())


class TestPartsClassification:
    """Tests for Part assembly (pairing PartCount with Image)."""

    @pytest.mark.skip(reason="Requires constraint solver integration - Phase 2")
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
        builder = (
            PageBuilder(page_number=1, width=100, height=200)
            .add_image(10, 30, 10, 15, id=0)  # img1
            .add_image(30, 30, 10, 15, id=1)  # img2
            .add_text("2x", 10, 50, 10, 10, id=2)  # t1
            .add_text("2x", 10, 50, 10, 10, id=3)  # t2 (duplicate bbox)
            .add_text("3x", 30, 50, 10, 10, id=4)  # t3
        )
        page = builder.build()
        _ = cast(Image, page.blocks[0])
        _ = cast(Image, page.blocks[1])
        t1 = cast(Text, page.blocks[2])
        t2 = cast(Text, page.blocks[3])
        t3 = cast(Text, page.blocks[4])

        result = ClassificationResult(page_data=page)
        # Register classifiers so result.construct_candidate works
        PartsImageClassifier(config=classifier.config).score(result)
        PartNumberClassifier(config=classifier.config).score(result)
        PieceLengthClassifier(config=classifier.config).score(result)

        factory = candidate_factory(result)

        # Manually score part_count candidates
        factory.add_part_count(t1, score=1.0)
        factory.add_part_count(t2, score=0.9)  # Slightly lower score for duplicate
        factory.add_part_count(t3, score=1.0)

        classifier.score(result)

        # Use build_all which uses constraint solver to handle conflicts
        classifier.build_all(result)

        # Verify that only 2 Parts were successfully built
        # (img1 pairs with one of t1/t2, img2 pairs with t3)
        assert result.count_successful_candidates("part") == 2

        # Verify the Part objects have the expected counts
        parts = [result.get_constructed(c) for c in result.get_built_candidates("part")]
        assert all(isinstance(p, Part) for p in parts)
        part_counts = sorted([p.count.count for p in parts])  # type: ignore[union-attr]
        assert part_counts == [2, 3]

        # Verify that only 2 PartCounts were successfully constructed
        # (one of t1/t2 lost conflict)
        assert result.count_successful_candidates("part_count") == 2

    def test_part_count_without_nearby_image(
        self,
        classifier: PartsClassifier,
        candidate_factory: Callable[[ClassificationResult], CandidateFactory],
    ) -> None:
        """Test that part counts are not paired if no images are above them."""
        builder = (
            PageBuilder(page_number=1, width=200, height=200)
            .add_text("2x", 10, 10, 10, 10, id=0)  # t1
            .add_image(10, 50, 10, 15, id=1)  # Image below part count
        )
        page = builder.build()
        t1 = cast(Text, page.blocks[0])
        _ = cast(Image, page.blocks[1])

        result = ClassificationResult(page_data=page)
        PartsImageClassifier(config=classifier.config).score(result)

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

    @pytest.mark.skip(reason="Requires constraint solver integration - Phase 2")
    def test_multiple_images_above_picks_closest(
        self,
        classifier: PartsClassifier,
        candidate_factory: Callable[[ClassificationResult], CandidateFactory],
    ) -> None:
        """Test that when multiple images are above a count, the closest is picked."""
        builder = (
            PageBuilder(page_number=1, width=100, height=200)
            .add_image(10, 20, 10, 15, id=0)  # img_far (y=20..35)
            .add_image(10, 40, 10, 15, id=1)  # img_near (y=40..55)
            .add_text("2x", 10, 60, 10, 10, id=2)  # t1 (y=60..70)
        )
        page = builder.build()
        _ = cast(Image, page.blocks[0])
        img_near = cast(Image, page.blocks[1])
        t1 = cast(Text, page.blocks[2])

        result = ClassificationResult(page_data=page)
        PartsImageClassifier(config=classifier.config).score(result)

        factory = candidate_factory(result)

        # Manually score part_count candidate
        factory.add_part_count(t1, score=1.0)

        classifier.score(result)

        # Use build_all which uses constraint solver to handle conflicts
        classifier.build_all(result)

        # Should create 1 Part
        parts = [result.get_constructed(c) for c in result.get_built_candidates("part")]
        assert len(parts) == 1
        part = parts[0]
        assert isinstance(part, Part)

        # The Part should use the closer image (img_near)
        assert part.diagram is not None
        assert part.diagram.bbox == img_near.bbox

        # The Part bbox should encompass both the count and the closer image
        assert part.bbox.y0 <= img_near.bbox.y0
        assert part.bbox.y1 >= t1.bbox.y1

    @pytest.mark.skip(reason="Requires constraint solver integration - Phase 2")
    def test_horizontal_alignment_required(
        self,
        classifier: PartsClassifier,
        candidate_factory: Callable[[ClassificationResult], CandidateFactory],
    ) -> None:
        """Test that images must be roughly left-aligned with part counts."""
        builder = (
            PageBuilder(page_number=1, width=200, height=200)
            .add_image(10, 30, 10, 15, id=0)  # img1
            .add_text("2x", 150, 50, 10, 10, id=1)  # t1 (Not aligned)
        )
        page = builder.build()
        _ = cast(Image, page.blocks[0])
        t1 = cast(Text, page.blocks[1])

        result = ClassificationResult(page_data=page)
        PartsImageClassifier(config=classifier.config).score(result)

        factory = candidate_factory(result)
        factory.add_part_count(t1, score=1.0)

        classifier.score(result)

        # No Part should be created (horizontal misalignment)
        assert result.count_successful_candidates("part") == 0

        # PartCount should still be constructible (not part of Part)
        pc_candidate = result.get_candidate_for_block(t1, "part_count")
        assert pc_candidate is not None
        assert result.build(pc_candidate) is not None

    @pytest.mark.skip(reason="Requires constraint solver integration - Phase 2")
    def test_one_to_one_pairing_enforcement(
        self,
        classifier: PartsClassifier,
        candidate_factory: Callable[[ClassificationResult], CandidateFactory],
    ) -> None:
        """Test that one-to-one pairing is enforced (no image/count reuse)."""
        builder = (
            PageBuilder(page_number=1, width=100, height=200)
            .add_image(10, 30, 10, 15, id=0)  # img1
            .add_text("2x", 10, 50, 10, 10, id=1)  # t1 (closer)
            .add_text("3x", 10, 70, 10, 10, id=2)  # t2 (farther)
        )
        page = builder.build()
        _ = cast(Image, page.blocks[0])
        t1 = cast(Text, page.blocks[1])
        t2 = cast(Text, page.blocks[2])

        result = ClassificationResult(page_data=page)
        PartsImageClassifier(config=classifier.config).score(result)

        factory = candidate_factory(result)
        factory.add_part_count(t1, score=1.0)
        factory.add_part_count(
            t2, score=0.9
        )  # t2 is farther, so lower score if other factors were equal

        classifier.score(result)

        # Use build_all which uses constraint solver to handle conflicts
        classifier.build_all(result)

        # Only 1 Part should be created (img1 pairs with t1, the closer one)
        parts = [result.get_constructed(c) for c in result.get_built_candidates("part")]
        assert len(parts) == 1
        assert isinstance(parts[0], Part)
        assert parts[0].count.count == 2  # paired with t1 (the closer one)

        # Only 1 PartCount should be successfully constructed
        # (t2 lost conflict, or simply not paired)
        assert result.count_successful_candidates("part_count") == 1
        unconstructed_pc_candidate = result.get_candidate_for_block(t2, "part_count")
        assert unconstructed_pc_candidate is not None
        assert result.get_constructed(unconstructed_pc_candidate) is None
