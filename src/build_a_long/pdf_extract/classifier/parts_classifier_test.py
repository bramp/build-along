"""Tests for the parts classifier (Part pairing logic)."""

from build_a_long.pdf_extract.classifier.classification_result import (
    Candidate,
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.classifier.part_count_classifier import (
    PartCountClassifier,
    _PartCountScore,
)
from build_a_long.pdf_extract.classifier.part_number_classifier import (
    PartNumberClassifier,
)
from build_a_long.pdf_extract.classifier.parts_classifier import PartsClassifier
from build_a_long.pdf_extract.classifier.piece_length_classifier import (
    PieceLengthClassifier,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Part,
)
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image, Text


class TestPartsClassification:
    """Tests for Part assembly (pairing PartCount with Image)."""

    def _setup_parts_classifier_test(
        self, page_data: PageData
    ) -> tuple[PartsClassifier, ClassificationResult]:
        config = ClassifierConfig()
        parts_classifier = PartsClassifier(config)
        result = ClassificationResult(page_data=page_data)

        # Register classifiers so result.construct_candidate works
        result.register_classifier("part_count", PartCountClassifier(config))
        result.register_classifier("part_number", PartNumberClassifier(config))
        result.register_classifier("piece_length", PieceLengthClassifier(config))
        result.register_classifier("part", parts_classifier)

        return parts_classifier, result

    def _create_and_score_part_count_candidate(
        self, result: ClassificationResult, text_block: Text, score_value: float = 1.0
    ) -> Candidate:
        # Simplified creation of PartCount candidate
        score_details = _PartCountScore(
            text_score=score_value,
            font_size_score=0.5,  # Dummy value
            matched_hint="catalog_part_count",  # Dummy value
        )
        candidate = Candidate(
            bbox=text_block.bbox,
            label="part_count",
            score=score_value,
            score_details=score_details,
            constructed=None,
            source_blocks=[text_block],
        )
        result.add_candidate("part_count", candidate)
        return candidate

    def _create_and_score_part_number_candidate(
        self, result: ClassificationResult, text_block: Text, score_value: float = 1.0
    ) -> Candidate:
        # Simplified creation of PartNumber candidate
        candidate = Candidate(
            bbox=text_block.bbox,
            label="part_number",
            score=score_value,
            score_details={"text_score": score_value},  # Simplified score details
            constructed=None,  # Added
            source_blocks=[text_block],
        )
        result.add_candidate("part_number", candidate)
        return candidate

    def _create_and_score_piece_length_candidate(
        self,
        result: ClassificationResult,
        text_block: Text,
        drawing_block: Drawing,
        score_value: float = 1.0,
    ) -> Candidate:
        # Simplified creation of PieceLength candidate
        candidate = Candidate(
            bbox=text_block.bbox.union(drawing_block.bbox),
            label="piece_length",
            score=score_value,
            score_details={"text_score": score_value},  # Simplified score details
            constructed=None,  # Added
            source_blocks=[text_block, drawing_block],
        )
        result.add_candidate("piece_length", candidate)
        return candidate

    def test_duplicate_part_counts_only_match_once(self) -> None:
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

        parts_classifier, result = self._setup_parts_classifier_test(page)

        # Manually score part_count candidates
        self._create_and_score_part_count_candidate(result, t1, score_value=1.0)
        self._create_and_score_part_count_candidate(
            result, t2, score_value=0.9
        )  # Slightly lower score for duplicate
        self._create_and_score_part_count_candidate(result, t3, score_value=1.0)

        parts_classifier.score(result)

        # Now construct the Parts
        parts: list[Part] = []
        for part_candidate in result.get_candidates("part"):
            part = result.construct_candidate(part_candidate)
            assert isinstance(part, Part)
            parts.append(part)

        # Verify that only 2 Parts are created (img1 pairs with one of t1/t2, img2 pairs with t3)
        assert len(parts) == 2

        # Verify the Part objects have the expected counts
        part_counts = sorted([p.count.count for p in parts])
        assert part_counts == [2, 3]

        # Verify that only 2 PartCounts were successfully constructed (one of t1/t2 lost conflict)
        assert result.count_successful_candidates("part_count") == 2

    def test_part_count_without_nearby_image(self) -> None:
        """Test that part counts are not paired if no images are above them."""
        page_bbox = BBox(0, 0, 200, 200)

        t1 = Text(id=0, bbox=BBox(10, 10, 20, 20), text="2x")
        img1 = Image(id=1, bbox=BBox(10, 50, 20, 65))  # Image below part count

        page = PageData(
            page_number=1,
            blocks=[img1, t1],
            bbox=page_bbox,
        )

        parts_classifier, result = self._setup_parts_classifier_test(page)

        # Manually score part_count candidate
        self._create_and_score_part_count_candidate(result, t1, score_value=1.0)

        parts_classifier.score(result)

        # No Part should be created (image is below, not above)
        assert result.count_successful_candidates("part") == 0

        # The PartCount should exist but not be consumed
        pc_candidate = result.get_candidate_for_block(t1, "part_count")
        assert pc_candidate is not None
        assert (
            result.construct_candidate(pc_candidate) is not None
        )  # Should be constructible

    def test_multiple_images_above_picks_closest(self) -> None:
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

        parts_classifier, result = self._setup_parts_classifier_test(page)

        # Manually score part_count candidate
        self._create_and_score_part_count_candidate(result, t1, score_value=1.0)

        parts_classifier.score(result)

        # Now construct the Parts
        parts: list[Part] = []
        for part_candidate in result.get_candidates("part"):
            part = result.construct_candidate(part_candidate)
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

    def test_horizontal_alignment_required(self) -> None:
        """Test that images must be roughly left-aligned with part counts."""
        page_bbox = BBox(0, 0, 200, 200)

        img1 = Image(id=0, bbox=BBox(10, 30, 20, 45))
        t1 = Text(id=1, bbox=BBox(150, 50, 160, 60), text="2x")  # Not aligned

        page = PageData(
            page_number=1,
            blocks=[img1, t1],
            bbox=page_bbox,
        )

        parts_classifier, result = self._setup_parts_classifier_test(page)
        self._create_and_score_part_count_candidate(result, t1, score_value=1.0)

        parts_classifier.score(result)

        # No Part should be created (horizontal misalignment)
        assert result.count_successful_candidates("part") == 0

        # PartCount should still be constructible (not part of Part)
        pc_candidate = result.get_candidate_for_block(t1, "part_count")
        assert pc_candidate is not None
        assert result.construct_candidate(pc_candidate) is not None

    def test_one_to_one_pairing_enforcement(self) -> None:
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

        parts_classifier, result = self._setup_parts_classifier_test(page)
        self._create_and_score_part_count_candidate(result, t1, score_value=1.0)
        self._create_and_score_part_count_candidate(
            result, t2, score_value=0.9
        )  # t2 is farther, so lower score if other factors were equal

        parts_classifier.score(result)

        # Manually construct Parts
        parts: list[Part] = []
        for part_candidate in result.get_candidates("part"):
            part = result.construct_candidate(part_candidate)
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
