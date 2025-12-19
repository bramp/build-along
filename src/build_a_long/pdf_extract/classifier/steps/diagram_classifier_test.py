"""Tests for DiagramClassifier."""

import pytest

from build_a_long.pdf_extract.classifier import (
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.classifier.steps.diagram_classifier import (
    DiagramClassifier,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import Diagram
from build_a_long.pdf_extract.extractor.page_blocks import Image


@pytest.fixture
def classifier() -> DiagramClassifier:
    return DiagramClassifier(config=ClassifierConfig())


class TestDiagramScoring:
    """Tests for diagram scoring (one candidate per image)."""

    def test_single_image_creates_diagram(self, classifier: DiagramClassifier) -> None:
        """Test that a single image creates a diagram candidate."""
        page_bbox = BBox(0, 0, 200, 300)
        img = Image(id=1, bbox=BBox(50, 100, 150, 200))

        page_data = PageData(
            page_number=1,
            blocks=[img],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)
        classifier.score(result)

        candidates = result.get_scored_candidates("diagram", valid_only=False)
        assert len(candidates) == 1
        assert candidates[0].bbox == img.bbox

    def test_multiple_images_create_multiple_candidates(
        self, classifier: DiagramClassifier
    ) -> None:
        """Test that multiple images create multiple candidates.

        No clustering occurs during the score phase.
        """
        page_bbox = BBox(0, 0, 200, 300)
        # Three overlapping images - each becomes a candidate
        img1 = Image(id=1, bbox=BBox(50, 100, 100, 150))
        img2 = Image(id=2, bbox=BBox(90, 120, 140, 170))
        img3 = Image(id=3, bbox=BBox(130, 140, 180, 190))

        page_data = PageData(
            page_number=1,
            blocks=[img1, img2, img3],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)
        classifier.score(result)

        candidates = result.get_scored_candidates("diagram", valid_only=False)
        # Each image creates a separate candidate
        assert len(candidates) == 3

    def test_filters_out_full_page_images(self, classifier: DiagramClassifier) -> None:
        """Test that full-page images (>95% of page) are filtered out."""
        page_bbox = BBox(0, 0, 200, 300)
        # Image covering 96% of page (background/border)
        # 199 * 299 = 59501. 200 * 300 = 60000. Ratio = 0.99
        large_img = Image(id=1, bbox=BBox(0, 0, 199, 299))
        # Normal image
        normal_img = Image(id=2, bbox=BBox(50, 100, 100, 150))

        page_data = PageData(
            page_number=1,
            blocks=[large_img, normal_img],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)
        classifier.score(result)

        candidates = result.get_scored_candidates("diagram", valid_only=False)
        # Should only have the normal image
        assert len(candidates) == 1
        assert candidates[0].bbox == normal_img.bbox


class TestDiagramBuilding:
    """Tests for diagram building (lazy clustering)."""

    def test_build_single_image(self, classifier: DiagramClassifier) -> None:
        """Test building a diagram from a single isolated image."""
        page_bbox = BBox(0, 0, 200, 300)
        img = Image(id=1, bbox=BBox(50, 100, 150, 200))

        page_data = PageData(
            page_number=1,
            blocks=[img],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)
        classifier.score(result)

        candidates = result.get_scored_candidates("diagram", valid_only=False)
        assert len(candidates) == 1

        # Build the diagram
        diagram = classifier.build(candidates[0], result)
        assert isinstance(diagram, Diagram)
        assert diagram.bbox == img.bbox

    def test_build_clusters_adjacent_images(
        self, classifier: DiagramClassifier
    ) -> None:
        """Test that build() clusters adjacent/overlapping unclaimed images."""
        page_bbox = BBox(0, 0, 200, 300)
        # Three overlapping images that should cluster together
        img1 = Image(id=1, bbox=BBox(50, 100, 100, 150))
        img2 = Image(id=2, bbox=BBox(90, 120, 140, 170))  # Overlaps img1
        img3 = Image(id=3, bbox=BBox(130, 140, 180, 190))  # Overlaps img2

        page_data = PageData(
            page_number=1,
            blocks=[img1, img2, img3],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)
        classifier.score(result)

        candidates = result.get_scored_candidates("diagram", valid_only=False)
        assert len(candidates) == 3  # Three candidates during scoring

        # Build the first candidate - should cluster all three
        # Use result.build to ensure source blocks are marked as consumed
        diagram = result.build(candidates[0])
        assert isinstance(diagram, Diagram)

        # Cluster bbox should encompass all three images
        assert diagram.bbox.x0 == 50
        assert diagram.bbox.y0 == 100
        assert diagram.bbox.x1 == 180
        assert diagram.bbox.y1 == 190

        # All three images should now be consumed
        assert img1.id in result._consumed_blocks
        assert img2.id in result._consumed_blocks
        assert img3.id in result._consumed_blocks

    def test_build_respects_consumed_blocks(
        self, classifier: DiagramClassifier
    ) -> None:
        """Test that build() doesn't include already-consumed images."""
        page_bbox = BBox(0, 0, 200, 300)
        # Three overlapping images
        img1 = Image(id=1, bbox=BBox(50, 100, 100, 150))
        img2 = Image(id=2, bbox=BBox(90, 120, 140, 170))  # Overlaps img1
        img3 = Image(id=3, bbox=BBox(130, 140, 180, 190))  # Overlaps img2

        page_data = PageData(
            page_number=1,
            blocks=[img1, img2, img3],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)
        classifier.score(result)

        # Simulate another classifier consuming img2
        result._consumed_blocks.add(img2.id)

        candidates = result.get_scored_candidates("diagram", valid_only=False)

        # Build diagram from img1 - should only cluster with unclaimed images
        # Since img2 is consumed, img1 and img3 are not connected
        img1_candidate = next(c for c in candidates if img1 in c.source_blocks)
        # Use result.build
        diagram = result.build(img1_candidate)

        assert isinstance(diagram, Diagram)
        # Should only contain img1 (img2 is consumed, so img3 is not reachable)
        assert diagram.bbox == img1.bbox

    def test_separate_images_build_separately(
        self, classifier: DiagramClassifier
    ) -> None:
        """Test that non-overlapping images build as separate diagrams."""
        page_bbox = BBox(0, 0, 400, 300)
        # Two separate images that don't overlap
        img1 = Image(id=1, bbox=BBox(50, 100, 120, 200))
        img2 = Image(id=2, bbox=BBox(250, 100, 320, 200))

        page_data = PageData(
            page_number=1,
            blocks=[img1, img2],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)
        classifier.score(result)

        candidates = result.get_scored_candidates("diagram", valid_only=False)
        assert len(candidates) == 2

        # Build both diagrams using result.build
        diagram1 = result.build(candidates[0])
        diagram2 = result.build(candidates[1])

        # Each should only contain its own image
        assert diagram1.bbox == img1.bbox or diagram1.bbox == img2.bbox
        assert diagram2.bbox == img1.bbox or diagram2.bbox == img2.bbox
        assert diagram1.bbox != diagram2.bbox
