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
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image


@pytest.fixture
def classifier() -> DiagramClassifier:
    return DiagramClassifier(config=ClassifierConfig())


class TestDiagramClassification:
    """Tests for diagram detection and clustering."""

    def test_single_image_creates_diagram(self, classifier: DiagramClassifier) -> None:
        """Test that a single image creates a diagram candidate."""
        page_bbox = BBox(0, 0, 200, 300)
        # Medium-sized image (25% of page area)
        img = Image(id=1, bbox=BBox(50, 100, 150, 200))

        page_data = PageData(
            page_number=1,
            blocks=[img],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)
        classifier.score(result)

        candidates = result.get_candidates("diagram")
        assert len(candidates) == 1
        assert candidates[0].bbox == img.bbox

    def test_clustered_images_create_single_diagram(
        self, classifier: DiagramClassifier
    ) -> None:
        """Test that overlapping images are clustered into a single diagram."""
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

        candidates = result.get_candidates("diagram")
        # Should create a single diagram cluster
        assert len(candidates) == 1
        # Cluster bbox should encompass all three images
        cluster_bbox = candidates[0].bbox
        assert cluster_bbox.x0 == 50
        assert cluster_bbox.y0 == 100
        assert cluster_bbox.x1 == 180
        assert cluster_bbox.y1 == 190

    def test_separate_images_create_multiple_diagrams(
        self, classifier: DiagramClassifier
    ) -> None:
        """Test that non-overlapping images create separate diagram candidates."""
        page_bbox = BBox(0, 0, 400, 300)  # 120,000 area
        # Two separate images that don't overlap (both > 3% = 3,600 area)
        img1 = Image(id=1, bbox=BBox(50, 100, 120, 200))  # 7,000 area
        img2 = Image(id=2, bbox=BBox(250, 100, 320, 200))  # 7,000 area

        page_data = PageData(
            page_number=1,
            blocks=[img1, img2],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)
        classifier.score(result)

        candidates = result.get_candidates("diagram")
        # Should create two separate diagrams
        assert len(candidates) == 2

    def test_filters_out_full_page_images(self, classifier: DiagramClassifier) -> None:
        """Test that full-page images (>95% of page) are filtered out."""
        page_bbox = BBox(0, 0, 200, 300)
        # Image covering 95% of page (background/border)
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

        candidates = result.get_candidates("diagram")
        # Should only have the normal image
        assert len(candidates) == 1
        assert candidates[0].bbox == normal_img.bbox

    def test_filters_out_small_images(self, classifier: DiagramClassifier) -> None:
        """Test that very small images (<3% of page) are filtered out."""
        page_bbox = BBox(0, 0, 200, 300)  # 60,000 area
        # Tiny image (< 3% = 1,800 area)
        tiny_img = Image(id=1, bbox=BBox(50, 100, 60, 150))  # 500 area
        # Normal image (> 3%)
        normal_img = Image(id=2, bbox=BBox(100, 100, 150, 200))  # 5,000 area

        page_data = PageData(
            page_number=1,
            blocks=[tiny_img, normal_img],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)
        classifier.score(result)

        candidates = result.get_candidates("diagram")
        # Should only have the normal image
        assert len(candidates) == 1
        assert candidates[0].bbox == normal_img.bbox

    def test_filters_out_progress_bar_overlap(
        self, classifier: DiagramClassifier
    ) -> None:
        """Test that images overlapping the progress bar are filtered out."""
        page_bbox = BBox(0, 0, 200, 300)
        # Image that would overlap with progress bar
        img_near_bar = Image(id=1, bbox=BBox(50, 280, 100, 295))
        # Normal image away from progress bar
        normal_img = Image(id=2, bbox=BBox(50, 100, 100, 150))
        # Mock progress bar drawing
        progress_bar_drawing = Drawing(id=99, bbox=BBox(0, 285, 200, 300))

        page_data = PageData(
            page_number=1,
            blocks=[img_near_bar, normal_img, progress_bar_drawing],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)

        # Mock a progress bar candidate
        from build_a_long.pdf_extract.classifier.candidate import Candidate
        from build_a_long.pdf_extract.classifier.score import Score

        class _MockScore(Score):
            def score(self):
                return 1.0

        result.add_candidate(
            Candidate(
                bbox=progress_bar_drawing.bbox,
                label="progress_bar",
                score=1.0,
                score_details=_MockScore(),
                source_blocks=[progress_bar_drawing],
            )
        )

        classifier.score(result)

        candidates = result.get_candidates("diagram")
        # Should only have the normal image (img_near_bar filtered out)
        assert len(candidates) == 1
        assert candidates[0].bbox == normal_img.bbox
