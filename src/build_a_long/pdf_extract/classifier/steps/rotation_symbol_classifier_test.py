"""Tests for rotation symbol classifier."""

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.classifier.steps.diagram_classifier import (
    _DiagramScore,
)
from build_a_long.pdf_extract.classifier.steps.rotation_symbol_classifier import (
    RotationSymbolClassifier,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Blocks, Drawing, Image


class TestRotationSymbolClassifier:
    """Tests for RotationSymbolClassifier.

    NOTE: Rotation symbols are currently only detected from Drawing clusters,
    not from Images.
    """

    def test_identifies_isolated_drawing_cluster_as_rotation_symbol(self):
        """Test that an isolated square cluster of drawings is identified."""
        classifier = RotationSymbolClassifier(ClassifierConfig())

        # Create a cluster of small drawings forming a ~46x46 square
        # Positioned OUTSIDE the diagram (not overlapping)
        rotation_drawings = [
            Drawing(id=1, bbox=BBox(10.0, 10.0, 25.0, 25.0)),
            Drawing(id=2, bbox=BBox(24.0, 10.0, 40.0, 25.0)),
            Drawing(id=3, bbox=BBox(39.0, 10.0, 56.0, 25.0)),
            Drawing(id=4, bbox=BBox(10.0, 24.0, 25.0, 40.0)),
            Drawing(id=5, bbox=BBox(39.0, 24.0, 56.0, 40.0)),
            Drawing(id=6, bbox=BBox(10.0, 39.0, 25.0, 56.0)),
            Drawing(id=7, bbox=BBox(24.0, 39.0, 40.0, 56.0)),
            Drawing(id=8, bbox=BBox(39.0, 39.0, 56.0, 56.0)),
        ]
        diagram_drawing = Drawing(id=99, bbox=BBox(200.0, 300.0, 400.0, 450.0))
        all_blocks: list[Blocks] = [*rotation_drawings, diagram_drawing]

        page = PageData(
            page_number=1,
            bbox=BBox(0.0, 0.0, 552.0, 496.0),
            blocks=all_blocks,
        )

        result = ClassificationResult(page_data=page)

        # Add a diagram candidate for proximity scoring
        result.add_candidate(
            Candidate(
                bbox=BBox(200.0, 300.0, 400.0, 450.0),
                label="diagram",
                score=1.0,
                score_details=_DiagramScore(
                    cluster_bbox=BBox(200.0, 300.0, 400.0, 450.0),
                    num_images=1,
                ),
                source_blocks=[diagram_drawing],
            )
        )

        classifier.score(result)

        candidates = result.get_scored_candidates(
            "rotation_symbol", valid_only=False, exclude_failed=True
        )

        assert len(candidates) == 1
        assert candidates[0].score > 0.3  # Should have decent score
        # Cluster bbox should be the union of all rotation drawings
        assert candidates[0].bbox.x0 == 10.0
        assert candidates[0].bbox.y0 == 10.0
        assert candidates[0].bbox.x1 == 56.0
        assert candidates[0].bbox.y1 == 56.0

    def test_rejects_too_large_cluster(self):
        """Test that large drawing clusters are not classified as rotation symbols."""
        classifier = RotationSymbolClassifier(ClassifierConfig())

        # Create a large cluster of drawings (150x150 pixels) - too big
        large_drawings: list[Blocks] = [
            Drawing(id=1, bbox=BBox(100.0, 100.0, 175.0, 175.0)),
            Drawing(id=2, bbox=BBox(174.0, 100.0, 250.0, 175.0)),
            Drawing(id=3, bbox=BBox(100.0, 174.0, 175.0, 250.0)),
            Drawing(id=4, bbox=BBox(174.0, 174.0, 250.0, 250.0)),
        ]

        page = PageData(
            page_number=1,
            bbox=BBox(0.0, 0.0, 552.0, 496.0),
            blocks=large_drawings,
        )

        result = ClassificationResult(page_data=page)
        classifier.score(result)

        candidates = result.get_scored_candidates(
            "rotation_symbol", valid_only=False, exclude_failed=True
        )

        assert len(candidates) == 0  # Should reject too large

    def test_rejects_non_square_aspect_ratio(self):
        """Test that very rectangular clusters are not rotation symbols."""
        classifier = RotationSymbolClassifier(ClassifierConfig())

        # Create a rectangular cluster (60x20 pixels) - wrong aspect ratio
        rect_drawings: list[Blocks] = [
            Drawing(id=1, bbox=BBox(100.0, 100.0, 130.0, 120.0)),
            Drawing(id=2, bbox=BBox(129.0, 100.0, 160.0, 120.0)),
        ]

        page = PageData(
            page_number=1,
            bbox=BBox(0.0, 0.0, 552.0, 496.0),
            blocks=rect_drawings,
        )

        result = ClassificationResult(page_data=page)
        classifier.score(result)

        candidates = result.get_scored_candidates(
            "rotation_symbol", valid_only=False, exclude_failed=True
        )

        assert len(candidates) == 0  # Should reject wrong aspect ratio

    def test_rejects_connected_cluster_that_is_too_large(self):
        """Test that connected drawings forming a too-large cluster are rejected.

        If small drawings overlap and form a cluster that exceeds the size
        threshold, they should be rejected.
        """
        classifier = RotationSymbolClassifier(ClassifierConfig())

        # Create small drawings that overlap and form a cluster that's too large
        # Each drawing is under the max size, but together they form ~100x100
        overlapping_drawings: list[Blocks] = [
            Drawing(id=1, bbox=BBox(100.0, 100.0, 140.0, 140.0)),
            Drawing(id=2, bbox=BBox(139.0, 100.0, 180.0, 140.0)),
            Drawing(id=3, bbox=BBox(100.0, 139.0, 140.0, 180.0)),
            Drawing(id=4, bbox=BBox(139.0, 139.0, 180.0, 180.0)),
            Drawing(id=5, bbox=BBox(179.0, 100.0, 200.0, 140.0)),
            Drawing(id=6, bbox=BBox(179.0, 139.0, 200.0, 180.0)),
        ]

        page = PageData(
            page_number=1,
            bbox=BBox(0.0, 0.0, 552.0, 496.0),
            blocks=overlapping_drawings,
        )

        result = ClassificationResult(page_data=page)
        classifier.score(result)

        candidates = result.get_scored_candidates(
            "rotation_symbol", valid_only=False, exclude_failed=True
        )

        # Should NOT find the drawings as rotation symbol because
        # the cluster formed is too large (100x80)
        assert len(candidates) == 0

    def test_images_are_ignored(self):
        """Test that Images are not considered for rotation symbol detection.

        Images can have transparent areas that make their bounding boxes
        overlap with diagrams even when visually disconnected. We only
        look at Drawings which have reliable bounding boxes.
        """
        classifier = RotationSymbolClassifier(ClassifierConfig())

        # Create a perfectly-sized square Image - should NOT be detected
        rotation_sized_image = Image(id=1, bbox=BBox(10.0, 10.0, 56.0, 56.0))
        diagram_drawing = Drawing(id=2, bbox=BBox(200.0, 300.0, 400.0, 450.0))

        page = PageData(
            page_number=1,
            bbox=BBox(0.0, 0.0, 552.0, 496.0),
            blocks=[rotation_sized_image, diagram_drawing],
        )

        result = ClassificationResult(page_data=page)

        # Add a diagram candidate
        result.add_candidate(
            Candidate(
                bbox=BBox(200.0, 300.0, 400.0, 450.0),
                label="diagram",
                score=1.0,
                score_details=_DiagramScore(
                    cluster_bbox=BBox(200.0, 300.0, 400.0, 450.0),
                    num_images=1,
                ),
                source_blocks=[diagram_drawing],
            )
        )

        classifier.score(result)

        candidates = result.get_scored_candidates(
            "rotation_symbol", valid_only=False, exclude_failed=True
        )

        # Images are ignored, so no rotation symbol should be detected
        assert len(candidates) == 0

    def test_builds_rotation_symbol_element(self):
        """Test building a RotationSymbol element from a candidate."""
        classifier = RotationSymbolClassifier(ClassifierConfig())

        # Create a cluster of small drawings forming a ~46x46 square
        rotation_drawings = [
            Drawing(id=1, bbox=BBox(10.0, 10.0, 35.0, 35.0)),
            Drawing(id=2, bbox=BBox(34.0, 10.0, 56.0, 35.0)),
            Drawing(id=3, bbox=BBox(10.0, 34.0, 35.0, 56.0)),
            Drawing(id=4, bbox=BBox(34.0, 34.0, 56.0, 56.0)),
        ]
        diagram_drawing = Drawing(id=99, bbox=BBox(200.0, 300.0, 400.0, 450.0))
        all_blocks: list[Blocks] = [*rotation_drawings, diagram_drawing]

        page = PageData(
            page_number=1,
            bbox=BBox(0.0, 0.0, 552.0, 496.0),
            blocks=all_blocks,
        )

        result = ClassificationResult(page_data=page)

        # Add diagram for proximity
        result.add_candidate(
            Candidate(
                bbox=BBox(200.0, 300.0, 400.0, 450.0),
                label="diagram",
                score=1.0,
                score_details=_DiagramScore(
                    cluster_bbox=BBox(200.0, 300.0, 400.0, 450.0),
                    num_images=1,
                ),
                source_blocks=[diagram_drawing],
            )
        )

        classifier.score(result)

        candidates = result.get_scored_candidates(
            "rotation_symbol", valid_only=False, exclude_failed=True
        )
        assert len(candidates) == 1

        # Build the element
        rotation_symbol = classifier.build(candidates[0], result)

        # Cluster bbox should be union of all rotation drawings
        assert rotation_symbol.bbox == BBox(10.0, 10.0, 56.0, 56.0)
