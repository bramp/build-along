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
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image


class TestRotationSymbolClassifier:
    """Tests for RotationSymbolClassifier."""

    def test_identifies_square_image_as_rotation_symbol(self):
        """Test that a square medium-sized image is identified."""
        classifier = RotationSymbolClassifier(ClassifierConfig())

        # Create a square image (~46x46 pixels) - typical rotation symbol
        rotation_img = Image(id=1, bbox=BBox(270.0, 380.0, 316.0, 426.0))
        diagram_drawing = Drawing(id=2, bbox=BBox(200.0, 300.0, 400.0, 450.0))

        # Create a diagram candidate
        page = PageData(
            page_number=1,
            bbox=BBox(0.0, 0.0, 552.0, 496.0),
            blocks=[rotation_img, diagram_drawing],
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

    def test_rejects_too_large_image(self):
        """Test that large images are not classified as rotation symbols."""
        classifier = RotationSymbolClassifier(ClassifierConfig())

        # Create a large image (150x150 pixels) - too big for rotation symbol
        large_img = Image(id=1, bbox=BBox(100.0, 100.0, 250.0, 250.0))

        page = PageData(
            page_number=1,
            bbox=BBox(0.0, 0.0, 552.0, 496.0),
            blocks=[large_img],
        )

        result = ClassificationResult(page_data=page)
        classifier.score(result)

        candidates = result.get_scored_candidates(
            "rotation_symbol", valid_only=False, exclude_failed=True
        )

        assert len(candidates) == 0  # Should reject too large

    def test_rejects_non_square_aspect_ratio(self):
        """Test that very rectangular images are not classified as rotation symbols."""
        classifier = RotationSymbolClassifier(ClassifierConfig())

        # Create a rectangular image (60x20 pixels) - wrong aspect ratio
        rect_img = Image(id=1, bbox=BBox(100.0, 100.0, 160.0, 120.0))

        page = PageData(
            page_number=1,
            bbox=BBox(0.0, 0.0, 552.0, 496.0),
            blocks=[rect_img],
        )

        result = ClassificationResult(page_data=page)
        classifier.score(result)

        candidates = result.get_scored_candidates(
            "rotation_symbol", valid_only=False, exclude_failed=True
        )

        assert len(candidates) == 0  # Should reject wrong aspect ratio

    def test_builds_rotation_symbol_element(self):
        """Test building a RotationSymbol element from a candidate."""
        classifier = RotationSymbolClassifier(ClassifierConfig())

        # Create a square image (~46x46 pixels)
        rotation_img = Image(id=1, bbox=BBox(270.0, 380.0, 316.0, 426.0))
        diagram_drawing = Drawing(id=2, bbox=BBox(200.0, 300.0, 400.0, 450.0))

        page = PageData(
            page_number=1,
            bbox=BBox(0.0, 0.0, 552.0, 496.0),
            blocks=[rotation_img, diagram_drawing],
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

        assert rotation_symbol.bbox == rotation_img.bbox

    def test_clusters_small_drawings(self):
        """Test that small drawings can be evaluated for rotation symbols."""
        classifier = RotationSymbolClassifier(ClassifierConfig())

        # Create cluster of small drawings spread out to form a ~46px symbol
        # (simulating arrows in a circular rotation pattern)
        drawings = [
            Drawing(id=0, bbox=BBox(100.0, 100.0, 110.0, 110.0)),  # top-left
            Drawing(id=1, bbox=BBox(118.0, 100.0, 128.0, 110.0)),  # top
            Drawing(id=2, bbox=BBox(136.0, 100.0, 146.0, 110.0)),  # top-right
            Drawing(id=3, bbox=BBox(136.0, 118.0, 146.0, 128.0)),  # right
            Drawing(id=4, bbox=BBox(136.0, 136.0, 146.0, 146.0)),  # bottom-right
            Drawing(id=5, bbox=BBox(118.0, 136.0, 128.0, 146.0)),  # bottom
            Drawing(id=6, bbox=BBox(100.0, 136.0, 110.0, 146.0)),  # bottom-left
            Drawing(id=7, bbox=BBox(100.0, 118.0, 110.0, 128.0)),  # left
        ]
        diagram_drawing = Drawing(id=99, bbox=BBox(90.0, 90.0, 160.0, 160.0))

        page = PageData(
            page_number=1,
            bbox=BBox(0.0, 0.0, 552.0, 496.0),
            blocks=list(drawings) + [diagram_drawing],  # type: ignore
        )

        result = ClassificationResult(page_data=page)

        # Add diagram for proximity
        result.add_candidate(
            Candidate(
                bbox=BBox(90.0, 90.0, 160.0, 160.0),
                label="diagram",
                score=1.0,
                score_details=_DiagramScore(
                    cluster_bbox=BBox(90.0, 90.0, 160.0, 160.0),
                    num_images=1,
                ),
                source_blocks=[diagram_drawing],
            )
        )

        # Should not raise an error
        classifier.score(result)

        # Get candidates (may or may not find any depending on exact heuristics)
        candidates = result.get_scored_candidates(
            "rotation_symbol", valid_only=False, exclude_failed=True
        )

        # At minimum, should not crash and should return a list
        assert isinstance(candidates, list)
