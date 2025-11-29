"""Tests for ArrowClassifier."""

from __future__ import annotations

import pytest

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.classifier.steps.arrow_classifier import (
    ArrowClassifier,
    _ArrowScore,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import Arrow
from build_a_long.pdf_extract.extractor.page_blocks import Drawing


@pytest.fixture
def config() -> ClassifierConfig:
    """Create default classifier config."""
    return ClassifierConfig()


@pytest.fixture
def arrow_classifier(config: ClassifierConfig) -> ArrowClassifier:
    """Create an ArrowClassifier instance."""
    return ArrowClassifier(config=config)


def make_page_data(blocks: list) -> PageData:
    """Create a PageData with given blocks."""
    return PageData(
        page_number=1,
        bbox=BBox(x0=0, y0=0, x1=500, y1=500),
        blocks=blocks,
    )


def make_drawing(
    bbox: BBox,
    *,
    fill_color: tuple[float, float, float] | None = (1.0, 1.0, 1.0),
    items: tuple[tuple, ...] | None = None,
) -> Drawing:
    """Create a Drawing block."""
    return Drawing(
        bbox=bbox,
        fill_color=fill_color,
        items=items,
        id=1,
    )


def make_triangular_arrow_items(
    x: float, y: float, width: float = 12.5, height: float = 9.0
) -> tuple[tuple, ...]:
    """Create line items for a triangular arrowhead pointing right.

    The triangle has:
    - Tip at (x + width, y + height/2)
    - Two back corners at (x, y) and (x, y + height)

    Returns 3 line items forming a closed triangle.
    """
    tip = (x + width, y + height / 2)
    top = (x, y)
    bottom = (x, y + height)

    return (
        ("l", top, tip),
        ("l", tip, bottom),
        ("l", bottom, top),
    )


class TestArrowScore:
    """Tests for _ArrowScore."""

    def test_score_calculation(self):
        """Test score combines shape and size scores with weights."""
        score = _ArrowScore(
            shape_score=1.0,
            size_score=0.8,
            direction=0.0,
            tip=(100.0, 50.0),
            shape_weight=0.7,
            size_weight=0.3,
        )
        # 1.0 * 0.7 + 0.8 * 0.3 = 0.7 + 0.24 = 0.94
        assert score.score() == pytest.approx(0.94)

    def test_score_with_low_shape_score(self):
        """Test score with lower shape score."""
        score = _ArrowScore(
            shape_score=0.5,
            size_score=1.0,
            direction=45.0,
            tip=(100.0, 100.0),
            shape_weight=0.6,
            size_weight=0.4,
        )
        # 0.5 * 0.6 + 1.0 * 0.4 = 0.3 + 0.4 = 0.7
        assert score.score() == pytest.approx(0.7)


class TestArrowClassifier:
    """Tests for ArrowClassifier."""

    def test_output_label(self, arrow_classifier: ArrowClassifier):
        """Test classifier output label."""
        assert arrow_classifier.output == "arrow"

    def test_requires_empty(self, arrow_classifier: ArrowClassifier):
        """Test classifier has no dependencies."""
        assert arrow_classifier.requires == frozenset()

    def test_score_finds_triangular_arrowhead(
        self, arrow_classifier: ArrowClassifier, config: ClassifierConfig
    ):
        """Test scoring a typical triangular arrowhead."""
        bbox = BBox(x0=396.0, y0=83.0, x1=408.5, y1=92.0)
        items = make_triangular_arrow_items(396.0, 83.0)
        drawing = make_drawing(bbox, fill_color=(1.0, 1.0, 1.0), items=items)

        page_data = make_page_data([drawing])
        result = ClassificationResult(page_data=page_data)

        arrow_classifier._score(result)

        # Use valid_only=False since candidates haven't been built yet
        candidates = result.get_scored_candidates("arrow", valid_only=False)
        assert len(candidates) == 1
        assert candidates[0].label == "arrow"
        assert candidates[0].score > config.arrow_min_score

    def test_score_rejects_drawing_without_items(
        self, arrow_classifier: ArrowClassifier
    ):
        """Test that drawings without items are rejected."""
        bbox = BBox(x0=100.0, y0=100.0, x1=112.0, y1=109.0)
        drawing = make_drawing(bbox, fill_color=(1.0, 1.0, 1.0), items=None)

        page_data = make_page_data([drawing])
        result = ClassificationResult(page_data=page_data)

        arrow_classifier._score(result)

        candidates = result.get_scored_candidates("arrow", valid_only=False)
        assert len(candidates) == 0

    def test_score_rejects_unfilled_drawing(self, arrow_classifier: ArrowClassifier):
        """Test that unfilled drawings are rejected."""
        bbox = BBox(x0=100.0, y0=100.0, x1=112.0, y1=109.0)
        items = make_triangular_arrow_items(100.0, 100.0)
        drawing = make_drawing(bbox, fill_color=None, items=items)

        page_data = make_page_data([drawing])
        result = ClassificationResult(page_data=page_data)

        arrow_classifier._score(result)

        candidates = result.get_scored_candidates("arrow", valid_only=False)
        assert len(candidates) == 0

    def test_score_rejects_too_large_drawing(self, arrow_classifier: ArrowClassifier):
        """Test that large drawings are rejected."""
        bbox = BBox(x0=100.0, y0=100.0, x1=150.0, y1=140.0)  # 50x40 - too large
        items = make_triangular_arrow_items(100.0, 100.0, width=50, height=40)
        drawing = make_drawing(bbox, fill_color=(1.0, 1.0, 1.0), items=items)

        page_data = make_page_data([drawing])
        result = ClassificationResult(page_data=page_data)

        arrow_classifier._score(result)

        candidates = result.get_scored_candidates("arrow", valid_only=False)
        assert len(candidates) == 0

    def test_score_rejects_too_small_drawing(self, arrow_classifier: ArrowClassifier):
        """Test that tiny drawings are rejected."""
        bbox = BBox(x0=100.0, y0=100.0, x1=103.0, y1=102.0)  # 3x2 - too small
        items = make_triangular_arrow_items(100.0, 100.0, width=3, height=2)
        drawing = make_drawing(bbox, fill_color=(1.0, 1.0, 1.0), items=items)

        page_data = make_page_data([drawing])
        result = ClassificationResult(page_data=page_data)

        arrow_classifier._score(result)

        candidates = result.get_scored_candidates("arrow", valid_only=False)
        assert len(candidates) == 0

    def test_score_calculates_direction_right(self, arrow_classifier: ArrowClassifier):
        """Test that direction is calculated correctly for right-pointing arrow."""
        bbox = BBox(x0=100.0, y0=100.0, x1=112.5, y1=109.0)
        items = make_triangular_arrow_items(100.0, 100.0)
        drawing = make_drawing(bbox, fill_color=(1.0, 1.0, 1.0), items=items)

        page_data = make_page_data([drawing])
        result = ClassificationResult(page_data=page_data)

        arrow_classifier._score(result)

        candidates = result.get_scored_candidates("arrow", valid_only=False)
        assert len(candidates) == 1
        score_details = candidates[0].score_details
        assert isinstance(score_details, _ArrowScore)
        # Right-pointing arrow has direction close to 0°
        assert abs(score_details.direction) < 30

    def test_build_creates_arrow(self, arrow_classifier: ArrowClassifier):
        """Test building an Arrow element from a candidate."""
        bbox = BBox(x0=100.0, y0=100.0, x1=112.5, y1=109.0)
        score_details = _ArrowScore(
            shape_score=1.0,
            size_score=0.9,
            direction=0.0,
            tip=(112.5, 104.5),
        )
        # Create a mock Drawing block for source_blocks
        drawing = make_drawing(
            bbox,
            fill_color=(1.0, 1.0, 1.0),
            items=make_triangular_arrow_items(100.0, 100.0),
        )
        candidate = Candidate(
            bbox=bbox,
            label="arrow",
            score=0.93,
            score_details=score_details,
            source_blocks=[drawing],
        )

        # Create a mock result (build doesn't use it for arrows)
        page_data = make_page_data([drawing])
        result = ClassificationResult(page_data=page_data)

        arrow = arrow_classifier.build(candidate, result)

        assert isinstance(arrow, Arrow)
        assert arrow.bbox == bbox
        assert arrow.direction == 0.0
        assert arrow.tip == (112.5, 104.5)


class TestExtractUniquePoints:
    """Tests for _extract_unique_points helper."""

    def test_extracts_triangle_points(self, arrow_classifier: ArrowClassifier):
        """Test extracting points from triangle line items."""
        items = [
            ("l", (0.0, 0.0), (10.0, 5.0)),
            ("l", (10.0, 5.0), (0.0, 10.0)),
            ("l", (0.0, 10.0), (0.0, 0.0)),
        ]
        points = arrow_classifier._extract_unique_points(items)

        assert len(points) == 3
        assert (0.0, 0.0) in points
        assert (10.0, 5.0) in points
        assert (0.0, 10.0) in points

    def test_deduplicates_close_points(self, arrow_classifier: ArrowClassifier):
        """Test that very close points are deduplicated when rounded."""
        items = [
            ("l", (0.0, 0.0), (10.0, 5.0)),
            ("l", (10.02, 5.04), (0.0, 10.0)),  # Very close to (10.0, 5.0)
        ]
        points = arrow_classifier._extract_unique_points(items)

        # Should deduplicate to 3 unique points (10.02 rounds to 10.0)
        assert len(points) == 3
