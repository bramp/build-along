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
    _ArrowHeadData,
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
    stroke_color: tuple[float, float, float] | None = None,
    items: tuple[tuple, ...] | None = None,
    block_id: int = 1,
) -> Drawing:
    """Create a Drawing block."""
    return Drawing(
        bbox=bbox,
        fill_color=fill_color,
        stroke_color=stroke_color,
        items=items,
        id=block_id,
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
        # Create a mock Drawing block for the head
        bbox = BBox(x0=100.0, y0=50.0, x1=110.0, y1=60.0)
        drawing = make_drawing(bbox, fill_color=(1.0, 1.0, 1.0))
        head = _ArrowHeadData(
            tip=(100.0, 50.0),
            direction=0.0,
            shape_score=1.0,
            size_score=0.8,
            block=drawing,
        )
        score = _ArrowScore(
            heads=[head],
            shape_weight=0.7,
            size_weight=0.3,
        )
        # 1.0 * 0.7 + 0.8 * 0.3 = 0.7 + 0.24 = 0.94
        assert score.score() == pytest.approx(0.94)

    def test_score_with_low_shape_score(self):
        """Test score with lower shape score."""
        # Create a mock Drawing block for the head
        bbox = BBox(x0=100.0, y0=100.0, x1=110.0, y1=110.0)
        drawing = make_drawing(bbox, fill_color=(1.0, 1.0, 1.0))
        head = _ArrowHeadData(
            tip=(100.0, 100.0),
            direction=45.0,
            shape_score=0.5,
            size_score=1.0,
            block=drawing,
        )
        score = _ArrowScore(
            heads=[head],
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
        assert candidates[0].score > config.arrow.min_score

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
        assert len(score_details.heads) == 1
        # Right-pointing arrow has direction close to 0°
        assert abs(score_details.heads[0].direction) < 30

    def test_build_creates_arrow(self, arrow_classifier: ArrowClassifier):
        """Test building an Arrow element from a candidate."""
        bbox = BBox(x0=100.0, y0=100.0, x1=112.5, y1=109.0)
        # Create a mock Drawing block for source_blocks
        drawing = make_drawing(
            bbox,
            fill_color=(1.0, 1.0, 1.0),
            items=make_triangular_arrow_items(100.0, 100.0),
        )
        head_data = _ArrowHeadData(
            tip=(112.5, 104.5),
            direction=0.0,
            shape_score=1.0,
            size_score=0.9,
            block=drawing,
        )
        score_details = _ArrowScore(
            heads=[head_data],
            shaft_block=None,
            tail=None,
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
        assert len(arrow.heads) == 1
        assert arrow.heads[0].direction == 0.0
        assert arrow.heads[0].tip == (112.5, 104.5)


def make_shaft_rect_items(bbox: BBox) -> tuple[tuple, ...]:
    """Create rectangle items for a shaft."""
    return (("re", (bbox.x0, bbox.y0, bbox.x1, bbox.y1), -1),)


class TestShaftDetection:
    """Tests for arrow shaft detection."""

    def test_finds_horizontal_shaft_for_right_pointing_arrow(
        self, arrow_classifier: ArrowClassifier
    ):
        """Test finding a horizontal shaft for a right-pointing arrow."""
        # Arrowhead pointing right at x=376, with tip at x=389
        arrowhead_bbox = BBox(x0=376.47, y0=413.39, x1=388.98, y1=422.49)
        arrowhead_items = (
            ("l", (388.98, 417.94), (376.47, 413.39)),
            ("l", (376.47, 413.39), (377.67, 417.94)),
            ("l", (377.67, 417.94), (376.47, 422.49)),
            ("l", (376.47, 422.49), (388.98, 417.94)),
        )
        arrowhead = make_drawing(
            arrowhead_bbox,
            fill_color=(1.0, 1.0, 1.0),
            items=arrowhead_items,
            block_id=84,
        )

        # Shaft to the left of arrowhead (1 pixel high horizontal line)
        shaft_bbox = BBox(x0=333.98, y0=417.44, x1=377.67, y1=418.44)
        shaft_items = make_shaft_rect_items(shaft_bbox)
        shaft = make_drawing(
            shaft_bbox,
            fill_color=(1.0, 1.0, 1.0),
            items=shaft_items,
            block_id=83,
        )

        page_data = make_page_data([shaft, arrowhead])
        result = ClassificationResult(page_data=page_data)

        arrow_classifier._score(result)

        candidates = result.get_scored_candidates("arrow", valid_only=False)
        assert len(candidates) == 1

        score_details = candidates[0].score_details
        assert isinstance(score_details, _ArrowScore)
        assert len(score_details.heads) == 1
        head_data = score_details.heads[0]
        assert head_data.shaft_block is shaft
        assert head_data.tail is not None
        # Tail should be at the left end of the shaft (far from arrowhead)
        assert head_data.tail[0] == pytest.approx(333.98, abs=1.0)

    def test_no_shaft_when_colors_dont_match(self, arrow_classifier: ArrowClassifier):
        """Test that shaft is not found when colors don't match."""
        # White arrowhead
        arrowhead_bbox = BBox(x0=376.47, y0=413.39, x1=388.98, y1=422.49)
        arrowhead_items = (
            ("l", (388.98, 417.94), (376.47, 413.39)),
            ("l", (376.47, 413.39), (377.67, 417.94)),
            ("l", (377.67, 417.94), (376.47, 422.49)),
            ("l", (376.47, 422.49), (388.98, 417.94)),
        )
        arrowhead = make_drawing(
            arrowhead_bbox,
            fill_color=(1.0, 1.0, 1.0),
            items=arrowhead_items,
            block_id=84,
        )

        # Red shaft (different color)
        shaft_bbox = BBox(x0=333.98, y0=417.44, x1=377.67, y1=418.44)
        shaft_items = make_shaft_rect_items(shaft_bbox)
        shaft = make_drawing(
            shaft_bbox,
            fill_color=(1.0, 0.0, 0.0),  # Red, not white
            items=shaft_items,
            block_id=83,
        )

        page_data = make_page_data([shaft, arrowhead])
        result = ClassificationResult(page_data=page_data)

        arrow_classifier._score(result)

        candidates = result.get_scored_candidates("arrow", valid_only=False)
        assert len(candidates) == 1

        score_details = candidates[0].score_details
        assert isinstance(score_details, _ArrowScore)
        assert len(score_details.heads) == 1
        head_data = score_details.heads[0]
        # No shaft should be found due to color mismatch
        assert head_data.shaft_block is None
        assert head_data.tail is None

    def test_no_shaft_when_too_thick(self, arrow_classifier: ArrowClassifier):
        """Test that thick rectangles are not matched as shafts."""
        arrowhead_bbox = BBox(x0=376.47, y0=413.39, x1=388.98, y1=422.49)
        arrowhead_items = (
            ("l", (388.98, 417.94), (376.47, 413.39)),
            ("l", (376.47, 413.39), (377.67, 417.94)),
            ("l", (377.67, 417.94), (376.47, 422.49)),
            ("l", (376.47, 422.49), (388.98, 417.94)),
        )
        arrowhead = make_drawing(
            arrowhead_bbox,
            fill_color=(1.0, 1.0, 1.0),
            items=arrowhead_items,
            block_id=84,
        )

        # Thick rectangle (10 pixels high, too thick for shaft)
        shaft_bbox = BBox(x0=333.98, y0=412.94, x1=377.67, y1=422.94)
        shaft_items = make_shaft_rect_items(shaft_bbox)
        shaft = make_drawing(
            shaft_bbox,
            fill_color=(1.0, 1.0, 1.0),
            items=shaft_items,
            block_id=83,
        )

        page_data = make_page_data([shaft, arrowhead])
        result = ClassificationResult(page_data=page_data)

        arrow_classifier._score(result)

        candidates = result.get_scored_candidates("arrow", valid_only=False)
        assert len(candidates) == 1

        score_details = candidates[0].score_details
        assert isinstance(score_details, _ArrowScore)
        assert len(score_details.heads) == 1
        head_data = score_details.heads[0]
        # No shaft should be found due to thickness
        assert head_data.shaft_block is None

    def test_no_shaft_when_too_short(self, arrow_classifier: ArrowClassifier):
        """Test that short rectangles are not matched as shafts."""
        arrowhead_bbox = BBox(x0=376.47, y0=413.39, x1=388.98, y1=422.49)
        arrowhead_items = (
            ("l", (388.98, 417.94), (376.47, 413.39)),
            ("l", (376.47, 413.39), (377.67, 417.94)),
            ("l", (377.67, 417.94), (376.47, 422.49)),
            ("l", (376.47, 422.49), (388.98, 417.94)),
        )
        arrowhead = make_drawing(
            arrowhead_bbox,
            fill_color=(1.0, 1.0, 1.0),
            items=arrowhead_items,
            block_id=84,
        )

        # Short rectangle (5 pixels long, too short for shaft)
        shaft_bbox = BBox(x0=372.67, y0=417.44, x1=377.67, y1=418.44)
        shaft_items = make_shaft_rect_items(shaft_bbox)
        shaft = make_drawing(
            shaft_bbox,
            fill_color=(1.0, 1.0, 1.0),
            items=shaft_items,
            block_id=83,
        )

        page_data = make_page_data([shaft, arrowhead])
        result = ClassificationResult(page_data=page_data)

        arrow_classifier._score(result)

        candidates = result.get_scored_candidates("arrow", valid_only=False)
        assert len(candidates) == 1

        score_details = candidates[0].score_details
        assert isinstance(score_details, _ArrowScore)
        assert len(score_details.heads) == 1
        head_data = score_details.heads[0]
        # No shaft should be found due to short length
        assert head_data.shaft_block is None

    def test_shaft_included_in_source_blocks(self, arrow_classifier: ArrowClassifier):
        """Test that detected shaft is included in source_blocks."""
        arrowhead_bbox = BBox(x0=376.47, y0=413.39, x1=388.98, y1=422.49)
        arrowhead_items = (
            ("l", (388.98, 417.94), (376.47, 413.39)),
            ("l", (376.47, 413.39), (377.67, 417.94)),
            ("l", (377.67, 417.94), (376.47, 422.49)),
            ("l", (376.47, 422.49), (388.98, 417.94)),
        )
        arrowhead = make_drawing(
            arrowhead_bbox,
            fill_color=(1.0, 1.0, 1.0),
            items=arrowhead_items,
            block_id=84,
        )

        shaft_bbox = BBox(x0=333.98, y0=417.44, x1=377.67, y1=418.44)
        shaft_items = make_shaft_rect_items(shaft_bbox)
        shaft = make_drawing(
            shaft_bbox,
            fill_color=(1.0, 1.0, 1.0),
            items=shaft_items,
            block_id=83,
        )

        page_data = make_page_data([shaft, arrowhead])
        result = ClassificationResult(page_data=page_data)

        arrow_classifier._score(result)

        candidates = result.get_scored_candidates("arrow", valid_only=False)
        assert len(candidates) == 1

        # Both arrowhead and shaft should be in source_blocks
        source_blocks = candidates[0].source_blocks
        assert len(source_blocks) == 2
        assert arrowhead in source_blocks
        assert shaft in source_blocks
