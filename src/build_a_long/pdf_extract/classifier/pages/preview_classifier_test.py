"""Tests for PreviewClassifier."""

from __future__ import annotations

import pytest

from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.classifier.config import PreviewConfig
from build_a_long.pdf_extract.classifier.pages.preview_classifier import (
    PreviewClassifier,
    _PreviewScore,
)
from build_a_long.pdf_extract.classifier.steps.diagram_classifier import (
    DiagramClassifier,
)
from build_a_long.pdf_extract.classifier.steps.step_count_classifier import (
    StepCountClassifier,
)
from build_a_long.pdf_extract.classifier.steps.step_number_classifier import (
    StepNumberClassifier,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import Preview
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image, Text


@pytest.fixture
def config() -> ClassifierConfig:
    """Create default classifier config."""
    return ClassifierConfig()


@pytest.fixture
def preview_classifier(config: ClassifierConfig) -> PreviewClassifier:
    """Create a PreviewClassifier instance."""
    return PreviewClassifier(config=config)


@pytest.fixture
def diagram_classifier(config: ClassifierConfig) -> DiagramClassifier:
    """Create a DiagramClassifier instance."""
    return DiagramClassifier(config=config)


@pytest.fixture
def step_count_classifier(config: ClassifierConfig) -> StepCountClassifier:
    """Create a StepCountClassifier instance."""
    return StepCountClassifier(config=config)


@pytest.fixture
def step_number_classifier(config: ClassifierConfig) -> StepNumberClassifier:
    """Create a StepNumberClassifier instance."""
    return StepNumberClassifier(config=config)


def make_page_data(blocks: list) -> PageData:
    """Create a PageData with given blocks."""
    return PageData(
        page_number=1,
        bbox=BBox(x0=0, y0=0, x1=600, y1=500),
        blocks=blocks,
    )


def make_drawing(
    bbox: BBox,
    *,
    fill_color: tuple[float, float, float] | None = (1.0, 1.0, 1.0),
    stroke_color: tuple[float, float, float] | None = None,
    items: tuple[tuple, ...] | None = None,
    id: int = 1,
) -> Drawing:
    """Create a Drawing block with white fill by default."""
    return Drawing(
        bbox=bbox,
        fill_color=fill_color,
        stroke_color=stroke_color,
        items=items,
        id=id,
    )


def make_image(bbox: BBox, id: int = 1) -> Image:
    """Create an Image block."""
    return Image(
        bbox=bbox,
        image_id=f"image_{id}",
        id=id,
    )


def make_text(bbox: BBox, text: str, id: int = 1, font_size: float = 10.0) -> Text:
    """Create a Text block."""
    return Text(
        bbox=bbox,
        text=text,
        font_name="CeraPro-Bold",
        font_size=font_size,
        color=0,
        id=id,
    )


class TestPreviewScore:
    """Tests for _PreviewScore."""

    def test_score_calculation(self):
        """Test score combines box, fill, and diagram scores with weights."""
        config = PreviewConfig(
            box_shape_weight=0.3,
            fill_color_weight=0.3,
            diagram_weight=0.4,
        )
        score = _PreviewScore(
            box_score=1.0,
            fill_score=1.0,
            has_images=True,
            config=config,
        )
        # 1.0 * 0.3 + 1.0 * 0.3 + 1.0 * 0.4 = 1.0
        assert score.score() == pytest.approx(1.0)

    def test_score_with_no_images(self):
        """Test score with only box and fill scores (no images found)."""
        config = PreviewConfig(
            box_shape_weight=0.3,
            fill_color_weight=0.3,
            diagram_weight=0.4,
        )
        score = _PreviewScore(
            box_score=1.0,
            fill_score=1.0,
            has_images=False,
            config=config,
        )
        # 1.0 * 0.3 + 1.0 * 0.3 + 0.0 * 0.4 = 0.6
        assert score.score() == pytest.approx(0.6)

    def test_score_with_partial_fill(self):
        """Test score with light gray fill (partial score)."""
        config = PreviewConfig(
            box_shape_weight=0.3,
            fill_color_weight=0.3,
            diagram_weight=0.4,
        )
        score = _PreviewScore(
            box_score=1.0,
            fill_score=0.7,  # Light gray
            has_images=True,
            config=config,
        )
        # 1.0 * 0.3 + 0.7 * 0.3 + 1.0 * 0.4 = 0.3 + 0.21 + 0.4 = 0.91
        assert score.score() == pytest.approx(0.91)


class TestPreviewClassifier:
    """Tests for PreviewClassifier."""

    def test_output_label(self, preview_classifier: PreviewClassifier):
        """Test classifier output label."""
        assert preview_classifier.output == "preview"

    def test_requires_dependencies(self, preview_classifier: PreviewClassifier):
        """Test classifier requires diagram."""
        assert "diagram" in preview_classifier.requires

    def test_score_finds_preview_box_with_image(
        self,
        preview_classifier: PreviewClassifier,
        diagram_classifier: DiagramClassifier,
        config: ClassifierConfig,
    ):
        """Test scoring a preview box with an image inside."""
        # Create a white box drawing (preview container)
        # Based on 6433200_page_004: (299.0, 14.5, 459.8, 274.6)
        box_bbox = BBox(x0=299.0, y0=14.5, x1=459.8, y1=274.6)
        box_drawing = make_drawing(box_bbox, fill_color=(1.0, 1.0, 1.0), id=1)

        # Create an image inside the box
        # Based on 6433200_page_004: (319.9, 30.2, 438.9, 258.5)
        image_bbox = BBox(x0=319.9, y0=30.2, x1=438.9, y1=258.5)
        image_block = make_image(image_bbox, id=2)

        page_data = make_page_data([box_drawing, image_block])
        result = ClassificationResult(page_data=page_data)

        # First run diagram classifier to create diagram candidates
        diagram_classifier._score(result)

        # Then run preview classifier
        preview_classifier._score(result)

        candidates = result.get_scored_candidates("preview")
        assert len(candidates) == 1
        assert candidates[0].label == "preview"
        assert candidates[0].score >= config.preview.min_score

    def test_score_rejects_small_boxes(
        self,
        preview_classifier: PreviewClassifier,
        diagram_classifier: DiagramClassifier,
    ):
        """Test that small boxes are rejected."""
        # Create a small white box (below min_width/min_height)
        box_bbox = BBox(x0=0, y0=0, x1=30, y1=30)  # 30x30 < 50x50 minimum
        box_drawing = make_drawing(box_bbox, fill_color=(1.0, 1.0, 1.0), id=1)

        page_data = make_page_data([box_drawing])
        result = ClassificationResult(page_data=page_data)

        diagram_classifier._score(result)
        preview_classifier._score(result)

        candidates = result.get_scored_candidates("preview")
        assert len(candidates) == 0

    def test_score_rejects_non_white_boxes(
        self,
        preview_classifier: PreviewClassifier,
        diagram_classifier: DiagramClassifier,
    ):
        """Test that non-white boxes are rejected."""
        # Create a gray box (not white enough)
        box_bbox = BBox(x0=100, y0=100, x1=300, y1=300)
        box_drawing = make_drawing(box_bbox, fill_color=(0.5, 0.5, 0.5), id=1)

        # Add an image inside
        image_bbox = BBox(x0=120, y0=120, x1=280, y1=280)
        image_block = make_image(image_bbox, id=2)

        page_data = make_page_data([box_drawing, image_block])
        result = ClassificationResult(page_data=page_data)

        diagram_classifier._score(result)
        preview_classifier._score(result)

        candidates = result.get_scored_candidates("preview")
        assert len(candidates) == 0

    def test_score_accepts_light_gray_boxes(
        self,
        preview_classifier: PreviewClassifier,
        diagram_classifier: DiagramClassifier,
        config: ClassifierConfig,
    ):
        """Test that light gray boxes are accepted with lower score."""
        # Create a light gray box (acceptable but not perfect white)
        box_bbox = BBox(x0=100, y0=100, x1=300, y1=300)
        box_drawing = make_drawing(box_bbox, fill_color=(0.85, 0.85, 0.85), id=1)

        # Add an image inside
        image_bbox = BBox(x0=120, y0=120, x1=280, y1=280)
        image_block = make_image(image_bbox, id=2)

        page_data = make_page_data([box_drawing, image_block])
        result = ClassificationResult(page_data=page_data)

        diagram_classifier._score(result)
        preview_classifier._score(result)

        candidates = result.get_scored_candidates("preview")
        assert len(candidates) == 1
        # Light gray gets fill_score of 0.7
        assert candidates[0].score >= config.preview.min_score

    def test_score_rejects_oversized_boxes(
        self,
        preview_classifier: PreviewClassifier,
        diagram_classifier: DiagramClassifier,
    ):
        """Test that oversized boxes (> 60% page size) are rejected."""
        # Create a very large white box (>60% of page width/height)
        # Page is 600x500, so 60% is 360x300
        box_bbox = BBox(x0=0, y0=0, x1=400, y1=350)  # 400x350 > 360x300
        box_drawing = make_drawing(box_bbox, fill_color=(1.0, 1.0, 1.0), id=1)

        # Add an image inside
        image_bbox = BBox(x0=50, y0=50, x1=350, y1=300)
        image_block = make_image(image_bbox, id=2)

        page_data = make_page_data([box_drawing, image_block])
        result = ClassificationResult(page_data=page_data)

        diagram_classifier._score(result)
        preview_classifier._score(result)

        candidates = result.get_scored_candidates("preview")
        assert len(candidates) == 0

    def test_build_creates_preview_with_diagram(
        self,
        preview_classifier: PreviewClassifier,
        diagram_classifier: DiagramClassifier,
    ):
        """Test building a preview element with a diagram inside."""
        # Create a white box drawing (preview container)
        box_bbox = BBox(x0=100, y0=100, x1=300, y1=300)
        box_drawing = make_drawing(box_bbox, fill_color=(1.0, 1.0, 1.0), id=1)

        # Create an image inside the box
        image_bbox = BBox(x0=120, y0=120, x1=280, y1=280)
        image_block = make_image(image_bbox, id=2)

        page_data = make_page_data([box_drawing, image_block])
        result = ClassificationResult(page_data=page_data)

        # Run classifiers - score() also registers them for build()
        diagram_classifier.score(result)
        preview_classifier.score(result)

        # Build the preview
        candidates = result.get_scored_candidates("preview")
        assert len(candidates) == 1

        preview = result.build(candidates[0])

        assert isinstance(preview, Preview)
        assert preview.bbox == box_bbox
        assert preview.diagram is not None
        # Diagram bbox should be within the preview
        assert preview.diagram.bbox.x0 >= box_bbox.x0
        assert preview.diagram.bbox.y0 >= box_bbox.y0

    def test_build_preview_without_diagram(
        self,
        preview_classifier: PreviewClassifier,
        diagram_classifier: DiagramClassifier,
    ):
        """Test building a preview element without any images inside."""
        # Create a white box drawing without any images
        box_bbox = BBox(x0=100, y0=100, x1=300, y1=300)
        box_drawing = make_drawing(box_bbox, fill_color=(1.0, 1.0, 1.0), id=1)

        page_data = make_page_data([box_drawing])
        result = ClassificationResult(page_data=page_data)

        # Run classifiers
        diagram_classifier._score(result)
        preview_classifier._score(result)

        # This won't find a candidate because has_images=False
        # and with default weights, score would be below min_score
        candidates = result.get_scored_candidates("preview")
        # Score without images: 1.0 * 0.3 + 1.0 * 0.3 + 0.0 * 0.4 = 0.6
        # min_score default is 0.5, so it should pass
        assert len(candidates) == 1

        preview = preview_classifier.build(candidates[0], result)
        assert isinstance(preview, Preview)
        assert preview.diagram is None

    def test_groups_similar_drawings(
        self,
        preview_classifier: PreviewClassifier,
        diagram_classifier: DiagramClassifier,
    ):
        """Test that similar bboxes (white fill + border) are grouped."""
        # Create a white fill box
        box_bbox = BBox(x0=100, y0=100, x1=300, y1=300)
        white_fill = make_drawing(box_bbox, fill_color=(1.0, 1.0, 1.0), id=1)

        # Create a slightly larger border box (same region)
        border_bbox = BBox(x0=99, y0=99, x1=301, y1=301)
        border = make_drawing(
            border_bbox,
            fill_color=None,
            stroke_color=(0.0, 0.0, 0.0),
            id=2,
        )

        # Add an image inside
        image_bbox = BBox(x0=120, y0=120, x1=280, y1=280)
        image_block = make_image(image_bbox, id=3)

        page_data = make_page_data([white_fill, border, image_block])
        result = ClassificationResult(page_data=page_data)

        diagram_classifier._score(result)
        preview_classifier._score(result)

        # Should only create one candidate (grouped)
        candidates = result.get_scored_candidates("preview")
        assert len(candidates) == 1

    def test_rejects_box_with_step_count_inside(
        self,
        preview_classifier: PreviewClassifier,
        diagram_classifier: DiagramClassifier,
        step_count_classifier: StepCountClassifier,
    ):
        """Test that boxes containing step_count labels are rejected.

        Step counts (like "2x") indicate this is a subassembly, not a preview.
        Previews show the final model and don't have count labels.
        """
        # Create a white box
        box_bbox = BBox(x0=100, y0=100, x1=300, y1=300)
        box_drawing = make_drawing(box_bbox, fill_color=(1.0, 1.0, 1.0), id=1)

        # Create an image inside (so it would otherwise be a valid preview)
        image_bbox = BBox(x0=120, y0=120, x1=280, y1=280)
        image_block = make_image(image_bbox, id=2)

        # Create a step count "2x" inside the box (indicates subassembly)
        step_count_bbox = BBox(x0=105, y0=105, x1=125, y1=120)
        step_count_text = make_text(step_count_bbox, "2x", id=3, font_size=10.0)

        page_data = make_page_data([box_drawing, image_block, step_count_text])
        result = ClassificationResult(page_data=page_data)

        # Run classifiers - step_count first, then diagram, then preview
        step_count_classifier._score(result)
        diagram_classifier._score(result)
        preview_classifier._score(result)

        # Should reject the box because it contains a step_count
        candidates = result.get_scored_candidates("preview")
        assert len(candidates) == 0

    def test_rejects_box_overlapping_step_number(
        self,
        preview_classifier: PreviewClassifier,
        diagram_classifier: DiagramClassifier,
        step_number_classifier: StepNumberClassifier,
    ):
        """Test that boxes below/overlapping step_numbers are rejected.

        White boxes that overlap with or are below step_numbers are subassemblies,
        not previews. Only boxes ABOVE all step_numbers can be previews.
        """
        # Create a white box that is BELOW the step number
        box_bbox = BBox(x0=100, y0=300, x1=300, y1=500)
        box_drawing = make_drawing(box_bbox, fill_color=(1.0, 1.0, 1.0), id=1)

        # Create an image inside (so it would otherwise be a valid preview)
        image_bbox = BBox(x0=120, y0=320, x1=280, y1=480)
        image_block = make_image(image_bbox, id=2)

        # Create a step number ABOVE the box
        # Step numbers have a specific format - typically large bold numbers
        step_num_bbox = BBox(x0=400, y0=100, x1=450, y1=150)
        step_num_text = make_text(step_num_bbox, "3", id=3, font_size=30.0)

        page_data = make_page_data([box_drawing, image_block, step_num_text])
        result = ClassificationResult(page_data=page_data)

        # Run classifiers - step_number first, then diagram, then preview
        step_number_classifier._score(result)
        diagram_classifier._score(result)
        preview_classifier._score(result)

        # Should reject preview candidates because box is below the step_number
        candidates = result.get_scored_candidates("preview")
        assert len(candidates) == 0

    def test_accepts_box_above_step_numbers(
        self,
        preview_classifier: PreviewClassifier,
        diagram_classifier: DiagramClassifier,
        step_number_classifier: StepNumberClassifier,
    ):
        """Test that boxes ABOVE all step_numbers are accepted as previews.

        Previews can appear at the top of instruction pages, above where the
        steps begin. If the white box is entirely above all step_numbers, it
        should be classified as a preview.
        """
        # Create a white box at the TOP of the page
        box_bbox = BBox(x0=100, y0=10, x1=300, y1=100)
        box_drawing = make_drawing(box_bbox, fill_color=(1.0, 1.0, 1.0), id=1)

        # Create an image inside (so it would otherwise be a valid preview)
        image_bbox = BBox(x0=120, y0=20, x1=280, y1=90)
        image_block = make_image(image_bbox, id=2)

        # Create a step number BELOW the box
        step_num_bbox = BBox(x0=400, y0=200, x1=450, y1=250)
        step_num_text = make_text(step_num_bbox, "3", id=3, font_size=30.0)

        page_data = make_page_data([box_drawing, image_block, step_num_text])
        result = ClassificationResult(page_data=page_data)

        # Run classifiers - step_number first, then diagram, then preview
        step_number_classifier._score(result)
        diagram_classifier._score(result)
        preview_classifier._score(result)

        # Should accept preview because it's above all step_numbers
        candidates = result.get_scored_candidates("preview")
        assert len(candidates) == 1
        assert candidates[0].bbox == box_bbox

    def test_accepts_box_on_page_without_step_numbers(
        self,
        preview_classifier: PreviewClassifier,
        diagram_classifier: DiagramClassifier,
        step_count_classifier: StepCountClassifier,
        config: ClassifierConfig,
    ):
        """Test previews are accepted on pages without step_numbers (INFO pages).

        This is the correct scenario for a preview - a white box with a diagram
        on an INFO page that doesn't have any step numbers.
        """
        # Create a white box
        box_bbox = BBox(x0=100, y0=100, x1=300, y1=300)
        box_drawing = make_drawing(box_bbox, fill_color=(1.0, 1.0, 1.0), id=1)

        # Create an image inside
        image_bbox = BBox(x0=120, y0=120, x1=280, y1=280)
        image_block = make_image(image_bbox, id=2)

        # NO step_number on this page - it's an INFO page

        page_data = make_page_data([box_drawing, image_block])
        result = ClassificationResult(page_data=page_data)

        # Run classifiers
        step_count_classifier._score(result)
        diagram_classifier._score(result)
        preview_classifier._score(result)

        # Should accept the box because this is an INFO page (no step_numbers)
        candidates = result.get_scored_candidates("preview")
        assert len(candidates) == 1
        assert candidates[0].score >= config.preview.min_score
