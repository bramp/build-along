"""Tests for SubAssemblyClassifier."""

from __future__ import annotations

import pytest

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.classifier.config import SubAssemblyConfig
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.classifier.steps.diagram_classifier import (
    DiagramClassifier,
)
from build_a_long.pdf_extract.classifier.steps.step_count_classifier import (
    StepCountClassifier,
)
from build_a_long.pdf_extract.classifier.steps.subassembly_classifier import (
    SubAssemblyClassifier,
    _SubAssemblyScore,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import SubAssembly
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image, Text


@pytest.fixture
def config() -> ClassifierConfig:
    """Create default classifier config."""
    return ClassifierConfig()


@pytest.fixture
def subassembly_classifier(config: ClassifierConfig) -> SubAssemblyClassifier:
    """Create a SubAssemblyClassifier instance."""
    return SubAssemblyClassifier(config=config)


@pytest.fixture
def step_count_classifier(config: ClassifierConfig) -> StepCountClassifier:
    """Create a StepCountClassifier instance."""
    return StepCountClassifier(config=config)


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
    stroke_color: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
    items: tuple[tuple, ...] | None = None,
    id: int = 1,
) -> Drawing:
    """Create a Drawing block with white fill and black stroke by default."""
    return Drawing(
        bbox=bbox,
        fill_color=fill_color,
        stroke_color=stroke_color,
        items=items,
        id=id,
    )


def make_text(
    bbox: BBox,
    text: str,
    font_size: float = 16.0,
    id: int = 1,
) -> Text:
    """Create a Text block."""
    return Text(
        bbox=bbox,
        text=text,
        font_name="CeraPro-Light",
        font_size=font_size,
        id=id,
    )


def make_image(bbox: BBox, id: int = 1) -> Image:
    """Create an Image block."""
    return Image(
        bbox=bbox,
        image_id=f"image_{id}",
        id=id,
    )


class TestSubAssemblyScore:
    """Tests for _SubAssemblyScore."""

    def test_score_calculation(self):
        """Test score combines box, count, and diagram scores with weights."""
        config = SubAssemblyConfig(
            box_shape_weight=0.4,
            count_weight=0.3,
            diagram_weight=0.3,
        )
        score = _SubAssemblyScore(
            box_score=0.8,
            has_step_count=True,
            has_substeps=True,
            has_images=False,
            config=config,
        )
        # 0.8 * 0.4 + 1.0 * 0.3 + 1.0 * 0.3 = 0.32 + 0.3 + 0.3 = 0.92
        assert score.score() == pytest.approx(0.92)

    def test_score_with_all_perfect(self):
        """Test score with all perfect scores."""
        config = SubAssemblyConfig(
            box_shape_weight=0.4,
            count_weight=0.3,
            diagram_weight=0.3,
        )
        score = _SubAssemblyScore(
            box_score=1.0,
            has_step_count=True,
            has_substeps=True,
            has_images=False,
            config=config,
        )
        assert score.score() == pytest.approx(1.0)

    def test_score_with_no_children(self):
        """Test score with only box score (no children found)."""
        config = SubAssemblyConfig(
            box_shape_weight=0.4,
            count_weight=0.3,
            diagram_weight=0.3,
        )
        score = _SubAssemblyScore(
            box_score=0.8,
            has_step_count=False,
            has_substeps=False,
            has_images=False,
            config=config,
        )
        # 0.8 * 0.4 + 0.0 * 0.3 + 0.0 * 0.3 = 0.32
        assert score.score() == pytest.approx(0.32)


class TestSubAssemblyClassifier:
    """Tests for SubAssemblyClassifier."""

    def test_output_label(self, subassembly_classifier: SubAssemblyClassifier):
        """Test classifier output label."""
        assert subassembly_classifier.output == "subassembly"

    def test_requires_dependencies(self, subassembly_classifier: SubAssemblyClassifier):
        """Test classifier requires step_count and substep."""
        assert "step_count" in subassembly_classifier.requires
        assert "substep" in subassembly_classifier.requires
        # Arrow is no longer required (removed from scoring)
        assert "arrow" not in subassembly_classifier.requires

    def test_score_finds_subassembly_box_with_step_count(
        self,
        subassembly_classifier: SubAssemblyClassifier,
        step_count_classifier: StepCountClassifier,
        config: ClassifierConfig,
    ):
        """Test scoring a subassembly box with a step_count candidate inside."""
        # Create a box drawing (subassembly container)
        box_bbox = BBox(x0=219.0, y0=72.0, x1=354.0, y1=131.0)
        box_drawing = make_drawing(box_bbox, id=1)

        # Create a "2x" text inside the box
        count_bbox = BBox(x0=328.0, y0=105.0, x1=345.0, y1=125.0)
        count_text = make_text(count_bbox, "2x", font_size=16.0, id=2)

        # Create an image inside the box
        image_bbox = BBox(x0=227.0, y0=80.0, x1=323.0, y1=122.0)
        image_block = make_image(image_bbox, id=3)

        page_data = make_page_data([box_drawing, count_text, image_block])
        result = ClassificationResult(page_data=page_data)

        # First run step_count classifier to create step_count candidates
        step_count_classifier._score(result)

        # Then run subassembly classifier
        subassembly_classifier._score(result)

        candidates = result.get_scored_candidates("subassembly", valid_only=False)
        assert len(candidates) == 1
        assert candidates[0].label == "subassembly"
        assert candidates[0].score >= config.subassembly.min_score

    def test_build_creates_subassembly(
        self,
        subassembly_classifier: SubAssemblyClassifier,
        config: ClassifierConfig,
    ):
        """Test building a SubAssembly element from a candidate."""
        box_bbox = BBox(x0=100.0, y0=100.0, x1=200.0, y1=160.0)
        box_drawing = make_drawing(box_bbox, id=1)
        # Add an image inside the box to serve as the diagram
        diagram_bbox = BBox(x0=120.0, y0=110.0, x1=180.0, y1=150.0)
        diagram_image = make_image(diagram_bbox, id=2)

        score_details = _SubAssemblyScore(
            box_score=0.9,
            has_step_count=False,
            has_substeps=False,
            has_images=True,  # Now has an image inside
            config=config.subassembly,
        )
        candidate = Candidate(
            bbox=box_bbox,
            label="subassembly",
            score=0.8,
            score_details=score_details,
            source_blocks=[box_drawing],
        )

        class SimpleScore(Score):
            """Simple score for testing."""

            def score(self) -> Weight:
                return 1.0

        diagram_candidate = Candidate(
            bbox=diagram_bbox,
            label="diagram",
            score=0.9,
            score_details=SimpleScore(),
            source_blocks=[diagram_image],
        )

        page_data = make_page_data([box_drawing, diagram_image])
        result = ClassificationResult(page_data=page_data)
        # Register diagram classifier and add the diagram candidate
        result._register_classifier("diagram", DiagramClassifier(config=config))
        result.add_candidate(diagram_candidate)

        subassembly = subassembly_classifier.build(candidate, result)

        assert isinstance(subassembly, SubAssembly)
        assert subassembly.bbox == box_bbox
        assert subassembly.count is None  # No step_count candidate
        # Should have a diagram built from the image inside
        assert subassembly.diagram is not None


class TestFindCandidateInside:
    """Tests for _find_candidate_inside helper."""

    def test_finds_candidate_inside_box(
        self,
        subassembly_classifier: SubAssemblyClassifier,
        step_count_classifier: StepCountClassifier,
    ):
        """Test finding a step_count candidate inside a box."""
        box_bbox = BBox(x0=100.0, y0=100.0, x1=200.0, y1=160.0)
        count_text = make_text(BBox(x0=150.0, y0=120.0, x1=180.0, y1=140.0), "3x", id=1)

        page_data = make_page_data([count_text])
        result = ClassificationResult(page_data=page_data)

        # Create step_count candidates
        step_count_classifier._score(result)
        candidates = result.get_scored_candidates("step_count", valid_only=False)

        found = subassembly_classifier._find_candidate_inside(box_bbox, candidates)

        assert found is not None

    def test_rejects_candidate_outside_box(
        self,
        subassembly_classifier: SubAssemblyClassifier,
        step_count_classifier: StepCountClassifier,
    ):
        """Test that candidates outside the box are not found."""
        box_bbox = BBox(x0=100.0, y0=100.0, x1=200.0, y1=160.0)
        count_text = make_text(BBox(x0=250.0, y0=120.0, x1=280.0, y1=140.0), "3x", id=1)

        page_data = make_page_data([count_text])
        result = ClassificationResult(page_data=page_data)

        # Create step_count candidates
        step_count_classifier._score(result)
        candidates = result.get_scored_candidates("step_count", valid_only=False)

        found = subassembly_classifier._find_candidate_inside(box_bbox, candidates)

        assert found is None
