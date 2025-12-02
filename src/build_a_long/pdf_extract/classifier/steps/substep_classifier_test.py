"""Tests for SubStepClassifier."""

from __future__ import annotations

import pytest

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.classifier.config import SubStepConfig
from build_a_long.pdf_extract.classifier.steps.step_count_classifier import (
    StepCountClassifier,
)
from build_a_long.pdf_extract.classifier.steps.substep_classifier import (
    SubStepClassifier,
    _SubStepScore,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import SubStep
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image, Text


@pytest.fixture
def config() -> ClassifierConfig:
    """Create default classifier config."""
    return ClassifierConfig()


@pytest.fixture
def substep_classifier(config: ClassifierConfig) -> SubStepClassifier:
    """Create a SubStepClassifier instance."""
    return SubStepClassifier(config=config)


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


class TestSubStepScore:
    """Tests for _SubStepScore."""

    def test_score_calculation(self):
        """Test score combines box, count, and diagram scores with weights."""
        config = SubStepConfig(
            box_shape_weight=0.4,
            count_weight=0.3,
            diagram_weight=0.3,
        )
        score = _SubStepScore(
            box_score=0.8,
            count_score=1.0,
            diagram_score=0.5,
            step_count_candidate=None,
            diagram_candidate=None,
            arrow_candidate=None,
            config=config,
        )
        # 0.8 * 0.4 + 1.0 * 0.3 + 0.5 * 0.3 = 0.32 + 0.3 + 0.15 = 0.77
        assert score.score() == pytest.approx(0.77)

    def test_score_with_all_perfect(self):
        """Test score with all perfect scores."""
        config = SubStepConfig(
            box_shape_weight=0.4,
            count_weight=0.3,
            diagram_weight=0.3,
        )
        score = _SubStepScore(
            box_score=1.0,
            count_score=1.0,
            diagram_score=1.0,
            step_count_candidate=None,
            diagram_candidate=None,
            arrow_candidate=None,
            config=config,
        )
        assert score.score() == pytest.approx(1.0)


class TestSubStepClassifier:
    """Tests for SubStepClassifier."""

    def test_output_label(self, substep_classifier: SubStepClassifier):
        """Test classifier output label."""
        assert substep_classifier.output == "substep"

    def test_requires_dependencies(self, substep_classifier: SubStepClassifier):
        """Test classifier requires arrow, step_count, and diagram dependencies."""
        assert "arrow" in substep_classifier.requires
        assert "step_count" in substep_classifier.requires
        assert "diagram" in substep_classifier.requires

    def test_score_finds_substep_box_with_step_count(
        self,
        substep_classifier: SubStepClassifier,
        step_count_classifier: StepCountClassifier,
        config: ClassifierConfig,
    ):
        """Test scoring a substep box with a step_count candidate inside."""
        # Create a box drawing (substep container)
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

        # Then run substep classifier
        substep_classifier._score(result)

        candidates = result.get_scored_candidates("substep", valid_only=False)
        assert len(candidates) == 1
        assert candidates[0].label == "substep"
        assert candidates[0].score >= config.substep.min_score

    def test_build_creates_substep(
        self,
        substep_classifier: SubStepClassifier,
        config: ClassifierConfig,
    ):
        """Test building a SubStep element from a candidate."""
        box_bbox = BBox(x0=100.0, y0=100.0, x1=200.0, y1=160.0)
        box_drawing = make_drawing(box_bbox, id=1)

        score_details = _SubStepScore(
            box_score=0.9,
            count_score=0.0,
            diagram_score=0.0,
            step_count_candidate=None,
            diagram_candidate=None,
            arrow_candidate=None,
            config=config.substep,
        )
        candidate = Candidate(
            bbox=box_bbox,
            label="substep",
            score=0.8,
            score_details=score_details,
            source_blocks=[box_drawing],
        )

        page_data = make_page_data([box_drawing])
        result = ClassificationResult(page_data=page_data)

        substep = substep_classifier.build(candidate, result)

        assert isinstance(substep, SubStep)
        assert substep.bbox == box_bbox
        assert substep.count is None  # No step_count candidate
        assert substep.diagram is None  # No diagram candidate


class TestFindCandidateInside:
    """Tests for _find_candidate_inside helper."""

    def test_finds_candidate_inside_box(
        self,
        substep_classifier: SubStepClassifier,
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

        found = substep_classifier._find_candidate_inside(box_bbox, candidates)

        assert found is not None

    def test_rejects_candidate_outside_box(
        self,
        substep_classifier: SubStepClassifier,
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

        found = substep_classifier._find_candidate_inside(box_bbox, candidates)

        assert found is None
