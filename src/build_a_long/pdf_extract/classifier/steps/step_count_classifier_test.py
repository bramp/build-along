"""Tests for StepCountClassifier."""

from __future__ import annotations

import pytest

from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.classifier.config import StepCountConfig
from build_a_long.pdf_extract.classifier.steps.step_count_classifier import (
    StepCountClassifier,
    _StepCountScore,
)
from build_a_long.pdf_extract.classifier.text import FontSizeHints
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import StepCount
from build_a_long.pdf_extract.extractor.page_blocks import Text


@pytest.fixture
def config() -> ClassifierConfig:
    """Create default classifier config."""
    return ClassifierConfig()


@pytest.fixture
def config_with_font_hints() -> ClassifierConfig:
    """Create classifier config with font size hints."""
    hints = FontSizeHints(
        part_count_size=10.0,  # Small font for part counts
        catalog_part_count_size=None,
        catalog_element_id_size=None,
        step_number_size=48.0,  # Large font for step numbers
        step_repeat_size=None,
        page_number_size=None,
        remaining_font_sizes={},
    )
    return ClassifierConfig(font_size_hints=hints)


@pytest.fixture
def step_count_classifier(config: ClassifierConfig) -> StepCountClassifier:
    """Create a StepCountClassifier instance."""
    return StepCountClassifier(config=config)


@pytest.fixture
def step_count_classifier_with_hints(
    config_with_font_hints: ClassifierConfig,
) -> StepCountClassifier:
    """Create a StepCountClassifier instance with font hints."""
    return StepCountClassifier(config=config_with_font_hints)


def make_page_data(blocks: list) -> PageData:
    """Create a PageData with given blocks."""
    return PageData(
        page_number=1,
        bbox=BBox(x0=0, y0=0, x1=600, y1=500),
        blocks=blocks,
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


class TestStepCountScore:
    """Tests for _StepCountScore."""

    def test_score_calculation(self):
        """Test score combines text and font_size scores with weights."""
        score = _StepCountScore(
            text_score=1.0,
            font_size_score=0.5,
            config=StepCountConfig(),  # Uses default weights: 0.6 and 0.4
        )
        # Using default weights: 0.6 * 1.0 + 0.4 * 0.5 = 0.6 + 0.2 = 0.8
        assert score.score() == pytest.approx(0.8)

    def test_score_with_all_perfect(self):
        """Test score with all perfect scores."""
        score = _StepCountScore(
            text_score=1.0,
            font_size_score=1.0,
            config=StepCountConfig(),
        )
        assert score.score() == pytest.approx(1.0)


class TestStepCountClassifier:
    """Tests for StepCountClassifier."""

    def test_output_label(self, step_count_classifier: StepCountClassifier):
        """Test classifier output label."""
        assert step_count_classifier.output == "step_count"

    def test_has_no_dependencies(self, step_count_classifier: StepCountClassifier):
        """Test classifier has no dependencies."""
        assert step_count_classifier.requires == set()

    def test_score_finds_valid_count_text(
        self, step_count_classifier: StepCountClassifier, config: ClassifierConfig
    ):
        """Test scoring a valid count text (e.g., '2x')."""
        count_bbox = BBox(x0=100.0, y0=100.0, x1=130.0, y1=120.0)
        count_text = make_text(count_bbox, "2x", font_size=16.0, id=1)

        page_data = make_page_data([count_text])
        result = ClassificationResult(page_data=page_data)

        step_count_classifier._score(result)

        candidates = result.get_scored_candidates("step_count", valid_only=False)
        assert len(candidates) == 1
        assert candidates[0].label == "step_count"
        assert candidates[0].score >= config.step_count.min_score

    def test_score_finds_multiple_count_texts(
        self, step_count_classifier: StepCountClassifier
    ):
        """Test scoring multiple count texts."""
        text1 = make_text(BBox(x0=100.0, y0=100.0, x1=130.0, y1=120.0), "2x", id=1)
        text2 = make_text(BBox(x0=200.0, y0=100.0, x1=230.0, y1=120.0), "4x", id=2)
        text3 = make_text(BBox(x0=300.0, y0=100.0, x1=340.0, y1=120.0), "10x", id=3)

        page_data = make_page_data([text1, text2, text3])
        result = ClassificationResult(page_data=page_data)

        step_count_classifier._score(result)

        candidates = result.get_scored_candidates("step_count", valid_only=False)
        assert len(candidates) == 3

    def test_score_extracts_count_value(
        self, step_count_classifier: StepCountClassifier
    ):
        """Test that count value is correctly extracted when building."""
        count_text = make_text(BBox(x0=100.0, y0=100.0, x1=130.0, y1=120.0), "5x", id=1)

        page_data = make_page_data([count_text])
        result = ClassificationResult(page_data=page_data)

        step_count_classifier._score(result)

        candidates = result.get_scored_candidates("step_count", valid_only=False)
        assert len(candidates) == 1

        # Build the StepCount element and verify the count value
        step_count = step_count_classifier.build(candidates[0], result)
        assert isinstance(step_count, StepCount)
        assert step_count.count == 5

    def test_score_rejects_non_count_text(
        self, step_count_classifier: StepCountClassifier
    ):
        """Test that non-count text is rejected."""
        text1 = make_text(BBox(x0=100.0, y0=100.0, x1=200.0, y1=120.0), "Step 1", id=1)
        text2 = make_text(BBox(x0=100.0, y0=150.0, x1=200.0, y1=170.0), "LEGO", id=2)
        text3 = make_text(BBox(x0=100.0, y0=200.0, x1=200.0, y1=220.0), "Page 5", id=3)

        page_data = make_page_data([text1, text2, text3])
        result = ClassificationResult(page_data=page_data)

        step_count_classifier._score(result)

        candidates = result.get_scored_candidates("step_count", valid_only=False)
        assert len(candidates) == 0

    def test_score_with_lowercase_x(self, step_count_classifier: StepCountClassifier):
        """Test that lowercase 'x' is accepted."""
        count_text = make_text(BBox(x0=100.0, y0=100.0, x1=130.0, y1=120.0), "3x", id=1)

        page_data = make_page_data([count_text])
        result = ClassificationResult(page_data=page_data)

        step_count_classifier._score(result)

        candidates = result.get_scored_candidates("step_count", valid_only=False)
        assert len(candidates) == 1

    def test_build_creates_step_count(
        self, step_count_classifier: StepCountClassifier, config: ClassifierConfig
    ):
        """Test building a StepCount element from a candidate."""
        count_bbox = BBox(x0=100.0, y0=100.0, x1=130.0, y1=120.0)
        count_text = make_text(count_bbox, "2x", font_size=16.0, id=1)

        page_data = make_page_data([count_text])
        result = ClassificationResult(page_data=page_data)

        step_count_classifier._score(result)
        candidates = result.get_scored_candidates("step_count", valid_only=False)
        assert len(candidates) == 1

        step_count = step_count_classifier.build(candidates[0], result)

        assert isinstance(step_count, StepCount)
        assert step_count.bbox == count_bbox
        assert step_count.count == 2


class TestFontSizeScoring:
    """Tests for font size scoring logic."""

    def test_font_size_in_range_scores_high(
        self, step_count_classifier_with_hints: StepCountClassifier
    ):
        """Test that font size in valid range scores 1.0."""
        # Font size between part_count_size (10.0) and step_number_size (48.0)
        # and greater than part_count_size + tolerance (11.0)
        count_text = make_text(
            BBox(x0=100.0, y0=100.0, x1=130.0, y1=120.0), "2x", font_size=24.0, id=1
        )

        page_data = make_page_data([count_text])
        result = ClassificationResult(page_data=page_data)

        step_count_classifier_with_hints._score(result)

        candidates = result.get_scored_candidates("step_count", valid_only=False)
        assert len(candidates) == 1
        score_details = candidates[0].score_details
        assert isinstance(score_details, _StepCountScore)
        assert score_details.font_size_score == 1.0

    def test_font_size_too_small_scores_zero(
        self, step_count_classifier_with_hints: StepCountClassifier
    ):
        """Test that font size below part_count_size scores 0.0."""
        # part_count_size is 10.0, tolerance is 1.0, so <9.0 should score 0.0
        count_text = make_text(
            BBox(x0=100.0, y0=100.0, x1=130.0, y1=120.0), "2x", font_size=8.0, id=1
        )

        page_data = make_page_data([count_text])
        result = ClassificationResult(page_data=page_data)

        step_count_classifier_with_hints._score(result)

        candidates = result.get_scored_candidates("step_count", valid_only=False)
        assert len(candidates) == 1
        score_details = candidates[0].score_details
        assert isinstance(score_details, _StepCountScore)
        assert score_details.font_size_score == 0.0

    def test_font_size_too_large_scores_zero(
        self, step_count_classifier_with_hints: StepCountClassifier
    ):
        """Test that font size above step_number_size scores 0.0."""
        # step_number_size is 48.0, tolerance is 1.0, so >49.0 should score 0.0
        count_text = make_text(
            BBox(x0=100.0, y0=100.0, x1=130.0, y1=120.0), "2x", font_size=60.0, id=1
        )

        page_data = make_page_data([count_text])
        result = ClassificationResult(page_data=page_data)

        step_count_classifier_with_hints._score(result)

        candidates = result.get_scored_candidates("step_count", valid_only=False)
        assert len(candidates) == 1
        score_details = candidates[0].score_details
        assert isinstance(score_details, _StepCountScore)
        assert score_details.font_size_score == 0.0

    def test_font_size_close_to_part_count_scores_medium(
        self, step_count_classifier_with_hints: StepCountClassifier
    ):
        """Test that font size close to part_count_size scores 0.7."""
        # Font size within tolerance of part_count_size (10.0 +/- 1.0)
        count_text = make_text(
            BBox(x0=100.0, y0=100.0, x1=130.0, y1=120.0), "2x", font_size=10.5, id=1
        )

        page_data = make_page_data([count_text])
        result = ClassificationResult(page_data=page_data)

        step_count_classifier_with_hints._score(result)

        candidates = result.get_scored_candidates("step_count", valid_only=False)
        assert len(candidates) == 1
        score_details = candidates[0].score_details
        assert isinstance(score_details, _StepCountScore)
        assert score_details.font_size_score == 0.7

    def test_font_size_with_no_hints_scores_default(
        self, step_count_classifier: StepCountClassifier
    ):
        """Test that font size without hints scores 0.5 (neutral)."""
        count_text = make_text(
            BBox(x0=100.0, y0=100.0, x1=130.0, y1=120.0), "2x", font_size=16.0, id=1
        )

        page_data = make_page_data([count_text])
        result = ClassificationResult(page_data=page_data)

        step_count_classifier._score(result)

        candidates = result.get_scored_candidates("step_count", valid_only=False)
        assert len(candidates) == 1
        score_details = candidates[0].score_details
        assert isinstance(score_details, _StepCountScore)
        assert score_details.font_size_score == 0.5
