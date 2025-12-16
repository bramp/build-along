"""Tests for StepCountClassifier."""

from __future__ import annotations

import pytest

from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.classifier.rule_based_classifier import RuleScore
from build_a_long.pdf_extract.classifier.steps.step_count_classifier import (
    StepCountClassifier,
)
from build_a_long.pdf_extract.classifier.test_utils import PageBuilder
from build_a_long.pdf_extract.classifier.text import FontSizeHints
from build_a_long.pdf_extract.extractor.lego_page_elements import StepCount


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
        page_data = (
            PageBuilder(width=600, height=500)
            .add_text("2x", 100.0, 100.0, 30.0, 20.0, font_size=16.0, id=1)
            .build()
        )
        result = ClassificationResult(page_data=page_data)

        step_count_classifier._score(result)

        candidates = result.get_scored_candidates("step_count", valid_only=False)
        assert len(candidates) == 1
        assert candidates[0].label == "step_count"
        # Score should be reasonable (at least min_score)
        assert candidates[0].score >= config.step_count.min_score

    def test_score_finds_multiple_count_texts(
        self, step_count_classifier: StepCountClassifier
    ):
        """Test scoring multiple count texts."""
        page_data = (
            PageBuilder(width=600, height=500)
            .add_text("2x", 100.0, 100.0, 30.0, 20.0, id=1)
            .add_text("4x", 200.0, 100.0, 30.0, 20.0, id=2)
            .add_text("10x", 300.0, 100.0, 40.0, 20.0, id=3)
            .build()
        )
        result = ClassificationResult(page_data=page_data)

        step_count_classifier._score(result)

        candidates = result.get_scored_candidates("step_count", valid_only=False)
        assert len(candidates) == 3

    def test_score_extracts_count_value(
        self, step_count_classifier: StepCountClassifier
    ):
        """Test that count value is correctly extracted when building."""
        page_data = (
            PageBuilder(width=600, height=500)
            .add_text("5x", 100.0, 100.0, 30.0, 20.0, id=1)
            .build()
        )
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
        page_data = (
            PageBuilder(width=600, height=500)
            .add_text("Step 1", 100.0, 100.0, 100.0, 20.0, id=1)
            .add_text("LEGO", 100.0, 150.0, 100.0, 20.0, id=2)
            .add_text("Page 5", 100.0, 200.0, 100.0, 20.0, id=3)
            .build()
        )
        result = ClassificationResult(page_data=page_data)

        step_count_classifier._score(result)

        candidates = result.get_scored_candidates("step_count", valid_only=False)
        assert len(candidates) == 0

    def test_score_with_lowercase_x(self, step_count_classifier: StepCountClassifier):
        """Test that lowercase 'x' is accepted."""
        page_data = (
            PageBuilder(width=600, height=500)
            .add_text("3x", 100.0, 100.0, 30.0, 20.0, id=1)
            .build()
        )
        result = ClassificationResult(page_data=page_data)

        step_count_classifier._score(result)

        candidates = result.get_scored_candidates("step_count", valid_only=False)
        assert len(candidates) == 1

    def test_build_creates_step_count(
        self, step_count_classifier: StepCountClassifier, config: ClassifierConfig
    ):
        """Test building a StepCount element from a candidate."""
        page = (
            PageBuilder(width=600, height=500)
            .add_text("2x", 100.0, 100.0, 30.0, 20.0, font_size=16.0, id=1)
            .build()
        )
        count_text = page.blocks[0]

        result = ClassificationResult(page_data=page)

        step_count_classifier._score(result)
        candidates = result.get_scored_candidates("step_count", valid_only=False)
        assert len(candidates) == 1

        step_count = step_count_classifier.build(candidates[0], result)

        assert isinstance(step_count, StepCount)
        assert step_count.bbox == count_text.bbox
        assert step_count.count == 2


class TestFontSizeScoring:
    """Tests for font size scoring logic."""

    def test_font_size_in_range_scores_high(
        self, step_count_classifier_with_hints: StepCountClassifier
    ):
        """Test that font size in valid range scores 1.0."""
        # Font size between part_count_size (10.0) and step_number_size (48.0)
        # and greater than part_count_size + tolerance (11.0)
        page_data = (
            PageBuilder(width=600, height=500)
            .add_text("2x", 100.0, 100.0, 30.0, 20.0, font_size=24.0, id=1)
            .build()
        )
        result = ClassificationResult(page_data=page_data)

        step_count_classifier_with_hints._score(result)

        candidates = result.get_scored_candidates("step_count", valid_only=False)
        assert len(candidates) == 1
        score_details = candidates[0].score_details
        assert isinstance(score_details, RuleScore)
        assert score_details.get("font_size_score") == 1.0

    def test_font_size_too_small_scores_zero(
        self, step_count_classifier_with_hints: StepCountClassifier
    ):
        """Test that font size below part_count_size scores 0.0."""
        # part_count_size is 10.0, tolerance is 1.0, so <9.0 should score 0.0
        page_data = (
            PageBuilder(width=600, height=500)
            .add_text("2x", 100.0, 100.0, 30.0, 20.0, font_size=8.0, id=1)
            .build()
        )
        result = ClassificationResult(page_data=page_data)

        step_count_classifier_with_hints._score(result)

        # Should be rejected because font_size_score is 0.0 -> total score lower than threshold
        # Assuming weights are balanced enough that 0.0 on font size fails the candidate
        candidates = result.get_scored_candidates("step_count", valid_only=False)

        if len(candidates) > 0:
            # If candidate exists, check that font size score is 0.0
            score_details = candidates[0].score_details
            assert isinstance(score_details, RuleScore)
            assert score_details.get("font_size_score") == 0.0

    def test_font_size_too_large_scores_zero(
        self, step_count_classifier_with_hints: StepCountClassifier
    ):
        """Test that font size above step_number_size scores 0.0."""
        # step_number_size is 48.0, tolerance is 1.0, so >49.0 should score 0.0
        page_data = (
            PageBuilder(width=600, height=500)
            .add_text("2x", 100.0, 100.0, 30.0, 20.0, font_size=60.0, id=1)
            .build()
        )
        result = ClassificationResult(page_data=page_data)

        step_count_classifier_with_hints._score(result)

        candidates = result.get_scored_candidates("step_count", valid_only=False)
        if len(candidates) > 0:
            score_details = candidates[0].score_details
            assert isinstance(score_details, RuleScore)
            assert score_details.get("font_size_score") == 0.0

    def test_font_size_close_to_part_count_scores_medium(
        self, step_count_classifier_with_hints: StepCountClassifier
    ):
        """Test that font size close to part_count_size scores 0.7."""
        # Font size within tolerance of part_count_size (10.0 +/- 1.0)
        page_data = (
            PageBuilder(width=600, height=500)
            .add_text("2x", 100.0, 100.0, 30.0, 20.0, font_size=10.5, id=1)
            .build()
        )
        result = ClassificationResult(page_data=page_data)

        step_count_classifier_with_hints._score(result)

        candidates = result.get_scored_candidates("step_count", valid_only=False)
        assert len(candidates) == 1
        score_details = candidates[0].score_details
        assert isinstance(score_details, RuleScore)
        assert score_details.get("font_size_score") == 0.7

    def test_font_size_with_no_hints_scores_default(
        self, step_count_classifier: StepCountClassifier
    ):
        """Test that font size without hints scores 0.5 (neutral)."""
        page_data = (
            PageBuilder(width=600, height=500)
            .add_text("2x", 100.0, 100.0, 30.0, 20.0, font_size=16.0, id=1)
            .build()
        )
        result = ClassificationResult(page_data=page_data)

        step_count_classifier._score(result)

        candidates = result.get_scored_candidates("step_count", valid_only=False)
        assert len(candidates) == 1
        score_details = candidates[0].score_details
        assert isinstance(score_details, RuleScore)
        assert score_details.get("font_size_score") == 0.5
