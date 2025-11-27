"""Tests for the step number classifier."""

import pytest

from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.classifier.font_size_hints import FontSizeHints
from build_a_long.pdf_extract.classifier.steps.step_number_classifier import (
    StepNumberClassifier,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import StepNumber
from build_a_long.pdf_extract.extractor.page_blocks import Text


@pytest.fixture
def classifier() -> StepNumberClassifier:
    return StepNumberClassifier(config=ClassifierConfig())


class TestStepNumberClassification:
    """Tests for step number detection."""

    def test_detect_step_numbers(self, classifier: StepNumberClassifier) -> None:
        """Test that multiple step numbers are correctly identified.

        Verifies that step numbers with different font sizes are detected.
        """
        page_bbox = BBox(0, 0, 200, 300)

        # Candidate step numbers
        big_step = Text(id=1, bbox=BBox(50, 100, 70, 120), text="12")  # height 20
        small_step = Text(id=2, bbox=BBox(80, 100, 88, 108), text="3")  # height 8

        # Non-step text
        text_block = Text(id=3, bbox=BBox(10, 10, 20, 20), text="abc")

        page = PageData(
            page_number=5,
            blocks=[big_step, small_step, text_block],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page)
        classifier.score(result)

        # Construct all candidates
        for candidate in result.get_candidates("step_number"):
            if candidate.constructed is None:
                result.build(candidate)

        # Should have 2 step_numbers successfully constructed
        assert result.count_successful_candidates("step_number") == 2

        # Verify that the specific blocks were classified correctly
        step_winners = result.get_winners_by_score("step_number", StepNumber)
        assert len(step_winners) == 2

        # Check that big_step and small_step are the source blocks
        big_candidate = result.get_candidate_for_block(big_step, "step_number")
        small_candidate = result.get_candidate_for_block(small_step, "step_number")
        assert big_candidate is not None and big_candidate.constructed is not None
        assert small_candidate is not None and small_candidate.constructed is not None

        # Verify the step numbers have the correct values
        step_values = {winner.value for winner in step_winners}
        assert step_values == {3, 12}

    def test_step_number_with_font_hints(self) -> None:
        """Test that StepNumberClassifier uses font size hints."""
        hints = FontSizeHints(
            part_count_size=None,
            catalog_part_count_size=None,
            catalog_element_id_size=None,
            step_number_size=15.0,
            step_repeat_size=None,
            page_number_size=None,
            remaining_font_sizes={},
        )
        config = ClassifierConfig(font_size_hints=hints)
        classifier = StepNumberClassifier(config)

        matching_text = Text(text="1", bbox=BBox(10, 10, 25, 25), id=1)
        different_text = Text(text="2", bbox=BBox(10, 40, 30, 60), id=2)

        page_data = PageData(
            page_number=1,
            bbox=BBox(0, 0, 100, 100),
            blocks=[matching_text, different_text],
        )

        result = ClassificationResult(page_data=page_data)
        classifier.score(result)

        # Construct all candidates
        for candidate in result.get_candidates("step_number"):
            if candidate.constructed is None:
                result.build(candidate)

        candidates = result.get_candidates("step_number")
        assert len(candidates) == 2

        matching_candidate = next(
            c for c in candidates if matching_text in c.source_blocks
        )
        different_candidate = next(
            c for c in candidates if different_text in c.source_blocks
        )

        assert matching_candidate.score > different_candidate.score, (
            f"Matching score ({matching_candidate.score}) should be higher than "
            f"different score ({different_candidate.score})"
        )
