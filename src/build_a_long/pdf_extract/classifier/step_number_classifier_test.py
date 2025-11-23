"""Tests for the step number classifier."""

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import PageNumber, StepNumber
from build_a_long.pdf_extract.extractor.page_blocks import Text


class TestStepNumberClassification:
    """Tests for step number detection."""

    def test_step_numbers_with_font_size_hints(self) -> None:
        """Test that multiple step numbers are correctly identified.

        Verifies that step numbers with different font sizes are both detected
        and paired with the page number correctly.
        """
        page_bbox = BBox(0, 0, 200, 300)
        # Page number near bottom, small height (10)
        pn = Text(id=0, bbox=BBox(10, 285, 20, 295), text="5")

        # Candidate step numbers elsewhere
        big_step = Text(id=1, bbox=BBox(50, 100, 70, 120), text="12")  # height 20
        small_step = Text(id=2, bbox=BBox(80, 100, 88, 108), text="3")  # height 8

        page = PageData(
            page_number=5,
            blocks=[pn, big_step, small_step],
            bbox=page_bbox,
        )

        result = classify_elements(page)

        # Test StepNumberClassifier results directly
        # Should have 1 page_number and 2 step_numbers successfully constructed
        assert result.count_successful_candidates("page_number") == 1
        assert result.count_successful_candidates("step_number") == 2

        # Verify that the specific blocks were classified correctly
        page_number_winners = result.get_winners_by_score(
            "page_number", PageNumber, max_count=1
        )
        assert len(page_number_winners) == 1
        page_number_candidate = result.get_candidate_for_block(pn, "page_number")
        assert page_number_candidate is not None
        assert page_number_candidate.source_blocks == [pn]

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
