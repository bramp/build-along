"""Tests for the step number classifier."""

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Text


class TestStepNumberClassification:
    """Tests for step number detection."""

    def test_step_numbers_with_font_size_hints(self) -> None:
        """Test that step numbers are correctly identified when using font size hints.

        With font size hints, we no longer need to compare to page number size.
        All numeric text matching the pattern is considered a potential step number,
        with the font size hint used to boost or lower the score.
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

        assert result.get_label(pn) == "page_number"
        # Both step numbers are detected since we no longer require
        # them to be taller than page numbers
        assert result.get_label(big_step) == "step_number"
        assert result.get_label(small_step) == "step_number"
