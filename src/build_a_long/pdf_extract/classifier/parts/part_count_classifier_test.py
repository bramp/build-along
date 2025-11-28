"""Tests for the part count classifier."""

import pytest

from build_a_long.pdf_extract.classifier import (
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.classifier.parts.part_count_classifier import (
    PartCountClassifier,
)
from build_a_long.pdf_extract.classifier.text import FontSizeHints
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import PartCount
from build_a_long.pdf_extract.extractor.page_blocks import Text


@pytest.fixture
def classifier() -> PartCountClassifier:
    return PartCountClassifier(config=ClassifierConfig())


class TestPartCountClassification:
    """Tests for detecting piece counts like '2x'."""

    def test_detect_multiple_piece_counts(
        self, classifier: PartCountClassifier
    ) -> None:
        """Test that multiple part counts with various formats are detected.

        Verifies that part counts with different notations (2x, 2X, 3×) are
        recognized as valid candidates.
        """
        page_bbox = BBox(0, 0, 100, 200)

        # Part counts below images (different x/X variations)
        t1 = Text(id=3, bbox=BBox(10, 50, 20, 60), text="2x")
        t2 = Text(id=4, bbox=BBox(30, 50, 40, 60), text="2X")  # uppercase X
        t3 = Text(id=5, bbox=BBox(50, 50, 60, 60), text="3×")  # times symbol
        t4 = Text(id=6, bbox=BBox(70, 50, 90, 60), text="hello")  # not a count

        page = PageData(
            page_number=1,
            blocks=[t1, t2, t3, t4],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page)
        classifier.score(result)

        # Verify that the valid part count texts were classified
        t1_candidate = result.get_candidate_for_block(t1, "part_count")
        t2_candidate = result.get_candidate_for_block(t2, "part_count")
        t3_candidate = result.get_candidate_for_block(t3, "part_count")
        t4_candidate = result.get_candidate_for_block(t4, "part_count")

        # The valid counts should be successfully constructed
        assert t1_candidate is not None
        part_count1 = result.build(t1_candidate)
        assert isinstance(part_count1, PartCount)
        assert part_count1.count == 2

        assert t2_candidate is not None
        part_count2 = result.build(t2_candidate)
        assert isinstance(part_count2, PartCount)
        assert part_count2.count == 2

        assert t3_candidate is not None
        part_count3 = result.build(t3_candidate)
        assert isinstance(part_count3, PartCount)
        assert part_count3.count == 3

        # The invalid text should either have no candidate or failed construction
        if t4_candidate is not None:
            # It might be scored low, but if construction is attempted it should fail?
            # PartCountClassifier construction checks score and text parsing.
            # _score_part_count_text checks extract_part_count_value.
            # "hello" returns None, so text_score is 0.0.
            # _construct_single checks text_score == 0.0 -> raise ValueError.
            # So result.construct_candidate should raise or return None if we handle it?
            # construct_candidate raises ValueError on failure.
            with pytest.raises(ValueError):
                result.build(t4_candidate)

    def test_part_count_with_font_hints(self) -> None:
        """Test that PartCountClassifier uses font size hints."""
        # Create font size hints directly
        hints = FontSizeHints(
            part_count_size=10.0,
            catalog_part_count_size=None,
            catalog_element_id_size=None,
            step_number_size=None,
            step_repeat_size=None,
            page_number_size=None,
            remaining_font_sizes={},
        )
        config = ClassifierConfig(font_size_hints=hints)
        classifier = PartCountClassifier(config)

        matching_text = Text(text="2x", bbox=BBox(0, 0, 10, 10), id=1)
        different_text = Text(text="3x", bbox=BBox(0, 0, 15, 15), id=2)

        page_data = PageData(
            page_number=1,
            bbox=BBox(0, 0, 100, 100),
            blocks=[matching_text, different_text],
        )

        result = ClassificationResult(page_data=page_data)
        classifier.score(result)

        # Construct all candidates
        candidates = result.get_candidates("part_count")
        for candidate in candidates:
            if candidate.constructed is None:
                result.build(candidate)

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
