"""Tests for the part count classifier."""

import pytest

from build_a_long.pdf_extract.classifier import (
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.classifier.parts.part_count_classifier import (
    PartCountClassifier,
)
from build_a_long.pdf_extract.classifier.test_utils import PageBuilder
from build_a_long.pdf_extract.classifier.text import FontSizeHints
from build_a_long.pdf_extract.extractor.lego_page_elements import PartCount


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
        builder = PageBuilder(page_number=1, width=100, height=200)
        # Part counts below images (different x/X variations)
        builder.add_text("2x", 10, 50, 10, 10, id=3)  # t1
        builder.add_text("2X", 30, 50, 10, 10, id=4)  # t2 uppercase X
        builder.add_text("3×", 50, 50, 10, 10, id=5)  # t3 times symbol
        builder.add_text("hello", 70, 50, 20, 10, id=6)  # t4 not a count

        page = builder.build()
        t1 = page.blocks[0]
        t2 = page.blocks[1]
        t3 = page.blocks[2]
        t4 = page.blocks[3]

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
        classifier = PartCountClassifier(config=config)

        builder = PageBuilder(page_number=1, width=100, height=100)
        builder.add_text("2x", 0, 0, 10, 10, id=1)  # matching_text
        builder.add_text("3x", 0, 0, 15, 15, id=2)  # different_text

        page = builder.build()
        matching_text = page.blocks[0]
        different_text = page.blocks[1]

        result = ClassificationResult(page_data=page)
        classifier.score(result)

        # Construct all candidates
        candidates = result.get_candidates("part_count")
        for candidate in candidates:
            if result.get_constructed(candidate) is None:
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
