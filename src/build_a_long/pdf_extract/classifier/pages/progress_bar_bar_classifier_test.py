"""Tests for ProgressBarBarClassifier."""

import pytest

from build_a_long.pdf_extract.classifier.classification_result import (
    CandidateFailedError,
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.classifier.pages.progress_bar_bar_classifier import (
    ProgressBarBarClassifier,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import ProgressBarBar
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Text


@pytest.fixture
def classifier() -> ProgressBarBarClassifier:
    return ProgressBarBarClassifier(config=ClassifierConfig())


def make_page_data(
    blocks: list, page_width: float = 800.0, page_height: float = 1000.0
) -> PageData:
    """Create a PageData with given blocks."""
    return PageData(
        page_number=1,
        bbox=BBox(x0=0, y0=0, x1=page_width, y1=page_height),
        blocks=blocks,
    )


class TestProgressBarBarClassifier:
    """Tests for progress bar bar classification."""

    def test_finds_single_bar(self, classifier: ProgressBarBarClassifier):
        """Test that a single valid bar is found."""
        page_width = 800.0
        page_height = 1000.0
        # Bar spanning 95% of the page width at the bottom (760/800 = 0.95)
        bar = Drawing(
            id=1,
            bbox=BBox(20, page_height - 20, page_width - 20, page_height - 15),
        )

        page = make_page_data([bar], page_width=page_width, page_height=page_height)
        result = ClassificationResult(page_data=page)
        classifier._score(result)

        candidate = result.get_candidate_for_block(bar, "progress_bar_bar")
        assert candidate is not None
        assert candidate.score > 0.5

        built = classifier.build(candidate, result)
        assert isinstance(built, ProgressBarBar)
        assert built.bbox == bar.bbox

    def test_merges_left_and_right_bars(self, classifier: ProgressBarBarClassifier):
        """Test that two bar segments at the same y-position are merged.

        This simulates a progress bar split by the indicator, where the left
        segment (completed) and right segment (remaining) are separate blocks.
        The right segment is large enough to be a candidate (>= 40%), and the
        left segment is merged during build even though it's smaller.
        """
        page_width = 800.0
        page_height = 1000.0
        y_top = page_height - 20
        y_bottom = page_height - 15

        # Left bar: 20 to 120 = 100px (12.5% of 800) - too small to be a candidate
        left_bar = Drawing(
            id=1,
            bbox=BBox(20, y_top, 120, y_bottom),
        )
        # Right bar: 140 to 780 = 640px (80% of 800) - large enough to be candidate
        # Gap of 20px for indicator
        right_bar = Drawing(
            id=2,
            bbox=BBox(140, y_top, 780, y_bottom),
        )

        page = make_page_data(
            [left_bar, right_bar], page_width=page_width, page_height=page_height
        )
        result = ClassificationResult(page_data=page)
        classifier._score(result)

        # Only right bar should be a candidate (left is too small)
        left_candidate = result.get_candidate_for_block(left_bar, "progress_bar_bar")
        right_candidate = result.get_candidate_for_block(right_bar, "progress_bar_bar")
        assert left_candidate is None  # Too narrow to be a candidate
        assert right_candidate is not None

        # Build from the right candidate - it should merge with the left block
        built = classifier.build(right_candidate, result)
        assert isinstance(built, ProgressBarBar)

        # The built bar should span from left edge to right edge (760px = 95%)
        assert built.bbox.x0 == 20
        assert built.bbox.x1 == 780
        assert built.bbox.y0 == y_top
        assert built.bbox.y1 == y_bottom

        # The source blocks should include both bars
        assert len(right_candidate.source_blocks) == 2
        assert left_bar in right_candidate.source_blocks
        assert right_bar in right_candidate.source_blocks

    def test_does_not_merge_bars_at_different_y(
        self, classifier: ProgressBarBarClassifier
    ):
        """Test that bars at different y-positions are not merged."""
        page_width = 800.0
        page_height = 1000.0

        # Bar at bottom spanning 95% of page
        bottom_bar = Drawing(
            id=1,
            bbox=BBox(20, page_height - 20, page_width - 20, page_height - 15),
        )
        # Bar higher up (different y) also spanning 95%
        higher_bar = Drawing(
            id=2,
            bbox=BBox(20, page_height - 50, page_width - 20, page_height - 45),
        )

        page = make_page_data(
            [bottom_bar, higher_bar], page_width=page_width, page_height=page_height
        )
        result = ClassificationResult(page_data=page)
        classifier._score(result)

        bottom_candidate = result.get_candidate_for_block(
            bottom_bar, "progress_bar_bar"
        )
        higher_candidate = result.get_candidate_for_block(
            higher_bar, "progress_bar_bar"
        )
        assert bottom_candidate is not None
        assert higher_candidate is not None

        # Build from the bottom candidate - should NOT merge with higher bar
        built = classifier.build(bottom_candidate, result)

        # The built bar should only be the bottom bar
        assert built.bbox == bottom_bar.bbox

        # The higher candidate should NOT be marked as merged
        assert higher_candidate.failure_reason is None

        # Source blocks should only contain the bottom bar
        assert len(bottom_candidate.source_blocks) == 1

    def test_does_not_merge_bars_with_different_heights(
        self, classifier: ProgressBarBarClassifier
    ):
        """Test that bars with different heights are not merged.

        Two bars at the same y-position but with different heights should not
        be merged due to the height tolerance.
        """
        page_width = 800.0
        page_height = 1000.0
        y_top = page_height - 20

        # Main bar: 20 to 780 = 760px (95%), height = 5
        main_bar = Drawing(
            id=1,
            bbox=BBox(20, y_top, 780, y_top + 5),
        )
        # Thicker bar nearby: 0 to 20 = 20px, but height = 15
        # Height difference of 10 > tolerance of 2, so it won't be merged
        thick_bar = Drawing(
            id=2,
            bbox=BBox(0, y_top, 20, y_top + 15),
        )

        page = make_page_data(
            [main_bar, thick_bar], page_width=page_width, page_height=page_height
        )
        result = ClassificationResult(page_data=page)
        classifier._score(result)

        main_candidate = result.get_candidate_for_block(main_bar, "progress_bar_bar")
        assert main_candidate is not None

        # Build should succeed with just the main bar (95% width)
        built = classifier.build(main_candidate, result)
        assert isinstance(built, ProgressBarBar)

        # The thick bar should NOT be included due to height difference
        assert thick_bar not in main_candidate.source_blocks
        assert built.bbox == main_bar.bbox

    def test_rejects_text_blocks(self, classifier: ProgressBarBarClassifier):
        """Test that text blocks are rejected."""
        page_width = 800.0
        page_height = 1000.0
        text = Text(
            id=1,
            bbox=BBox(50, page_height - 20, page_width - 50, page_height - 15),
            text="Some text",
            font_size=12.0,
        )

        page = make_page_data([text], page_width=page_width, page_height=page_height)
        result = ClassificationResult(page_data=page)
        classifier._score(result)

        candidate = result.get_candidate_for_block(text, "progress_bar_bar")
        assert candidate is None

    def test_rejects_bar_too_narrow(self, classifier: ProgressBarBarClassifier):
        """Test that bars not spanning enough page width are rejected."""
        page_width = 800.0
        page_height = 1000.0
        # Bar only 20% of page width (less than min_width_ratio of 0.5)
        narrow_bar = Drawing(
            id=1,
            bbox=BBox(300, page_height - 20, 460, page_height - 15),
        )

        page = make_page_data(
            [narrow_bar], page_width=page_width, page_height=page_height
        )
        result = ClassificationResult(page_data=page)
        classifier._score(result)

        candidate = result.get_candidate_for_block(narrow_bar, "progress_bar_bar")
        # Should be rejected due to width requirement
        assert candidate is None

    def test_rejects_bar_wrong_aspect_ratio(self, classifier: ProgressBarBarClassifier):
        """Test that bars with wrong aspect ratio are rejected."""
        page_width = 800.0
        page_height = 1000.0
        # Square-ish bar (aspect ratio ~1.0, not thin enough)
        square_bar = Drawing(
            id=1,
            bbox=BBox(50, page_height - 500, page_width - 50, page_height - 15),
        )

        page = make_page_data(
            [square_bar], page_width=page_width, page_height=page_height
        )
        result = ClassificationResult(page_data=page)
        classifier._score(result)

        candidate = result.get_candidate_for_block(square_bar, "progress_bar_bar")
        # Should be rejected due to aspect ratio (needs to be thin/wide)
        assert candidate is None

    def test_rejects_merged_bar_too_narrow(self, classifier: ProgressBarBarClassifier):
        """Test that merged bars that don't span enough width are rejected.

        A bar segment that passes the individual width threshold but even with
        merging doesn't reach the 90% merged width threshold.
        """
        page_width = 800.0
        page_height = 1000.0
        y_top = page_height - 20
        y_bottom = page_height - 15

        # Main bar: 20 to 360 = 340px (42.5%) - passes 40% individual threshold
        main_bar = Drawing(id=1, bbox=BBox(20, y_top, 360, y_bottom))
        # Small bar: 380 to 500 = 120px (15%) - will be merged
        # Together: 20 to 500 = 480px (60%) - fails 90% merged threshold
        small_bar = Drawing(id=2, bbox=BBox(380, y_top, 500, y_bottom))

        page = make_page_data(
            [main_bar, small_bar], page_width=page_width, page_height=page_height
        )
        result = ClassificationResult(page_data=page)
        classifier._score(result)

        main_candidate = result.get_candidate_for_block(main_bar, "progress_bar_bar")
        assert main_candidate is not None

        # Build should fail because merged width (60%) < 90%
        with pytest.raises(CandidateFailedError, match="below minimum"):
            classifier.build(main_candidate, result)
