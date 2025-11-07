"""Tests for the part count classifier."""

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Text


class TestPartCountClassification:
    """Tests for detecting piece counts like '2x'."""

    def test_detect_multiple_piece_counts(self) -> None:
        page_bbox = BBox(0, 0, 100, 200)
        t1 = Text(id=0, bbox=BBox(10, 50, 20, 60), text="2x")
        t2 = Text(id=1, bbox=BBox(10, 65, 20, 75), text="2X")  # uppercase X
        t3 = Text(id=2, bbox=BBox(30, 50, 40, 60), text="2×")  # times symbol
        t4 = Text(id=3, bbox=BBox(50, 50, 70, 60), text="hello")

        page = PageData(
            page_number=1,
            blocks=[t1, t2, t3, t4],
            bbox=page_bbox,
        )

        result = classify_elements(page)

        assert result.get_label(t1) == "part_count"
        assert result.get_label(t2) == "part_count"
        assert result.get_label(t3) == "part_count"
        assert result.get_label(t4) is None

    def test_similar_bbox_blocks_are_removed(self) -> None:
        """Blocks with similar bboxes should be deduplicated.

        When two blocks have very similar (or identical) bounding boxes,
        they're likely duplicates (e.g., drop shadows, overlapping detections).
        The classifier should keep the first one and remove the duplicate.
        """
        page_bbox = BBox(0, 0, 100, 200)
        # Two blocks at the exact same location - clear duplicates
        t1 = Text(id=0, bbox=BBox(10, 50, 20, 60), text="2x")
        t2 = Text(id=1, bbox=BBox(10, 50, 20, 60), text="2x")  # duplicate
        # Block at different locations
        t3 = Text(id=2, bbox=BBox(30, 50, 40, 60), text="3x")

        page = PageData(
            page_number=1,
            blocks=[t1, t2, t3],
            bbox=page_bbox,
        )

        result = classify_elements(page)

        # First element should be labeled
        assert result.get_label(t1) == "part_count"

        # Duplicate should be removed (not labeled)
        assert result.is_removed(t2), "Duplicate element should be removed"
        assert result.get_label(t2) is None

        # Non-duplicate should be labeled
        assert result.get_label(t3) == "part_count"
