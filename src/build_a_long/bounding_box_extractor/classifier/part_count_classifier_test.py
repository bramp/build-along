"""Tests for the part count classifier."""

from build_a_long.bounding_box_extractor.classifier.classifier import classify_elements
from build_a_long.bounding_box_extractor.extractor import PageData
from build_a_long.bounding_box_extractor.extractor.bbox import BBox
from build_a_long.bounding_box_extractor.extractor.page_elements import Text


class TestPartCountClassification:
    """Tests for detecting piece counts like '2x'."""

    def test_detect_multiple_piece_counts(self) -> None:
        page_bbox = BBox(0, 0, 100, 200)
        t1 = Text(bbox=BBox(10, 50, 20, 60), text="2x")
        t2 = Text(bbox=BBox(10, 50, 20, 60), text="2X")  # uppercase X
        t3 = Text(bbox=BBox(30, 50, 40, 60), text="2×")  # times symbol
        t4 = Text(bbox=BBox(50, 50, 70, 60), text="hello")

        page = PageData(
            page_number=1,
            elements=[t1, t2, t3, t4],
            bbox=page_bbox,
        )
        classify_elements([page])

        assert t1.label == "part_count"
        assert t2.label == "part_count"
        assert t3.label == "part_count"
        assert t4.label is None
