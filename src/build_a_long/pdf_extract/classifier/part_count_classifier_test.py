"""Tests for the part count classifier."""

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import PartCount
from build_a_long.pdf_extract.extractor.page_blocks import Image, Text


class TestPartCountClassification:
    """Tests for detecting piece counts like '2x'."""

    def test_detect_multiple_piece_counts(self) -> None:
        """Test that multiple part counts with various formats are detected.

        Verifies that part counts with different notations (2x, 2X, 3×) are all
        recognized and successfully paired with nearby images to create Part objects.
        """
        page_bbox = BBox(0, 0, 100, 200)

        # Create images above the part counts
        img1 = Image(id=0, bbox=BBox(10, 30, 20, 45))
        img2 = Image(id=1, bbox=BBox(30, 30, 40, 45))
        img3 = Image(id=2, bbox=BBox(50, 30, 60, 45))

        # Part counts below images (different x/X variations)
        t1 = Text(id=3, bbox=BBox(10, 50, 20, 60), text="2x")
        t2 = Text(id=4, bbox=BBox(30, 50, 40, 60), text="2X")  # uppercase X
        t3 = Text(id=5, bbox=BBox(50, 50, 60, 60), text="3×")  # times symbol
        t4 = Text(id=6, bbox=BBox(70, 50, 90, 60), text="hello")  # not a count

        page = PageData(
            page_number=1,
            blocks=[img1, img2, img3, t1, t2, t3, t4],
            bbox=page_bbox,
        )

        result = classify_elements(page)

        # Verify that the valid part count texts were classified
        t1_candidate = result.get_candidate_for_block(t1, "part_count")
        t2_candidate = result.get_candidate_for_block(t2, "part_count")
        t3_candidate = result.get_candidate_for_block(t3, "part_count")
        t4_candidate = result.get_candidate_for_block(t4, "part_count")

        # The valid counts should be successfully constructed
        assert t1_candidate is not None
        assert t1_candidate.constructed is not None
        assert isinstance(t1_candidate.constructed, PartCount)
        assert t1_candidate.constructed.count == 2

        assert t2_candidate is not None
        assert t2_candidate.constructed is not None
        assert isinstance(t2_candidate.constructed, PartCount)
        assert t2_candidate.constructed.count == 2

        assert t3_candidate is not None
        assert t3_candidate.constructed is not None
        assert isinstance(t3_candidate.constructed, PartCount)
        assert t3_candidate.constructed.count == 3

        # The invalid text should either have no candidate or failed construction
        if t4_candidate is not None:
            assert t4_candidate.constructed is None

        # Verify parts were created by pairing with images
        assert result.count_successful_candidates("part") == 3
