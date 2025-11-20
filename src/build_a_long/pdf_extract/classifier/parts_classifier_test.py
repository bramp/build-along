"""Tests for the parts classifier (Part pairing logic)."""

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import Part
from build_a_long.pdf_extract.extractor.page_blocks import Image, Text


class TestPartsClassification:
    """Tests for Part assembly (pairing PartCount with Image)."""

    def test_duplicate_part_counts_only_match_once(self) -> None:
        """Test that duplicate part counts don't both pair with the same image.

        When two part count blocks have identical bounding boxes (e.g., drop
        shadows, overlapping detections), both create PartCount candidates,
        but only one should pair with any given image to create a Part.
        """
        page_bbox = BBox(0, 0, 100, 200)

        img1 = Image(id=0, bbox=BBox(10, 30, 20, 45))
        img2 = Image(id=1, bbox=BBox(30, 30, 40, 45))

        # Two blocks at the exact same location - clear duplicates
        t1 = Text(id=2, bbox=BBox(10, 50, 20, 60), text="2x")
        t2 = Text(id=3, bbox=BBox(10, 50, 20, 60), text="2x")  # duplicate
        # Block at different location
        t3 = Text(id=4, bbox=BBox(30, 50, 40, 60), text="3x")

        page = PageData(
            page_number=1,
            blocks=[img1, img2, t1, t2, t3],
            bbox=page_bbox,
        )

        result = classify_elements(page)

        # Verify that both duplicate part counts were recognized
        assert result.count_successful_candidates("part_count") == 3

        # But only 2 Parts should be created (img1 pairs with either t1 or t2,
        # not both; img2 pairs with t3)
        assert result.count_successful_candidates("part") == 2

        # Verify the Part objects have the expected counts
        parts = result.get_winners_by_score("part", Part)
        assert len(parts) == 2
        part_counts = sorted([p.count.count for p in parts])
        assert part_counts == [2, 3]

    def test_part_count_without_nearby_image(self) -> None:
        """Test that part counts require images to be above them.

        Images below the part count should not be paired.
        """
        page_bbox = BBox(0, 0, 200, 200)

        # Part count at the top
        t1 = Text(id=0, bbox=BBox(10, 10, 20, 20), text="2x")

        # Image below the part count (should not pair)
        img1 = Image(id=1, bbox=BBox(10, 50, 20, 65))

        page = PageData(
            page_number=1,
            blocks=[img1, t1],
            bbox=page_bbox,
        )

        result = classify_elements(page)

        # PartCount should be created
        assert result.count_successful_candidates("part_count") == 1

        # But no Part should be created (image is below, not above)
        assert result.count_successful_candidates("part") == 0

    def test_multiple_images_above_picks_closest(self) -> None:
        """Test that when multiple images are above a count, the closest is picked."""
        page_bbox = BBox(0, 0, 100, 200)

        # Two images above the part count, at different vertical positions
        img_far = Image(id=0, bbox=BBox(10, 20, 20, 35))  # farther
        img_near = Image(id=1, bbox=BBox(10, 40, 20, 55))  # closer

        # Part count below both images
        t1 = Text(id=2, bbox=BBox(10, 60, 20, 70), text="2x")

        page = PageData(
            page_number=1,
            blocks=[img_far, img_near, t1],
            bbox=page_bbox,
        )

        result = classify_elements(page)

        # Should create 1 PartCount and 1 Part
        assert result.count_successful_candidates("part_count") == 1
        assert result.count_successful_candidates("part") == 1

        # The Part should use the closer image (img_near)
        parts = result.get_winners_by_score("part", Part)
        assert len(parts) == 1
        part = parts[0]

        # The Part bbox should encompass both the count and the closer image
        # img_near is at y: 40-55, count is at y: 60-70
        # So combined bbox should span y: 40-70
        assert part.bbox.y0 <= 40
        assert part.bbox.y1 >= 70

        # And it should NOT include the far image (which is at y: 20-35)
        # If the far image was used, y0 would be <= 20
        assert part.bbox.y0 > 20

    def test_horizontal_alignment_required(self) -> None:
        """Test that images must be roughly left-aligned with part counts."""
        page_bbox = BBox(0, 0, 200, 200)

        # Image on the left
        img1 = Image(id=0, bbox=BBox(10, 30, 20, 45))

        # Part count far to the right (not aligned)
        t1 = Text(id=1, bbox=BBox(150, 50, 160, 60), text="2x")

        page = PageData(
            page_number=1,
            blocks=[img1, t1],
            bbox=page_bbox,
        )

        result = classify_elements(page)

        # PartCount should be created
        assert result.count_successful_candidates("part_count") == 1

        # But no Part should be created (horizontal misalignment)
        assert result.count_successful_candidates("part") == 0

    def test_one_to_one_pairing_enforcement(self) -> None:
        """Test that one-to-one pairing is enforced (no image/count reuse)."""
        page_bbox = BBox(0, 0, 100, 200)

        # One image
        img1 = Image(id=0, bbox=BBox(10, 30, 20, 45))

        # Two part counts at different vertical positions, both aligned with img1
        t1 = Text(id=1, bbox=BBox(10, 50, 20, 60), text="2x")  # closer
        t2 = Text(id=2, bbox=BBox(10, 70, 20, 80), text="3x")  # farther

        page = PageData(
            page_number=1,
            blocks=[img1, t1, t2],
            bbox=page_bbox,
        )

        result = classify_elements(page)

        # Both PartCounts should be created
        assert result.count_successful_candidates("part_count") == 2

        # But only 1 Part should be created (img1 pairs with t1, the closer one)
        # t2 has no image left to pair with
        assert result.count_successful_candidates("part") == 1

        parts = result.get_winners_by_score("part", Part)
        assert len(parts) == 1
        assert parts[0].count.count == 2  # paired with t1 (the closer one)
