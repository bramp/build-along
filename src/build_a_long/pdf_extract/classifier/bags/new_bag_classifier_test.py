"""Tests for the new bag classifier."""

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import NewBag
from build_a_long.pdf_extract.extractor.page_blocks import Image, Text


class TestNewBagClassification:
    """Tests for new bag element detection."""

    def test_new_bag_with_images(self) -> None:
        """Test identifying a new bag element with surrounding images."""
        page_bbox = BBox(0, 0, 552.76, 496.06)

        # Bag number text
        bag_number_text = Text(
            id=0,
            bbox=BBox(113.93, 104.82, 148.61, 169.89),
            text="1",
            font_size=50.0,
        )

        # Surrounding images forming the bag graphic
        image1 = Image(id=1, bbox=BBox(16.0, 39.0, 253.0, 254.0))
        image2 = Image(id=2, bbox=BBox(28.35, 15.94, 208.48, 206.98))
        image3 = Image(id=3, bbox=BBox(46.0, 30.0, 132.0, 112.0))

        page_data = PageData(
            page_number=10,
            blocks=[bag_number_text, image1, image2, image3],
            bbox=page_bbox,
        )

        result = classify_elements(page_data)

        # Get new_bag candidates
        new_bag_candidates = [
            c for c in result.candidates.get("new_bag", []) if c.constructed
        ]

        # Should have at least one new bag candidate
        assert len(new_bag_candidates) > 0

        # Check the first candidate
        candidate = new_bag_candidates[0]
        assert candidate.constructed is not None
        assert isinstance(candidate.constructed, NewBag)
        new_bag = candidate.constructed
        assert new_bag.number is not None
        assert new_bag.number.value == 1

    def test_new_bag_without_bag_number(self) -> None:
        """Test that large square images in top-left create a numberless new bag."""
        page_bbox = BBox(0, 0, 552.76, 496.06)

        # Large square-ish images in top-left (typical numberless bag icon)
        # These match the dimensions from 40573/6433200.pdf page 4
        image1 = Image(id=1, bbox=BBox(15.0, 15.0, 254.0, 254.0))  # 239x239 square
        image2 = Image(id=2, bbox=BBox(28.35, 15.94, 208.48, 207.98))  # 180x192

        page_data = PageData(
            page_number=4,
            blocks=[image1, image2],
            bbox=page_bbox,
        )

        result = classify_elements(page_data)

        # Should have a new bag candidate without a bag number
        new_bag_candidates = [
            c for c in result.candidates.get("new_bag", []) if c.constructed
        ]
        assert len(new_bag_candidates) > 0

        # Check that the new bag has no bag number
        candidate = new_bag_candidates[0]
        assert candidate.constructed is not None
        assert isinstance(candidate.constructed, NewBag)
        new_bag = candidate.constructed
        assert new_bag.number is None

    def test_small_images_dont_create_numberless_bag(self) -> None:
        """Test that small images don't create a numberless new bag."""
        page_bbox = BBox(0, 0, 552.76, 496.06)

        # Small images that shouldn't trigger numberless bag detection
        image1 = Image(id=1, bbox=BBox(16.0, 39.0, 100.0, 120.0))  # 84x81
        image2 = Image(id=2, bbox=BBox(28.35, 15.94, 100.0, 100.0))  # 72x84

        page_data = PageData(
            page_number=10,
            blocks=[image1, image2],
            bbox=page_bbox,
        )

        result = classify_elements(page_data)

        # Should have no new bag candidates
        new_bag_candidates = result.candidates.get("new_bag", [])
        assert len(new_bag_candidates) == 0

    def test_new_bag_without_images(self) -> None:
        """Test that bag number alone doesn't create a new bag element."""
        page_bbox = BBox(0, 0, 552.76, 496.06)

        # Only bag number text, no images
        bag_number_text = Text(
            id=0,
            bbox=BBox(113.93, 104.82, 148.61, 169.89),
            text="1",
            font_size=50.0,
        )

        page_data = PageData(
            page_number=10,
            blocks=[bag_number_text],
            bbox=page_bbox,
        )

        result = classify_elements(page_data)

        # Should have bag_number but no new_bag
        bag_candidates = result.candidates.get("bag_number", [])
        assert len(bag_candidates) > 0

        new_bag_candidates = result.candidates.get("new_bag", [])
        assert len(new_bag_candidates) == 0
