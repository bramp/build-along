"""Tests for the new bag classifier."""

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import NewBag
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image, Text


def _make_circle_drawing(id: int, x: float, y: float, size: float) -> Drawing:
    """Create a circular drawing with bezier curves.

    Real bag icon circles are made of bezier curves (type 'c').
    We create a simple circle approximation for testing.
    """
    # Bezier curves for a circle approximation
    # Each 'c' item represents a bezier curve
    items = (
        ("m", x, y),  # move to start
        ("c", x + size * 0.55, y, x + size, y + size * 0.45, x + size, y + size * 0.5),
        (
            "c",
            x + size,
            y + size * 0.55,
            x + size * 0.55,
            y + size,
            x + size * 0.5,
            y + size,
        ),
        ("c", x + size * 0.45, y + size, x, y + size * 0.55, x, y + size * 0.5),
        ("c", x, y + size * 0.45, x + size * 0.45, y, x + size * 0.5, y),
    )
    return Drawing(
        id=id,
        bbox=BBox(x, y, x + size, y + size),
        items=items,
    )


class TestNewBagClassification:
    """Tests for new bag element detection."""

    def test_new_bag_with_circle_and_number(self) -> None:
        """Test identifying a new bag element with circular outline and number."""
        page_bbox = BBox(0, 0, 552.76, 496.06)

        # Bag number text inside the circle
        bag_number_text = Text(
            id=0,
            bbox=BBox(90.0, 90.0, 130.0, 130.0),
            text="1",
            font_size=50.0,
        )

        # Large circular drawing (bag icon outline) - 200x200 circle at top-left
        circle = _make_circle_drawing(id=1, x=15.0, y=15.0, size=200.0)

        # Images inside the circle (bag graphic parts)
        image1 = Image(id=2, bbox=BBox(50.0, 50.0, 180.0, 180.0))
        image2 = Image(id=3, bbox=BBox(80.0, 80.0, 150.0, 150.0))

        page_data = PageData(
            page_number=10,
            blocks=[bag_number_text, circle, image1, image2],
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

        # Check that the circle claims the overlapping blocks
        assert len(candidate.source_blocks) >= 3  # circle + 2 images

    def test_new_bag_without_bag_number(self) -> None:
        """Test that circle without bag number creates a numberless new bag."""
        page_bbox = BBox(0, 0, 552.76, 496.06)

        # Large circular drawing (bag icon outline) without a bag number
        circle = _make_circle_drawing(id=1, x=15.0, y=15.0, size=200.0)

        # Images inside the circle
        image1 = Image(id=2, bbox=BBox(50.0, 50.0, 180.0, 180.0))

        page_data = PageData(
            page_number=4,
            blocks=[circle, image1],
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

    def test_small_circle_doesnt_create_new_bag(self) -> None:
        """Test that small circles don't create a new bag."""
        page_bbox = BBox(0, 0, 552.76, 496.06)

        # Small circle below threshold (default min_circle_size is 150)
        small_circle = _make_circle_drawing(id=1, x=16.0, y=39.0, size=100.0)

        page_data = PageData(
            page_number=10,
            blocks=[small_circle],
            bbox=page_bbox,
        )

        result = classify_elements(page_data)

        # Should have no new bag candidates
        new_bag_candidates = result.candidates.get("new_bag", [])
        assert len(new_bag_candidates) == 0

    def test_no_circle_means_no_new_bag(self) -> None:
        """Test that bag number alone (no circle) doesn't create a new bag."""
        page_bbox = BBox(0, 0, 552.76, 496.06)

        # Only bag number text, no circular outline
        bag_number_text = Text(
            id=0,
            bbox=BBox(113.93, 104.82, 148.61, 169.89),
            text="1",
            font_size=50.0,
        )

        # Images without a circular outline
        image1 = Image(id=1, bbox=BBox(16.0, 39.0, 253.0, 254.0))

        page_data = PageData(
            page_number=10,
            blocks=[bag_number_text, image1],
            bbox=page_bbox,
        )

        result = classify_elements(page_data)

        # Should have bag_number but no new_bag (requires circle)
        bag_candidates = result.candidates.get("bag_number", [])
        assert len(bag_candidates) > 0

        new_bag_candidates = result.candidates.get("new_bag", [])
        assert len(new_bag_candidates) == 0
