"""Tests for the open bag classifier."""

from typing import Any

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.classifier.test_utils import PageBuilder
from build_a_long.pdf_extract.extractor.lego_page_elements import OpenBag


def _get_circle_items(x: float, y: float, size: float) -> tuple[tuple[Any, ...], ...]:
    """Create bezier curve items for a circle approximation.

    Real bag icon circles are made of bezier curves (type 'c').
    We create a simple circle approximation for testing.
    """
    # Bezier curves for a circle approximation
    # Each 'c' item represents a bezier curve
    return (
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


class TestOpenBagClassification:
    """Tests for open bag element detection."""

    def test_open_bag_with_circle_and_number(self) -> None:
        """Test identifying an open bag element with circular outline and number."""
        page = (
            PageBuilder(page_number=10, width=552.76, height=496.06)
            # Bag number text inside the circle
            .add_text("1", 90.0, 90.0, 40.0, 40.0, id=0, font_size=50.0)
            # Large circular drawing (bag icon outline) - 200x200 circle at top-left
            .add_drawing(
                15.0,
                15.0,
                200.0,
                200.0,
                id=1,
                items=_get_circle_items(15.0, 15.0, 200.0),
            )
            # Images inside the circle (bag graphic parts)
            .add_image(50.0, 50.0, 130.0, 130.0, id=2)
            .add_image(80.0, 80.0, 70.0, 70.0, id=3)
            .build()
        )

        result = classify_elements(page)

        # Get open_bag candidates
        open_bag_candidates = [
            c for c in result.candidates.get("open_bag", []) if c.constructed
        ]

        # Should have at least one open bag candidate
        assert len(open_bag_candidates) > 0

        # Check the first candidate
        candidate = open_bag_candidates[0]
        assert candidate.constructed is not None
        assert isinstance(candidate.constructed, OpenBag)
        open_bag = candidate.constructed
        assert open_bag.number is not None
        assert open_bag.number.value == 1

        # Check that the circle claims the overlapping blocks
        assert len(candidate.source_blocks) >= 3  # circle + 2 images

    def test_open_bag_without_bag_number(self) -> None:
        """Test that circle without bag number creates a numberless open bag."""
        page = (
            PageBuilder(page_number=4, width=552.76, height=496.06)
            # Large circular drawing (bag icon outline) without a bag number
            .add_drawing(
                15.0,
                15.0,
                200.0,
                200.0,
                id=1,
                items=_get_circle_items(15.0, 15.0, 200.0),
            )
            # Images inside the circle
            .add_image(50.0, 50.0, 130.0, 130.0, id=2)
            .build()
        )

        result = classify_elements(page)

        # Should have an open bag candidate without a bag number
        open_bag_candidates = [
            c for c in result.candidates.get("open_bag", []) if c.constructed
        ]
        assert len(open_bag_candidates) > 0

        # Check that the open bag has no bag number
        candidate = open_bag_candidates[0]
        assert candidate.constructed is not None
        assert isinstance(candidate.constructed, OpenBag)
        open_bag = candidate.constructed
        assert open_bag.number is None

    def test_small_circle_doesnt_create_open_bag(self) -> None:
        """Test that small circles don't create an open bag."""
        page = (
            PageBuilder(page_number=10, width=552.76, height=496.06)
            # Small circle below threshold (default min_circle_size is 150)
            .add_drawing(
                16.0,
                39.0,
                100.0,
                100.0,
                id=1,
                items=_get_circle_items(16.0, 39.0, 100.0),
            )
            .build()
        )

        result = classify_elements(page)

        # Should have no open bag candidates
        open_bag_candidates = result.candidates.get("open_bag", [])
        assert len(open_bag_candidates) == 0

    def test_no_circle_means_no_open_bag(self) -> None:
        """Test that bag number alone (no circle) doesn't create an open bag."""
        page = (
            PageBuilder(page_number=10, width=552.76, height=496.06)
            # Only bag number text, no circular outline
            .add_text("1", 113.93, 104.82, 34.68, 65.07, id=0, font_size=50.0)
            # Images without a circular outline
            .add_image(16.0, 39.0, 237.0, 215.0, id=1)
            .build()
        )

        result = classify_elements(page)

        # Should have bag_number but no open_bag (requires circle)
        bag_candidates = result.candidates.get("bag_number", [])
        assert len(bag_candidates) > 0

        open_bag_candidates = result.candidates.get("open_bag", [])
        assert len(open_bag_candidates) == 0
