"""Tests for the block filter module."""

from build_a_long.pdf_extract.classifier.block_filter import filter_duplicate_blocks
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image, Text


class TestFilterDuplicateBlocks:
    """Tests for the filter_duplicate_blocks function."""

    def test_empty_list(self) -> None:
        """Test with empty list returns empty list."""
        kept, removed = filter_duplicate_blocks([])
        assert kept == []
        assert removed == {}

    def test_no_duplicates(self) -> None:
        """Test with blocks that are not similar - all should be kept."""
        blocks = [
            Text(id=1, bbox=BBox(10, 10, 20, 20), text="A"),
            Text(id=2, bbox=BBox(50, 50, 60, 60), text="B"),
            Drawing(id=3, bbox=BBox(100, 100, 120, 120)),
        ]
        kept, removed = filter_duplicate_blocks(blocks)
        assert len(kept) == 3
        assert set(kept) == set(blocks)
        assert removed == {}

    def test_iou_catches_overlapping_blocks(self) -> None:
        """Test that IOU check catches blocks with very high overlap.

        This demonstrates why we need IOU: blocks can overlap substantially
        even when their centers are far apart. Example: outline + filled shape,
        or stacked text with slightly different sizes.
        """
        blocks: list[Drawing] = [
            Drawing(id=1, bbox=BBox(10, 10, 30, 30)),  # area = 400, center = (20, 20)
            # Very high overlap - almost identical
            # IOU = 380/404 = 0.94 (well above 0.9 threshold)
            Drawing(id=2, bbox=BBox(10.5, 10.5, 30.5, 30.5)),  # area = 400
            # Larger, contains most of block 1
            Drawing(id=3, bbox=BBox(10, 10, 32, 32)),  # area = 484 (largest)
        ]
        kept, removed = filter_duplicate_blocks(blocks)
        # Should filter to reduce duplicates
        assert len(kept) < len(blocks)
        # The largest block should be kept
        assert any(b.id == 3 for b in kept)
        # At least one block should be removed
        assert len(removed) >= 1

    def test_center_proximity_catches_small_offsets(self) -> None:
        """Test that center proximity check catches minimally-overlapping duplicates.

        This demonstrates why we need center+area check: drop shadows offset by
        tiny amounts have similar centers and areas even if IOU is below threshold.
        """
        blocks: list[Drawing] = [
            Drawing(id=1, bbox=BBox(10, 10, 30, 30)),  # area = 400
            # Offset by just 0.2px - very close centers and same area
            # Center distance = 0.28px (well within tolerance)
            # IOU would be very high but this tests the center+area fallback
            Drawing(id=2, bbox=BBox(10.2, 10.2, 30.2, 30.2)),  # area = 400
            Drawing(id=3, bbox=BBox(10, 10, 32, 32)),  # area = 484 (largest)
        ]
        kept, removed = filter_duplicate_blocks(blocks)
        # Should detect duplicates via center proximity or IOU
        assert len(kept) < len(blocks)
        assert any(b.id == 3 for b in kept)  # Largest should be kept

    def test_high_iou_duplicates_keep_largest(self) -> None:
        """Test blocks with high IOU - keep the one with largest area."""
        # Create blocks with very high overlap (IOU > 0.8)
        blocks: list[Drawing] = [
            Drawing(id=1, bbox=BBox(10, 10, 30, 30)),  # area = 400
            # area = 361, very high IOU (~0.975)
            Drawing(id=2, bbox=BBox(10.5, 10.5, 29.5, 29.5)),
            Drawing(id=3, bbox=BBox(10, 10, 35, 35)),  # area = 625 (largest)
        ]
        kept, removed = filter_duplicate_blocks(blocks)
        # Block 1 and 2 should be grouped together (high IOU)
        # Block 3 should be grouped with block 1 (high IOU)
        # So all three should end up in one group, keeping only the largest
        assert len(kept) <= 2  # At least blocks 1 and 2 should be grouped
        # The largest block should be kept
        assert any(b.id == 3 for b in kept)

    def test_center_proximity_duplicates(self) -> None:
        """Test blocks with similar centers and areas - keep largest."""
        # Create blocks with very close centers and similar areas
        # These use the center proximity + area similarity check
        blocks: list[Text] = [
            # center (15, 15), area 100
            Text(id=1, bbox=BBox(10, 10, 20, 20), text="A"),
            # center (15.06, 15.06), area 100 - very close center
            Text(id=2, bbox=BBox(10.06, 10.06, 20.06, 20.06), text="B"),
            # center (15.5, 15.5), area 121 (largest)
            Text(id=3, bbox=BBox(10, 10, 21, 21), text="C"),
        ]
        kept, removed = filter_duplicate_blocks(blocks)
        # Should filter via center proximity since centers are very close
        assert len(kept) < len(blocks)
        # Largest should be kept
        assert any(b.id == 3 for b in kept)

    def test_mixed_types_preserved(self) -> None:
        """Test that different block types are handled correctly."""
        blocks = [
            Text(id=1, bbox=BBox(10, 10, 20, 20), text="A"),
            Drawing(id=2, bbox=BBox(11, 11, 21, 21)),  # Similar to id=1
            Image(id=3, bbox=BBox(50, 50, 60, 60)),
        ]
        kept, removed = filter_duplicate_blocks(blocks)
        # Text and Drawing can be similar and filtered
        # Image should be preserved as it's not similar to others
        assert any(b.id == 3 for b in kept)  # Image should be kept

    def test_drop_shadow_scenario(self) -> None:
        """Test realistic drop shadow scenario - multiple overlapping blocks."""
        # Simulate a drop shadow effect: main block plus 2 shadow copies
        blocks: list[Drawing] = [
            Drawing(id=1, bbox=BBox(100, 100, 200, 150)),  # Main (area=5000)
            Drawing(id=2, bbox=BBox(101, 101, 201, 151)),  # Shadow 1
            Drawing(id=3, bbox=BBox(102, 102, 202, 152)),  # Shadow 2
        ]
        kept, removed = filter_duplicate_blocks(blocks)
        # Should keep only one of the three similar blocks
        assert len(kept) == 1
        assert len(removed) == 2

    def test_multiple_groups_of_duplicates(self) -> None:
        """Test multiple groups of similar blocks - one from each group kept."""
        blocks = [
            # Group 1: three similar blocks at position (10, 10)
            # Using center proximity + area similarity
            Text(id=1, bbox=BBox(10, 10, 20, 20), text="A"),  # area=100
            Text(id=2, bbox=BBox(10.125, 10.125, 20.125, 20.125), text="A"),  # area~100
            Text(id=3, bbox=BBox(10, 10, 22, 22), text="A"),  # area=144 (largest)
            # Group 2: two similar blocks at position (100, 100)
            Drawing(id=4, bbox=BBox(100, 100, 120, 120)),  # area=400
            Drawing(id=5, bbox=BBox(100.25, 100.25, 120.25, 120.25)),  # area~400
        ]
        kept, removed = filter_duplicate_blocks(blocks)
        # Group 1: blocks 2 and 3 should be grouped (id=3 is largest)
        # Group 2: blocks 4 and 5 should be grouped (one kept)
        # Block 1 might not group with 2 depending on threshold
        assert len(kept) <= 3  # At most 3 blocks kept
        # Verify the largest from group 1 is kept
        assert any(b.id == 3 for b in kept)  # Largest from group 1
        # Verify one from group 2 is kept
        assert any(b.id in {4, 5} for b in kept)
        assert len(removed) >= 2  # At least blocks 2 and 5 removed
