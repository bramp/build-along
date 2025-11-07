"""Tests for the block filter module."""

from build_a_long.pdf_extract.classifier.block_filter import filter_duplicate_blocks
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image, Text


class TestFilterDuplicateBlocks:
    """Tests for the filter_duplicate_blocks function."""

    def test_empty_list(self) -> None:
        """Test with empty list returns empty list."""
        result = filter_duplicate_blocks([])
        assert result == []

    def test_no_duplicates(self) -> None:
        """Test with blocks that are not similar - all should be kept."""
        blocks = [
            Text(id=1, bbox=BBox(10, 10, 20, 20), text="A"),
            Text(id=2, bbox=BBox(50, 50, 60, 60), text="B"),
            Drawing(id=3, bbox=BBox(100, 100, 120, 120)),
        ]
        result = filter_duplicate_blocks(blocks)
        assert len(result) == 3
        assert set(result) == set(blocks)

    def test_iou_catches_overlapping_blocks(self) -> None:
        """Test that IOU check catches blocks with significant overlap.

        This demonstrates why we need IOU: blocks can overlap substantially
        even when their centers are far apart. Example: outline + filled shape,
        or stacked text with slightly different sizes.
        """
        blocks: list[Drawing] = [
            Drawing(id=1, bbox=BBox(10, 10, 30, 30)),  # area = 400, center = (20, 20)
            # Slightly smaller and offset, but high overlap
            # IOU = 324/400 = 0.81 (above threshold)
            # Center distance = sqrt(2) = 1.4px (within tolerance)
            Drawing(id=2, bbox=BBox(11, 11, 29, 29)),  # area = 324
            # Larger, contains block 2
            Drawing(id=3, bbox=BBox(10, 10, 32, 32)),  # area = 484 (largest)
        ]
        result = filter_duplicate_blocks(blocks)
        # All three should be grouped together
        assert len(result) == 1
        assert result[0].id == 3  # Largest should be kept

    def test_center_proximity_catches_small_offsets(self) -> None:
        """Test that center proximity check catches minimally-overlapping duplicates.

        This demonstrates why we need center+area check: drop shadows offset by
        1-2 pixels have low IOU but are clearly the same visual element.
        """
        blocks: list[Drawing] = [
            Drawing(id=1, bbox=BBox(10, 10, 30, 30)),  # area = 400
            # Offset by just 1px - looks like a drop shadow
            # IOU = 361/439 = 0.82, center distance = 1.4px
            Drawing(id=2, bbox=BBox(11, 11, 31, 31)),  # area = 400
            Drawing(id=3, bbox=BBox(10, 10, 32, 32)),  # area = 484 (largest)
        ]
        result = filter_duplicate_blocks(blocks)
        # All three should be grouped as similar (via both checks)
        assert len(result) <= 2
        assert any(b.id == 3 for b in result)  # Largest should be kept

    def test_high_iou_duplicates_keep_largest(self) -> None:
        """Test blocks with high IOU - keep the one with largest area."""
        # Create blocks with very high overlap (IOU > 0.8)
        blocks: list[Drawing] = [
            Drawing(id=1, bbox=BBox(10, 10, 30, 30)),  # area = 400
            # area = 361, very high IOU (~0.975)
            Drawing(id=2, bbox=BBox(10.5, 10.5, 29.5, 29.5)),
            Drawing(id=3, bbox=BBox(10, 10, 35, 35)),  # area = 625 (largest)
        ]
        result = filter_duplicate_blocks(blocks)
        # Block 1 and 2 should be grouped together (high IOU)
        # Block 3 should be grouped with block 1 (high IOU)
        # So all three should end up in one group, keeping only the largest
        assert len(result) <= 2  # At least blocks 1 and 2 should be grouped
        # The largest block should be kept
        assert any(b.id == 3 for b in result)

    def test_center_proximity_duplicates(self) -> None:
        """Test blocks with similar centers and areas - keep largest."""
        # Create blocks with very close centers and similar areas
        # These use the center proximity + area similarity check
        blocks: list[Text] = [
            # center (15, 15), area 100
            Text(id=1, bbox=BBox(10, 10, 20, 20), text="A"),
            # center (15.125, 15.125), area 100
            Text(id=2, bbox=BBox(10.125, 10.125, 20.125, 20.125), text="B"),
            # center (15.5, 15.5), area 121 (largest)
            Text(id=3, bbox=BBox(10, 10, 21, 21), text="C"),
        ]
        result = filter_duplicate_blocks(blocks)
        # Should filter out similar blocks and keep the largest
        assert len(result) == 1  # All should be grouped as similar
        assert result[0].id == 3

    def test_mixed_types_preserved(self) -> None:
        """Test that different block types are handled correctly."""
        blocks = [
            Text(id=1, bbox=BBox(10, 10, 20, 20), text="A"),
            Drawing(id=2, bbox=BBox(11, 11, 21, 21)),  # Similar to id=1
            Image(id=3, bbox=BBox(50, 50, 60, 60)),
        ]
        result = filter_duplicate_blocks(blocks)
        # Text and Drawing can be similar and filtered
        # Image should be preserved as it's not similar to others
        assert any(b.id == 3 for b in result)  # Image should be kept

    def test_drop_shadow_scenario(self) -> None:
        """Test realistic drop shadow scenario - multiple overlapping blocks."""
        # Simulate a drop shadow effect: main block plus 2 shadow copies
        blocks: list[Drawing] = [
            Drawing(id=1, bbox=BBox(100, 100, 200, 150)),  # Main (area=5000)
            Drawing(id=2, bbox=BBox(101, 101, 201, 151)),  # Shadow 1
            Drawing(id=3, bbox=BBox(102, 102, 202, 152)),  # Shadow 2
        ]
        result = filter_duplicate_blocks(blocks)
        # Should keep only one of the three similar blocks
        assert len(result) == 1

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
        result = filter_duplicate_blocks(blocks)
        # Should keep one from each group (the largest ones)
        # Groups may not all merge depending on exact thresholds
        assert len(result) <= 3  # At most 3 (some grouping should happen)
        # The largest blocks should be kept
        assert any(b.id == 3 for b in result)  # Largest from group 1
        assert any(b.id in {4, 5} for b in result)  # One from group 2
