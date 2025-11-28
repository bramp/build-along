"""Tests for the block filter module."""

from build_a_long.pdf_extract.classifier.block_filter import (
    filter_background_blocks,
    filter_duplicate_blocks,
    filter_overlapping_text_blocks,
)
from build_a_long.pdf_extract.classifier.removal_reason import RemovalReason
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image, Text


class TestFilterBackgroundBlocks:
    """Tests for the filter_background_blocks function."""

    def test_empty_list(self) -> None:
        """Test with empty list returns empty list."""
        kept, removed = filter_background_blocks([], 100, 100)
        assert kept == []
        assert removed == {}

    def test_no_background_blocks(self) -> None:
        """Test with blocks smaller than threshold - all should be kept."""
        page_width = 100.0
        page_height = 100.0
        blocks = [
            # 50% size
            Drawing(id=1, bbox=BBox(0, 0, 50, 50)),
            # 98% size (below 99% threshold)
            Drawing(id=2, bbox=BBox(1, 1, 99, 99)),
        ]
        kept, removed = filter_background_blocks(blocks, page_width, page_height)
        assert len(kept) == 2
        assert set(kept) == set(blocks)
        assert removed == {}

    def test_background_block_removal(self) -> None:
        """Test that full-page blocks are removed."""
        page_width = 100.0
        page_height = 200.0
        blocks = [
            # Full page block (100% size)
            Drawing(id=1, bbox=BBox(0, 0, 100, 200)),
            # Small content block
            Text(id=2, bbox=BBox(10, 10, 20, 20), text="A"),
        ]
        kept, removed = filter_background_blocks(blocks, page_width, page_height)

        assert len(kept) == 1
        assert kept[0].id == 2  # Only text block kept
        assert len(removed) == 1
        removed_block = next(iter(removed.keys()))
        assert removed_block.id == 1
        reason = removed[removed_block]
        assert isinstance(reason, RemovalReason)
        assert reason.reason_type == "background_block"
        assert reason.target_block is None

    def test_nearly_full_page_removal(self) -> None:
        """Test that blocks slightly smaller than full page (>=99%) are removed."""
        page_width = 1000.0
        page_height = 1000.0

        # 990x990 is exactly 99% - should be removed
        # 989x989 is 98.9% - should be kept

        blocks = [
            # Exactly 99%
            Drawing(id=1, bbox=BBox(0, 0, 990, 990)),
            # Slightly larger (99.5%)
            Drawing(id=2, bbox=BBox(0, 0, 995, 995)),
            # Slightly smaller (98.9%)
            Drawing(id=3, bbox=BBox(0, 0, 989, 989)),
        ]

        kept, removed = filter_background_blocks(blocks, page_width, page_height)

        assert len(kept) == 1
        assert kept[0].id == 3

        assert len(removed) == 2
        removed_ids = {b.id for b in removed}
        assert 1 in removed_ids
        assert 2 in removed_ids

        for block, reason in removed.items():
            assert isinstance(reason, RemovalReason)
            assert reason.reason_type == "background_block"
            assert reason.target_block is None


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
            # Adjusted to 31,31 (21x21=441) to ensure IOU > 0.9 with block 1 (400/441 = 0.907)
            Drawing(id=3, bbox=BBox(10, 10, 31, 31)),  # area = 441 (largest)
        ]
        kept, removed = filter_duplicate_blocks(blocks)
        # Should filter to reduce duplicates
        assert len(kept) < len(blocks)
        # The largest block should be kept
        assert any(b.id == 3 for b in kept)
        # At least one block should be removed
        assert len(removed) >= 1

        for block, reason in removed.items():
            assert isinstance(reason, RemovalReason)
            assert reason.reason_type == "duplicate_bbox"
            assert reason.target_block is not None
            assert reason.target_block.id == 3

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

        kept_block = kept[0]
        for block, reason in removed.items():
            assert reason.reason_type == "duplicate_bbox"
            assert reason.target_block == kept_block

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


class TestFilterOverlappingTextBlocks:
    """Tests for the filter_overlapping_text_blocks function."""

    def test_empty_list(self) -> None:
        """Test with empty list returns empty list."""
        kept, removed = filter_overlapping_text_blocks([])
        assert kept == []
        assert removed == {}

    def test_no_text_blocks(self) -> None:
        """Test with non-text blocks - all should pass through unchanged."""
        blocks = [
            Drawing(id=1, bbox=BBox(10, 10, 20, 20)),
            Image(id=2, bbox=BBox(50, 50, 60, 60)),
        ]
        kept, removed = filter_overlapping_text_blocks(blocks)
        assert kept == blocks
        assert removed == {}

    def test_no_overlapping_text(self) -> None:
        """Test with text blocks at different positions - all should be kept."""
        blocks = [
            Text(id=1, bbox=BBox(10, 10, 20, 20), text="A"),
            Text(id=2, bbox=BBox(50, 50, 60, 60), text="B"),
            Text(id=3, bbox=BBox(100, 100, 110, 110), text="C"),
        ]
        kept, removed = filter_overlapping_text_blocks(blocks)
        assert len(kept) == 3
        assert set(kept) == set(blocks)
        assert removed == {}

    def test_overlapping_text_keeps_longest(self) -> None:
        """Test overlapping text at same origin - keeps longest text.

        This is the main use case: PDF has "4" and "43" at same origin
        as separate text blocks. We keep "43" (longer text).
        """
        blocks = [
            Text(id=1, bbox=BBox(100, 200, 110, 220), text="4"),
            Text(id=2, bbox=BBox(100, 200, 125, 220), text="43"),
        ]
        kept, removed = filter_overlapping_text_blocks(blocks)
        assert len(kept) == 1
        assert isinstance(kept[0], Text) and kept[0].text == "43"
        assert len(removed) == 1
        # Block with "4" should be removed
        removed_block = next(iter(removed.keys()))
        assert isinstance(removed_block, Text) and removed_block.text == "4"

        reason = removed[removed_block]
        assert isinstance(reason, RemovalReason)
        assert reason.reason_type == "overlapping_text"
        assert reason.target_block == kept[0]

    def test_multiple_groups_of_overlapping_text(self) -> None:
        """Test multiple groups of overlapping text at different positions."""
        blocks = [
            # Group at origin (100, 200)
            Text(id=1, bbox=BBox(100, 200, 110, 220), text="1"),
            Text(id=2, bbox=BBox(100, 200, 125, 220), text="12"),
            # Group at origin (300, 400)
            Text(id=3, bbox=BBox(300, 400, 310, 420), text="A"),
            Text(id=4, bbox=BBox(300, 400, 325, 420), text="AB"),
            Text(id=5, bbox=BBox(300, 400, 340, 420), text="ABC"),
        ]
        kept, removed = filter_overlapping_text_blocks(blocks)
        assert len(kept) == 2
        # Should keep "12" from first group and "ABC" from second
        kept_texts = {b.text for b in kept if isinstance(b, Text)}
        assert kept_texts == {"12", "ABC"}
        assert len(removed) == 3

    def test_mixed_blocks_preserves_non_text(self) -> None:
        """Test that non-text blocks are preserved even at same bbox."""
        blocks = [
            Text(id=1, bbox=BBox(100, 200, 110, 220), text="4"),
            Text(id=2, bbox=BBox(100, 200, 125, 220), text="43"),
            Drawing(id=3, bbox=BBox(100, 200, 130, 220)),  # Same origin
            Image(id=4, bbox=BBox(100, 200, 140, 220)),  # Same origin
        ]
        kept, removed = filter_overlapping_text_blocks(blocks)
        # Text deduplication: keeps "43"
        # Non-text blocks: preserved
        assert len(kept) == 3
        assert any(b.id == 2 for b in kept)  # Text "43"
        assert any(b.id == 3 for b in kept)  # Drawing
        assert any(b.id == 4 for b in kept)  # Image
        assert len(removed) == 1

    def test_tolerance_groups_nearby_origins(self) -> None:
        """Test that text with slightly different origins are grouped.

        The tolerance is 0.5, and values are rounded to nearest 0.5.
        So (100.2, 200.2, 220.2) rounds to (100.0, 200.0, 220.0).
        """
        blocks = [
            Text(id=1, bbox=BBox(100.0, 200.0, 110.0, 220.0), text="4"),
            # Slightly offset (within rounding to same 0.5 bucket)
            Text(id=2, bbox=BBox(100.2, 200.2, 125.0, 220.2), text="43"),
        ]
        kept, removed = filter_overlapping_text_blocks(blocks)
        assert len(kept) == 1
        assert isinstance(kept[0], Text) and kept[0].text == "43"
        assert len(removed) == 1

    def test_outside_tolerance_not_grouped(self) -> None:
        """Test that text with origins beyond tolerance are not grouped."""
        blocks = [
            Text(id=1, bbox=BBox(100.0, 200.0, 110.0, 220.0), text="4"),
            # Offset by more than 0.5 tolerance
            Text(id=2, bbox=BBox(101.0, 200.0, 126.0, 220.0), text="43"),
        ]
        kept, removed = filter_overlapping_text_blocks(blocks)
        # Both should be kept as they're at different origins
        assert len(kept) == 2
        assert removed == {}

    def test_same_length_text_keeps_widest(self) -> None:
        """Test same length text at same origin - keeps the one with widest bbox."""
        blocks = [
            Text(id=1, bbox=BBox(100, 200, 110, 220), text="AB"),  # width=10
            Text(id=2, bbox=BBox(100, 200, 125, 220), text="CD"),  # width=25
        ]
        kept, removed = filter_overlapping_text_blocks(blocks)
        assert len(kept) == 1
        # Should keep the one with the widest bbox (id=2, width=25)
        assert kept[0].id == 2
        assert isinstance(kept[0], Text) and kept[0].text == "CD"
