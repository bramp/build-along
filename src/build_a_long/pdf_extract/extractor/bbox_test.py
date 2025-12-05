from dataclasses import dataclass

import pytest

from build_a_long.pdf_extract.extractor.bbox import (
    BBox,
    build_all_connected_clusters,
    build_connected_cluster,
)


def test_overlaps():
    bbox1 = BBox(0, 0, 10, 10)
    bbox2 = BBox(5, 5, 15, 15)
    bbox3 = BBox(10, 10, 20, 20)  # Touches at corner
    bbox4 = BBox(11, 11, 20, 20)  # No overlap
    bbox5 = BBox(0, 0, 10, 5)  # Partial overlap

    assert bbox1.overlaps(bbox2)
    assert bbox2.overlaps(bbox1)
    assert not bbox1.overlaps(bbox3)  # Touching at corner is not overlapping
    assert not bbox3.overlaps(bbox1)
    assert not bbox1.overlaps(bbox4)
    assert bbox1.overlaps(bbox5)


def test_contains():
    bbox1 = BBox(0, 0, 10, 10)
    bbox2 = BBox(2, 2, 8, 8)
    bbox3 = BBox(0, 0, 10, 10)  # Same bbox
    bbox4 = BBox(0, 0, 10, 11)  # Not fully inside

    assert bbox1.contains(bbox2)
    assert bbox1.contains(bbox3)
    assert not bbox2.contains(bbox1)
    assert not bbox1.contains(bbox4)


def test_adjacent():
    bbox1 = BBox(0, 0, 10, 10)
    bbox2 = BBox(10, 0, 20, 10)  # Right adjacent
    bbox3 = BBox(0, 10, 10, 20)  # Top adjacent
    bbox4 = BBox(10, 10, 20, 20)  # Corner adjacent
    bbox5 = BBox(11, 0, 20, 10)  # Not adjacent
    bbox6 = BBox(5, 5, 15, 15)  # Overlapping, not adjacent

    assert bbox1.adjacent(bbox2)
    assert bbox2.adjacent(bbox1)
    assert bbox1.adjacent(bbox3)
    assert bbox3.adjacent(bbox1)
    # Corner adjacency is not considered adjacent
    assert not bbox1.adjacent(bbox4)
    assert not bbox1.adjacent(bbox5)
    assert not bbox1.adjacent(bbox6)


def test_similar():
    bbox1 = BBox(0, 0, 10, 10)
    bbox2 = BBox(0.5, 0.5, 10.5, 10.5)  # Within default tolerance of 1.0
    bbox3 = BBox(2, 2, 12, 12)  # Outside default tolerance
    bbox4 = BBox(0, 0, 10, 10)  # Exact same bbox

    # Default tolerance (1.0)
    assert bbox1.similar(bbox4)
    assert bbox1.similar(bbox2)
    assert not bbox1.similar(bbox3)

    # Custom tolerance
    assert bbox1.similar(bbox3, tolerance=2.0)
    assert not bbox1.similar(bbox2, tolerance=0.1)

    # Similar is symmetric
    assert bbox1.similar(bbox2) == bbox2.similar(bbox1)


def test_union():
    bbox1 = BBox(0, 0, 10, 10)
    bbox2 = BBox(5, 5, 15, 15)
    bbox3 = BBox(20, 20, 30, 30)

    # Test overlapping boxes
    union_12 = bbox1.union(bbox2)
    assert union_12.x0 == 0
    assert union_12.y0 == 0
    assert union_12.x1 == 15
    assert union_12.y1 == 15

    # Test non-overlapping boxes
    union_13 = bbox1.union(bbox3)
    assert union_13.x0 == 0
    assert union_13.y0 == 0
    assert union_13.x1 == 30
    assert union_13.y1 == 30

    # Test union is commutative
    assert bbox1.union(bbox2) == bbox2.union(bbox1)


def test_union_all():
    bbox1 = BBox(0, 0, 10, 10)
    bbox2 = BBox(5, 5, 15, 15)
    bbox3 = BBox(20, 20, 30, 30)

    # Test with multiple boxes
    union = BBox.union_all([bbox1, bbox2, bbox3])
    assert union.x0 == 0
    assert union.y0 == 0
    assert union.x1 == 30
    assert union.y1 == 30

    # Test with single box
    union_single = BBox.union_all([bbox1])
    assert union_single == bbox1

    # Test with two boxes
    union_two = BBox.union_all([bbox1, bbox2])
    assert union_two == bbox1.union(bbox2)


def test_union_all_empty():
    # Test empty list raises ValueError
    with pytest.raises(ValueError, match="Cannot compute union of empty list"):
        BBox.union_all([])


def test_clip_to_already_contained():
    """Test clipping when bbox is already fully inside bounds."""
    bbox = BBox(2, 3, 8, 7)
    bounds = BBox(0, 0, 10, 10)

    clipped = bbox.clip_to(bounds)
    assert clipped == bbox  # Should be unchanged


def test_clip_to_extends_right():
    """Test clipping when bbox extends beyond right edge."""
    bbox = BBox(5, 2, 15, 8)
    bounds = BBox(0, 0, 10, 10)

    clipped = bbox.clip_to(bounds)
    assert clipped.x0 == 5
    assert clipped.y0 == 2
    assert clipped.x1 == 10  # Clipped to bounds
    assert clipped.y1 == 8


def test_clip_to_extends_left():
    """Test clipping when bbox extends beyond left edge."""
    bbox = BBox(-5, 2, 8, 7)
    bounds = BBox(0, 0, 10, 10)

    clipped = bbox.clip_to(bounds)
    assert clipped.x0 == 0  # Clipped to bounds
    assert clipped.y0 == 2
    assert clipped.x1 == 8
    assert clipped.y1 == 7


def test_clip_to_extends_top():
    """Test clipping when bbox extends beyond top edge."""
    bbox = BBox(2, -3, 8, 7)
    bounds = BBox(0, 0, 10, 10)

    clipped = bbox.clip_to(bounds)
    assert clipped.x0 == 2
    assert clipped.y0 == 0  # Clipped to bounds
    assert clipped.x1 == 8
    assert clipped.y1 == 7


def test_clip_to_extends_bottom():
    """Test clipping when bbox extends beyond bottom edge."""
    bbox = BBox(2, 3, 8, 15)
    bounds = BBox(0, 0, 10, 10)

    clipped = bbox.clip_to(bounds)
    assert clipped.x0 == 2
    assert clipped.y0 == 3
    assert clipped.x1 == 8
    assert clipped.y1 == 10  # Clipped to bounds


def test_clip_to_extends_all_sides():
    """Test clipping when bbox extends beyond all edges."""
    bbox = BBox(-5, -3, 15, 12)
    bounds = BBox(0, 0, 10, 10)

    clipped = bbox.clip_to(bounds)
    assert clipped.x0 == 0  # Clipped to left
    assert clipped.y0 == 0  # Clipped to top
    assert clipped.x1 == 10  # Clipped to right
    assert clipped.y1 == 10  # Clipped to bottom


def test_clip_to_completely_outside():
    """Test clipping when bbox is completely outside bounds."""
    bbox = BBox(15, 15, 20, 20)
    bounds = BBox(0, 0, 10, 10)

    clipped = bbox.clip_to(bounds)
    # When completely outside, should return a degenerate bbox at the nearest edge
    assert clipped.x0 == 10
    assert clipped.y0 == 10
    assert clipped.x1 == 10
    assert clipped.y1 == 10
    # Verify it's a valid (degenerate) bbox
    assert clipped.width == 0
    assert clipped.height == 0


def test_clip_to_progress_bar_case():
    """Test real-world case from ProgressBar with extends beyond page bounds."""
    # Progress bar that extends beyond page width
    progress_bar = BBox(35.02, 474.1, 1070.5, 482.59)
    page_bounds = BBox(0.0, 0.0, 552.76, 496.06)

    clipped = progress_bar.clip_to(page_bounds)
    assert clipped.x0 == 35.02
    assert clipped.y0 == 474.1
    assert clipped.x1 == 552.76  # Clipped to page width
    assert clipped.y1 == 482.59

    # Progress bar with negative coordinates
    progress_bar2 = BBox(-517.74, 474.1, 517.74, 482.59)
    clipped2 = progress_bar2.clip_to(page_bounds)
    assert clipped2.x0 == 0.0  # Clipped to left edge
    assert clipped2.y0 == 474.1
    assert clipped2.x1 == 517.74
    assert clipped2.y1 == 482.59


def test_clip_to_result_is_valid():
    """Test that clipped bbox results are always valid (x0<=x1, y0<=y1)."""
    bbox = BBox(5, 5, 15, 15)
    bounds = BBox(0, 0, 10, 10)

    clipped = bbox.clip_to(bounds)
    # Should not raise ValueError for invalid bbox
    assert clipped.x0 <= clipped.x1
    assert clipped.y0 <= clipped.y1


@dataclass
class MockItem:
    """Mock item with a bbox for testing clustering."""

    id: int
    bbox: BBox


def test_build_connected_cluster_empty():
    """Test clustering with empty seed items."""
    items = [MockItem(1, BBox(0, 0, 10, 10))]
    result = build_connected_cluster([], items)
    assert result == []


def test_build_connected_cluster_single_seed():
    """Test clustering with a single seed item."""
    item1 = MockItem(1, BBox(0, 0, 10, 10))
    item2 = MockItem(2, BBox(5, 5, 15, 15))  # Overlaps with item1
    item3 = MockItem(3, BBox(20, 20, 30, 30))  # No overlap

    items = [item1, item2, item3]
    result = build_connected_cluster([item1], items)

    assert len(result) == 2
    assert item1 in result
    assert item2 in result
    assert item3 not in result


def test_build_connected_cluster_chain():
    """Test clustering with a chain of overlapping items."""
    item1 = MockItem(1, BBox(0, 0, 10, 10))
    item2 = MockItem(2, BBox(5, 5, 15, 15))  # Overlaps with item1
    item3 = MockItem(3, BBox(11, 11, 21, 21))  # Overlaps with item2
    item4 = MockItem(4, BBox(13, 13, 23, 23))  # Overlaps with item3
    item5 = MockItem(5, BBox(50, 50, 60, 60))  # Isolated

    items = [item1, item2, item3, item4, item5]
    result = build_connected_cluster([item1], items)

    # All items except isolated item5 should be in cluster
    assert len(result) == 4
    assert item1 in result
    assert item2 in result
    assert item3 in result
    assert item4 in result
    assert item5 not in result


def test_build_connected_cluster_multiple_seeds():
    """Test clustering with multiple seed items."""
    item1 = MockItem(1, BBox(0, 0, 10, 10))
    item2 = MockItem(2, BBox(5, 5, 15, 15))  # Overlaps with item1
    item3 = MockItem(3, BBox(50, 50, 60, 60))  # Isolated
    item4 = MockItem(4, BBox(55, 55, 65, 65))  # Overlaps with item3

    items = [item1, item2, item3, item4]
    result = build_connected_cluster([item1, item3], items)

    # Should create two separate clusters
    assert len(result) == 4
    assert all(item in result for item in items)


def test_build_connected_cluster_all_overlapping():
    """Test clustering where all items overlap."""
    item1 = MockItem(1, BBox(0, 0, 20, 20))
    item2 = MockItem(2, BBox(5, 5, 15, 15))  # Inside item1
    item3 = MockItem(3, BBox(10, 10, 25, 25))  # Overlaps with item1 and item2
    item4 = MockItem(4, BBox(15, 15, 30, 30))  # Overlaps with item3

    items = [item1, item2, item3, item4]
    result = build_connected_cluster([item1], items)

    assert len(result) == 4
    assert all(item in result for item in items)


def test_build_connected_cluster_preserves_order():
    """Test that the result preserves the original order of items."""
    item1 = MockItem(1, BBox(0, 0, 10, 10))
    item2 = MockItem(2, BBox(5, 5, 15, 15))
    item3 = MockItem(3, BBox(20, 20, 30, 30))  # Doesn't overlap anything
    item4 = MockItem(4, BBox(7, 7, 12, 12))  # Overlaps item2

    items = [item1, item2, item3, item4]
    result = build_connected_cluster([item1], items)

    # Result should be in same order as original list
    assert result == [item1, item2, item4]


# Tests for build_all_connected_clusters


def test_build_all_connected_clusters_empty():
    """Test with empty list."""
    result = build_all_connected_clusters([])
    assert result == []


def test_build_all_connected_clusters_single_item():
    """Test with single item returns one cluster."""
    item = MockItem(1, BBox(0, 0, 10, 10))
    result = build_all_connected_clusters([item])
    assert len(result) == 1
    assert result[0] == [item]


def test_build_all_connected_clusters_separate():
    """Test that non-overlapping items form separate clusters."""
    item1 = MockItem(1, BBox(0, 0, 10, 10))
    item2 = MockItem(2, BBox(20, 20, 30, 30))  # No overlap
    item3 = MockItem(3, BBox(50, 50, 60, 60))  # No overlap

    result = build_all_connected_clusters([item1, item2, item3])
    assert len(result) == 3
    # Each item in its own cluster
    assert [item1] in result
    assert [item2] in result
    assert [item3] in result


def test_build_all_connected_clusters_chain():
    """Test that overlapping items form one cluster."""
    item1 = MockItem(1, BBox(0, 0, 10, 10))
    item2 = MockItem(2, BBox(5, 5, 15, 15))  # Overlaps item1
    item3 = MockItem(3, BBox(10, 10, 20, 20))  # Overlaps item2
    item4 = MockItem(4, BBox(50, 50, 60, 60))  # Separate

    result = build_all_connected_clusters([item1, item2, item3, item4])
    assert len(result) == 2
    # item1, item2, item3 should form one cluster
    cluster1 = next(c for c in result if len(c) == 3)
    assert sorted(i.id for i in cluster1) == [1, 2, 3]
    # item4 should be alone
    cluster2 = next(c for c in result if len(c) == 1)
    assert cluster2[0].id == 4


def test_build_all_connected_clusters_multiple_groups():
    """Test with multiple distinct groups."""
    # Group 1
    item1 = MockItem(1, BBox(0, 0, 10, 10))
    item2 = MockItem(2, BBox(5, 5, 15, 15))  # Overlaps item1
    # Group 2
    item3 = MockItem(3, BBox(100, 100, 110, 110))
    item4 = MockItem(4, BBox(105, 105, 115, 115))  # Overlaps item3
    # Isolated
    item5 = MockItem(5, BBox(200, 200, 210, 210))

    result = build_all_connected_clusters([item1, item2, item3, item4, item5])
    assert len(result) == 3
    # Check group sizes
    sizes = sorted(len(c) for c in result)
    assert sizes == [1, 2, 2]
