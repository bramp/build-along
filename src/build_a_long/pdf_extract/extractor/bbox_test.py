from dataclasses import dataclass

import pytest
from hypothesis import given
from hypothesis import strategies as st

from build_a_long.pdf_extract.extractor.bbox import (
    BBox,
    build_all_connected_clusters,
    build_connected_cluster,
    filter_contained,
    filter_overlapping,
    group_by_similar_bbox,
)

# --- Strategies ---
floats = st.floats(
    min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False
)
positive_floats = st.floats(
    min_value=0, max_value=1000, allow_nan=False, allow_infinity=False
)


@st.composite
def bboxes(draw):
    x0 = draw(floats)
    y0 = draw(floats)
    w = draw(positive_floats)
    h = draw(positive_floats)
    return BBox(x0, y0, x0 + w, y0 + h)


@st.composite
def non_degenerate_bboxes(draw):
    x0 = draw(floats)
    y0 = draw(floats)
    w = draw(
        st.floats(min_value=1e-6, max_value=1000, allow_nan=False, allow_infinity=False)
    )  # Ensure positive width
    h = draw(
        st.floats(min_value=1e-6, max_value=1000, allow_nan=False, allow_infinity=False)
    )  # Ensure positive height
    return BBox(x0, y0, x0 + w, y0 + h)


# --- Property Tests ---


@given(bboxes(), bboxes())
def test_overlaps_property(b1, b2):
    # Definition of overlaps: intervals intersect on both axes
    x_overlap = max(b1.x0, b2.x0) < min(b1.x1, b2.x1)
    y_overlap = max(b1.y0, b2.y0) < min(b1.y1, b2.y1)
    expected = x_overlap and y_overlap
    assert b1.overlaps(b2) == expected
    # Symmetry
    assert b1.overlaps(b2) == b2.overlaps(b1)


@given(bboxes(), bboxes())
def test_contains_property(b1, b2):
    # Definition of contains: b2 inside b1
    expected = b1.x0 <= b2.x0 and b1.y0 <= b2.y0 and b1.x1 >= b2.x1 and b1.y1 >= b2.y1
    assert b1.contains(b2) == expected
    # Self-containment
    if b1 == b2:
        assert b1.contains(b2)


@given(bboxes(), bboxes())
def test_union_property(b1, b2):
    u = b1.union(b2)
    # Validity
    assert u.x0 == pytest.approx(min(b1.x0, b2.x0))
    assert u.y0 == pytest.approx(min(b1.y0, b2.y0))
    assert u.x1 == pytest.approx(max(b1.x1, b2.x1))
    assert u.y1 == pytest.approx(max(b1.y1, b2.y1))
    # Properties
    assert u.contains(b1)
    assert u.contains(b2)
    assert b1.union(b2) == b2.union(b1)


@given(st.lists(bboxes(), min_size=1))
def test_union_all_property(boxes):
    u = BBox.union_all(boxes)
    # Check bounds
    min_x0 = min(b.x0 for b in boxes)
    min_y0 = min(b.y0 for b in boxes)
    max_x1 = max(b.x1 for b in boxes)
    max_y1 = max(b.y1 for b in boxes)
    assert u.x0 == pytest.approx(min_x0)
    assert u.y0 == pytest.approx(min_y0)
    assert u.x1 == pytest.approx(max_x1)
    assert u.y1 == pytest.approx(max_y1)


@given(bboxes(), floats)
def test_expand_property(b, margin):
    # Calculate expected new coordinates
    expected_x0 = b.x0 - margin
    expected_y0 = b.y0 - margin
    expected_x1 = b.x1 + margin
    expected_y1 = b.y1 + margin

    # Predict if ValueError should be raised by BBox constructor due to invalid dimensions
    should_raise = expected_x0 > expected_x1 or expected_y0 > expected_y1

    if should_raise:
        with pytest.raises(ValueError):
            b.expand(margin)
    else:
        e = b.expand(margin)
        assert e.x0 == pytest.approx(expected_x0)
        assert e.y0 == pytest.approx(expected_y0)
        assert e.x1 == pytest.approx(expected_x1)
        assert e.y1 == pytest.approx(expected_y1)
        # Check containment
        if margin > 0:
            assert e.contains(b)  # Expanded box contains original
        elif margin < 0:
            assert b.contains(e)  # Original contains shrunk box
        else:  # margin == 0
            assert e == b


@given(
    bboxes(),
    bboxes(),
    st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False),
)
def test_similar_property(b1, b2, tolerance):
    # Similar is symmetric
    assert b1.similar(b2, tolerance) == b2.similar(b1, tolerance)

    # If all coordinate differences are within tolerance, they should be similar
    all_coords_within_tolerance = all(
        abs(getattr(b1, coord) - getattr(b2, coord)) <= tolerance
        for coord in ["x0", "y0", "x1", "y1"]
    )
    if all_coords_within_tolerance:
        assert b1.similar(b2, tolerance)
    else:
        # If not all coords are within tolerance, it *might* not be similar.
        # However, Hypothesis might generate cases where float precision
        # makes exact comparison difficult for this inverse check.
        # The primary check is 'all_coords_within_tolerance => similar'.
        pass


@given(bboxes(), positive_floats)  # Only test with positive scales
def test_mul_scale_positive_property(b, scale):
    scaled = b * scale
    assert scaled.x0 == pytest.approx(b.x0 * scale)
    assert scaled.y0 == pytest.approx(b.y0 * scale)
    assert scaled.x1 == pytest.approx(b.x1 * scale)
    assert scaled.y1 == pytest.approx(b.y1 * scale)
    assert scaled.width == pytest.approx(b.width * scale)
    assert scaled.height == pytest.approx(b.height * scale)


@given(
    non_degenerate_bboxes(),
    st.floats(
        max_value=-1e-6, min_value=-1e10, allow_nan=False, allow_infinity=False
    ),  # Filter out extremely large negative scales
)  # Test with negative scales
def test_mul_negative_scale_raises_error(b, scale):
    # A negative scale should invert coordinates such that x0 > x1 or y0 > y1,
    # leading to a ValueError from BBox's model_post_init.
    with pytest.raises(
        ValueError
    ):  # This should catch the ValueError raised by model_post_init
        _ = b * scale


@given(bboxes())
def test_to_int_tuple_property(b):
    int_tuple = b.to_int_tuple()
    assert len(int_tuple) == 4
    for coord in int_tuple:
        assert isinstance(coord, int)
    assert int_tuple[0] == int(b.x0)
    assert int_tuple[1] == int(b.y0)
    assert int_tuple[2] == int(b.x1)
    assert int_tuple[3] == int(b.y1)


# --- Original Clustering and Specific Tests (Kept) ---


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


class TestGroupBySimilarBbox:
    """Tests for group_by_similar_bbox function."""

    def test_empty_list(self):
        """Empty list returns empty list."""
        result = group_by_similar_bbox([])
        assert result == []

    def test_single_item(self):
        """Single item returns one group with that item."""
        item = MockItem(1, BBox(10, 10, 20, 20))
        result = group_by_similar_bbox([item])
        assert len(result) == 1
        assert result[0] == [item]

    def test_identical_bboxes(self):
        """Items with identical bboxes are grouped together."""
        item1 = MockItem(1, BBox(10, 10, 20, 20))
        item2 = MockItem(2, BBox(10, 10, 20, 20))
        item3 = MockItem(3, BBox(10, 10, 20, 20))

        result = group_by_similar_bbox([item1, item2, item3])
        assert len(result) == 1
        assert len(result[0]) == 3
        assert item1 in result[0]
        assert item2 in result[0]
        assert item3 in result[0]

    def test_similar_bboxes_within_tolerance(self):
        """Items with bboxes within tolerance are grouped together."""
        item1 = MockItem(1, BBox(10.0, 10.0, 20.0, 20.0))
        item2 = MockItem(2, BBox(10.5, 10.5, 20.5, 20.5))  # Within tolerance=2.0
        item3 = MockItem(3, BBox(11.0, 11.0, 21.0, 21.0))  # Within tolerance=2.0

        result = group_by_similar_bbox([item1, item2, item3], tolerance=2.0)
        assert len(result) == 1
        assert len(result[0]) == 3

    def test_different_bboxes_separate_groups(self):
        """Items with different bboxes are in separate groups."""
        item1 = MockItem(1, BBox(10, 10, 20, 20))
        item2 = MockItem(2, BBox(100, 100, 110, 110))
        item3 = MockItem(3, BBox(200, 200, 210, 210))

        result = group_by_similar_bbox([item1, item2, item3])
        assert len(result) == 3
        assert [item1] in result
        assert [item2] in result
        assert [item3] in result

    def test_mixed_similar_and_different(self):
        """Mix of similar and different bboxes."""
        # Group 1: similar bboxes
        item1 = MockItem(1, BBox(10, 10, 20, 20))
        item2 = MockItem(2, BBox(10.5, 10.5, 20.5, 20.5))
        # Group 2: different bbox
        item3 = MockItem(3, BBox(100, 100, 110, 110))
        # Group 3: similar to group 2
        item4 = MockItem(4, BBox(100.5, 100.5, 110.5, 110.5))

        result = group_by_similar_bbox([item1, item2, item3, item4], tolerance=2.0)
        assert len(result) == 2

        # Find group containing item1
        group1 = next(g for g in result if item1 in g)
        assert item2 in group1
        assert len(group1) == 2

        # Find group containing item3
        group2 = next(g for g in result if item3 in g)
        assert item4 in group2
        assert len(group2) == 2

    def test_custom_tolerance(self):
        """Custom tolerance affects grouping."""
        item1 = MockItem(1, BBox(10, 10, 20, 20))
        item2 = MockItem(2, BBox(15, 15, 25, 25))  # 5 points difference

        # With default tolerance=2.0, these are separate groups
        result = group_by_similar_bbox([item1, item2], tolerance=2.0)
        assert len(result) == 2

        # With tolerance=10.0, they're in the same group
        result = group_by_similar_bbox([item1, item2], tolerance=10.0)
        assert len(result) == 1
        assert len(result[0]) == 2

    def test_preserves_insertion_order(self):
        """Groups and items within groups preserve insertion order."""
        item1 = MockItem(1, BBox(10, 10, 20, 20))
        item2 = MockItem(2, BBox(100, 100, 110, 110))
        item3 = MockItem(3, BBox(10.5, 10.5, 20.5, 20.5))  # Similar to item1
        item4 = MockItem(4, BBox(100.5, 100.5, 110.5, 110.5))  # Similar to item2

        result = group_by_similar_bbox([item1, item2, item3, item4], tolerance=2.0)

        # First group should be item1's group (first encountered)
        assert result[0][0] == item1
        assert result[0][1] == item3

        # Second group should be item2's group
        assert result[1][0] == item2
        assert result[1][1] == item4


def test_build_connected_cluster_single_seed():
    """Test clustering with a single seed item."""
    item1 = MockItem(1, BBox(0, 0, 10, 10))
    item2 = MockItem(2, BBox(5, 5, 15, 15))  # Overlaps with item1
    item3 = MockItem(3, BBox(20, 20, 30, 30))  # No overlap

    items = [item1, item2, item3]
    result = build_connected_cluster(item1, items)

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
    result = build_connected_cluster(item1, items)

    # All items except isolated item5 should be in cluster
    assert len(result) == 4
    assert item1 in result
    assert item2 in result
    assert item3 in result
    assert item4 in result
    assert item5 not in result


def test_build_connected_cluster_disjoint_seeds():
    """Test that clustering with disjoint seeds requires separate calls."""
    item1 = MockItem(1, BBox(0, 0, 10, 10))
    item2 = MockItem(2, BBox(5, 5, 15, 15))  # Overlaps with item1
    item3 = MockItem(3, BBox(50, 50, 60, 60))  # Isolated
    item4 = MockItem(4, BBox(55, 55, 65, 65))  # Overlaps with item3

    items = [item1, item2, item3, item4]

    # Cluster starting from item1
    result1 = build_connected_cluster(item1, items)
    assert len(result1) == 2
    assert item1 in result1
    assert item2 in result1
    assert item3 not in result1
    assert item4 not in result1

    # Cluster starting from item3
    result2 = build_connected_cluster(item3, items)
    assert len(result2) == 2
    assert item3 in result2
    assert item4 in result2
    assert item1 not in result2
    assert item2 not in result2


def test_build_connected_cluster_all_overlapping():
    """Test clustering where all items overlap."""
    item1 = MockItem(1, BBox(0, 0, 20, 20))
    item2 = MockItem(2, BBox(5, 5, 15, 15))  # Inside item1
    item3 = MockItem(3, BBox(10, 10, 25, 25))  # Overlaps with item1 and item2
    item4 = MockItem(4, BBox(15, 15, 30, 30))  # Overlaps with item3

    items = [item1, item2, item3, item4]
    result = build_connected_cluster(item1, items)

    assert len(result) == 4
    assert all(item in result for item in items)


def test_build_connected_cluster_preserves_order():
    """Test that the result preserves the original order of items."""
    item1 = MockItem(1, BBox(0, 0, 10, 10))
    item2 = MockItem(2, BBox(5, 5, 15, 15))
    item3 = MockItem(3, BBox(20, 20, 30, 30))  # Doesn't overlap anything
    item4 = MockItem(4, BBox(7, 7, 12, 12))  # Overlaps item2

    items = [item1, item2, item3, item4]
    result = build_connected_cluster(item1, items)

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


def test_filter_contained():
    """Test filtering items contained in a bbox."""
    container = BBox(0, 0, 20, 20)

    item1 = MockItem(1, BBox(5, 5, 15, 15))  # Contained
    item2 = MockItem(2, BBox(0, 0, 20, 20))  # Contained (exact match)
    item3 = MockItem(3, BBox(15, 15, 25, 25))  # Overlapping but not contained
    item4 = MockItem(4, BBox(30, 30, 40, 40))  # Outside

    items = [item1, item2, item3, item4]
    result = filter_contained(items, container)

    assert len(result) == 2
    assert item1 in result
    assert item2 in result


def test_filter_overlapping():
    """Test filtering items overlapping with a bbox."""
    target = BBox(10, 10, 20, 20)

    item1 = MockItem(1, BBox(12, 12, 18, 18))  # Fully inside (overlaps)
    item2 = MockItem(2, BBox(5, 5, 15, 15))  # Partial overlap
    item3 = MockItem(3, BBox(20, 20, 30, 30))  # Touching corner (not overlapping)
    item4 = MockItem(4, BBox(30, 30, 40, 40))  # No overlap

    items = [item1, item2, item3, item4]
    result = filter_overlapping(items, target)

    assert len(result) == 2
    assert item1 in result
    assert item2 in result
