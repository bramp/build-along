from build_a_long.pdf_extract.extractor.bbox import BBox


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


def test_fully_inside():
    bbox1 = BBox(0, 0, 10, 10)
    bbox2 = BBox(2, 2, 8, 8)
    bbox3 = BBox(0, 0, 10, 10)  # Same bbox
    bbox4 = BBox(0, 0, 10, 11)  # Not fully inside

    assert bbox2.fully_inside(bbox1)
    assert bbox3.fully_inside(bbox1)
    assert not bbox1.fully_inside(bbox2)
    assert not bbox4.fully_inside(bbox1)


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
    import pytest

    # Test empty list raises ValueError
    with pytest.raises(ValueError, match="Cannot compute union of empty list"):
        BBox.union_all([])


def test_clip_to_fully_inside():
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
