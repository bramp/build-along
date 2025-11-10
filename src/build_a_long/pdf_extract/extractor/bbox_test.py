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
