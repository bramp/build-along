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
