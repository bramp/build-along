import pytest

from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Part,
    PartCount,
    PartsList,
    StepNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Drawing,
    Image,
    Text,
)


def test_step_number():
    sn = StepNumber(bbox=BBox(0, 0, 10, 10), value=3)
    assert sn.value == 3
    assert isinstance(sn.bbox, BBox)


def test_drawing_optional_id():
    d = Drawing(bbox=BBox(1, 1, 100, 100), id=0)
    assert d.bbox == BBox(1, 1, 100, 100)
    assert not d.is_clipped


def test_part_and_count():
    cnt = PartCount(bbox=BBox(5, 5, 7, 7), count=2)
    p = Part(bbox=BBox(0, 0, 10, 10), count=cnt)
    assert p.count.count == 2
    assert p.bbox.x1 == 10


def test_parts_list_total_items():
    p1 = Part(
        bbox=BBox(0, 0, 10, 10),
        count=PartCount(bbox=BBox(8, 0, 10, 2), count=2),
    )
    p2 = Part(
        bbox=BBox(0, 10, 10, 20),
        count=PartCount(bbox=BBox(8, 10, 10, 12), count=5),
    )
    pl = PartsList(bbox=BBox(0, 0, 100, 200), parts=[p1, p2])
    assert pl.total_items == 7


def test_partcount_positive():
    """Test that PartCount requires count > 0."""
    PartCount(bbox=BBox(0, 0, 1, 1), count=1)  # ok

    with pytest.raises(ValueError):
        PartCount(bbox=BBox(0, 0, 1, 1), count=0)

    with pytest.raises(ValueError):
        PartCount(bbox=BBox(0, 0, 1, 1), count=-1)



def test_drawing_is_clipped_property():
    """Test the is_clipped property on Drawing instances."""
    # Unclipped drawing - bbox same as original_bbox
    unclipped = Drawing(bbox=BBox(0, 0, 10, 10), original_bbox=BBox(0, 0, 10, 10), id=1)
    assert not unclipped.is_clipped

    # Clipped drawing - bbox differs from original_bbox
    clipped = Drawing(bbox=BBox(2, 2, 8, 8), original_bbox=BBox(0, 0, 10, 10), id=2)
    assert clipped.is_clipped

    # Verify the bboxes are indeed different
    assert clipped.bbox != clipped.original_bbox
    assert unclipped.bbox == unclipped.original_bbox
