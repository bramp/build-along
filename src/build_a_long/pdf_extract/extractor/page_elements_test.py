from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Part,
    PartCount,
    PartsList,
    StepNumber,
)
from build_a_long.pdf_extract.extractor.page_elements import (
    Drawing,
)


def test_step_number():
    sn = StepNumber(bbox=BBox(0, 0, 10, 10), value=3)
    assert sn.value == 3
    assert isinstance(sn.bbox, BBox)


def test_drawing_optional_id():
    d = Drawing(bbox=BBox(1, 1, 100, 100))
    assert d.image_id is None
    d2 = Drawing(bbox=BBox(1, 1, 100, 100), image_id="img_1")
    assert d2.image_id == "img_1"


def test_part_and_count():
    cnt = PartCount(bbox=BBox(5, 5, 7, 7), count=2)
    p = Part(bbox=BBox(0, 0, 10, 10), name=None, number=None, count=cnt)
    assert p.count.count == 2
    assert p.bbox.x1 == 10


def test_parts_list_total_items():
    p1 = Part(
        bbox=BBox(0, 0, 10, 10),
        name="Brick",
        number="3001",
        count=PartCount(bbox=BBox(8, 0, 10, 2), count=2),
    )
    p2 = Part(
        bbox=BBox(0, 10, 10, 20),
        name="Plate",
        number="3020",
        count=PartCount(bbox=BBox(8, 10, 10, 12), count=5),
    )
    pl = PartsList(bbox=BBox(0, 0, 100, 200), parts=[p1, p2])
    assert pl.total_items == 7


def test_partcount_non_negative():
    PartCount(bbox=BBox(0, 0, 1, 1), count=0)  # ok
    try:
        PartCount(bbox=BBox(0, 0, 1, 1), count=-1)
        assert False, "Expected ValueError"
    except ValueError:
        pass
