from build_a_long.bounding_box_extractor.extractor.bbox import BBox
from build_a_long.bounding_box_extractor.extractor.hierarchy import (
    build_hierarchy_from_elements,
)
from build_a_long.bounding_box_extractor.extractor.page_elements import (
    Drawing,
    StepNumber,
    Text,
)


def test_build_hierarchy_basic_containment():
    # One big image with a small number inside it
    elements = [
        Drawing(bbox=BBox(0, 0, 100, 100), image_id="image_0"),
        StepNumber(bbox=BBox(10, 10, 20, 20), value=1),
    ]
    roots = build_hierarchy_from_elements(elements)
    assert len(roots) == 1
    root = roots[0]
    assert isinstance(root, Drawing)
    assert len(root.children) == 1
    assert isinstance(root.children[0], StepNumber)


def test_build_hierarchy_with_children():
    # Drawing parent with nested text child fully contained
    elements = [
        Drawing(bbox=BBox(0, 0, 50, 50), image_id="drawing_0"),
        Text(bbox=BBox(5, 5, 10, 10), content="x3"),
    ]
    roots = build_hierarchy_from_elements(elements)
    assert len(roots) == 1
    root = roots[0]
    assert isinstance(root, Drawing)
    # The Drawing element should have the text as a child
    assert len(root.children) == 1
    assert isinstance(root.children[0], Text)
    assert root.children[0].content == "x3"
