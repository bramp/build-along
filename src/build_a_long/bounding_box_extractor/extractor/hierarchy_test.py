from build_a_long.bounding_box_extractor.extractor.bbox import BBox
from build_a_long.bounding_box_extractor.extractor.hierarchy import (
    build_hierarchy_from_elements,
)
from build_a_long.bounding_box_extractor.extractor.page_elements import (
    Drawing,
    StepNumber,
    Unknown,
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
    assert isinstance(root.element, Drawing)
    assert len(root.children) == 1
    assert isinstance(root.children[0].element, StepNumber)


def test_build_hierarchy_unknown_and_children_mirroring():
    # Unknown parent (parts_list), with inner unknown child fully contained
    elements = [
        Unknown(
            bbox=BBox(0, 0, 50, 50),
            label="parts_list",
            raw_type="parts_list",
            source_id="text_0",
        ),
        Unknown(
            bbox=BBox(5, 5, 10, 10), raw_type="text", content="x3", source_id="text_1"
        ),
    ]
    roots = build_hierarchy_from_elements(elements)
    assert len(roots) == 1
    node = roots[0]
    assert isinstance(node.element, Unknown)
    # The Unknown element mirrors nested children as elements as well
    assert len(node.children) == 1
    assert isinstance(node.children[0].element, Unknown)
    assert isinstance(node.element, Unknown)
    assert len(node.element.children) == 1
    # Ensure element-level mirroring exists and is Unknown for this test
    assert isinstance(node.element.children[0], Unknown)
