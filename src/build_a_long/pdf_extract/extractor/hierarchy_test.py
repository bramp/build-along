from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.hierarchy import (
    build_hierarchy_from_elements,
)
from build_a_long.pdf_extract.extractor.page_elements import (
    Drawing,
    Text,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    StepNumber,
)


def test_build_hierarchy_basic_containment():
    # One big image with a small number inside it
    elements = [
        Drawing(bbox=BBox(0, 0, 100, 100), image_id="image_0"),
        StepNumber(bbox=BBox(10, 10, 20, 20), value=1),
    ]
    tree = build_hierarchy_from_elements(elements)
    assert len(tree.roots) == 1
    root = tree.roots[0]
    assert isinstance(root, Drawing)
    children = tree.get_children(root)
    assert len(children) == 1
    assert isinstance(children[0], StepNumber)
    # Test depth calculation
    assert tree.get_depth(root) == 0
    assert tree.get_depth(children[0]) == 1


def test_build_hierarchy_with_children():
    # Drawing parent with nested text child fully contained
    elements = [
        Drawing(bbox=BBox(0, 0, 50, 50), image_id="drawing_0"),
        Text(bbox=BBox(5, 5, 10, 10), text="x3"),
    ]
    tree = build_hierarchy_from_elements(elements)
    assert len(tree.roots) == 1
    root = tree.roots[0]
    assert isinstance(root, Drawing)
    # The Drawing element should have the text as a child
    children = tree.get_children(root)
    assert len(children) == 1
    assert isinstance(children[0], Text)
    assert children[0].text == "x3"
    # Test depth calculation
    assert tree.get_depth(root) == 0
    assert tree.get_depth(children[0]) == 1


def test_build_hierarchy_depth_calculation():
    # Test multi-level nesting: outer -> middle -> inner
    outer = Drawing(bbox=BBox(0, 0, 100, 100), image_id="outer")
    middle = Drawing(bbox=BBox(10, 10, 90, 90), image_id="middle")
    inner = Text(bbox=BBox(20, 20, 30, 30), text="inner")
    sibling = Text(bbox=BBox(50, 50, 60, 60), text="sibling")

    elements = [outer, middle, inner, sibling]
    tree = build_hierarchy_from_elements(elements)

    # Verify depths
    assert tree.get_depth(outer) == 0
    assert tree.get_depth(middle) == 1
    assert tree.get_depth(inner) == 2
    assert tree.get_depth(sibling) == 2  # sibling to inner, both inside middle
