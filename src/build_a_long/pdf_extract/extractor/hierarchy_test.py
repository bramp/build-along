from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.hierarchy import (
    build_hierarchy_from_blocks,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    StepNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Drawing,
    Text,
)


def test_build_hierarchy_basic_containment():
    # One big image with a small number inside it
    blocks = [
        Drawing(bbox=BBox(0, 0, 100, 100), image_id="image_0", id=0),
        StepNumber(bbox=BBox(10, 10, 20, 20), value=1),
    ]
    tree = build_hierarchy_from_blocks(blocks)
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
    blocks = [
        Drawing(bbox=BBox(0, 0, 50, 50), image_id="drawing_0", id=0),
        Text(bbox=BBox(5, 5, 10, 10), text="x3", id=1),
    ]
    tree = build_hierarchy_from_blocks(blocks)
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
    outer = Drawing(bbox=BBox(0, 0, 100, 100), image_id="outer", id=0)
    middle = Drawing(bbox=BBox(10, 10, 90, 90), image_id="middle", id=1)
    inner = Text(bbox=BBox(20, 20, 30, 30), text="inner", id=2)
    sibling = Text(bbox=BBox(50, 50, 60, 60), text="sibling", id=3)

    blocks = [outer, middle, inner, sibling]
    tree = build_hierarchy_from_blocks(blocks)

    # Verify depths
    assert tree.get_depth(outer) == 0
    assert tree.get_depth(middle) == 1
    assert tree.get_depth(inner) == 2
    assert tree.get_depth(sibling) == 2  # sibling to inner, both inside middle


def test_build_hierarchy_identical_blocks():
    """Test hierarchy building with duplicate blocks at the same position.

    This can happen when PDFs have duplicate text blocks for visual effects.
    With the fix, identical blocks are treated as independent roots (not parent-child)
    since they occupy exactly the same space.
    """
    # Two identical text blocks (duplicates)
    block1 = Text(bbox=BBox(10, 10, 50, 30), text="1", id=0)
    block2 = Text(bbox=BBox(10, 10, 50, 30), text="1", id=1)
    # A different block
    block3 = Text(bbox=BBox(100, 100, 150, 120), text="2", id=2)

    blocks = [block1, block2, block3]
    tree = build_hierarchy_from_blocks(blocks)

    # Should not crash, and all blocks should be in the tree
    all_blocks_in_tree = set()
    for root in tree.roots:
        all_blocks_in_tree.add(id(root))

        def collect_descendants(block):
            for child in tree.get_children(block):
                all_blocks_in_tree.add(id(child))
                collect_descendants(child)

        collect_descendants(root)

    # All blocks should be accounted for in the tree
    assert len(all_blocks_in_tree) == 3

    # With the fix, all blocks should be roots (no parent-child for identical)
    roots = tree.roots
    assert len(roots) == 3, "All blocks should be independent roots"
    assert block1 in roots
    assert block2 in roots
    assert block3 in roots


def test_build_hierarchy_same_area_different_position():
    """Test blocks with same area but different positions.

    These blocks should be independent roots, not parent-child.
    """
    # Two blocks with same area but different positions
    block1 = Text(bbox=BBox(10, 10, 30, 30), text="A", id=0)  # area = 400
    block2 = Text(bbox=BBox(50, 50, 70, 70), text="B", id=1)  # area = 400

    blocks = [block1, block2]
    tree = build_hierarchy_from_blocks(blocks)

    # Both should be roots (not contained in each other)
    assert len(tree.roots) == 2
    assert block1 in tree.roots
    assert block2 in tree.roots
    assert tree.get_parent(block1) is None
    assert tree.get_parent(block2) is None


def test_build_hierarchy_nested_same_area():
    """Test nested containment where child has same area as parent.

    Edge case: if bboxes are identical, one will be the parent of the other.
    """
    # Outer container
    outer = Drawing(bbox=BBox(0, 0, 100, 100), image_id="outer", id=0)
    # Two blocks with identical bbox inside outer
    inner1 = Text(bbox=BBox(20, 20, 40, 40), text="X", id=1)
    inner2 = Text(bbox=BBox(20, 20, 40, 40), text="X", id=2)

    blocks = [outer, inner1, inner2]
    tree = build_hierarchy_from_blocks(blocks)

    # outer should be the root
    assert len(tree.roots) == 1
    assert tree.roots[0] == outer

    # Both inner blocks should be descendants of outer
    outer_children = tree.get_children(outer)
    # One of inner1/inner2 should be direct child of outer
    # The other should be child of its sibling (due to identical bbox)
    assert len(outer_children) >= 1

    # Verify all blocks are in the tree
    all_descendants = {id(outer)}

    def collect_all(block):
        for child in tree.get_children(block):
            all_descendants.add(id(child))
            collect_all(child)

    collect_all(outer)
    assert len(all_descendants) == 3
