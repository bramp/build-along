"""Tests for drawing module."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pymupdf

from build_a_long.pdf_extract.classifier.classification_result import (
    Candidate,
    ClassificationResult,
    RemovalReason,
)
from build_a_long.pdf_extract.drawing.drawing import (
    DrawableItem,
    _create_drawable_items,
    draw_and_save_bboxes,
)
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.extractor import PageData
from build_a_long.pdf_extract.extractor.lego_page_elements import Page, PageNumber
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image, Text


def _make_candidate(
    label: str,
    bbox: BBox,
    source_block: Text | Drawing | Image | None = None,
    has_constructed: bool = True,
) -> Candidate:
    """Helper to create a Candidate with minimal required fields.

    Args:
        label: The label for this candidate
        bbox: The bounding box
        source_block: The source block if any
        has_constructed: Whether construction succeeded (creates a PageNumber if True)
    """
    return Candidate(
        label=label,
        bbox=bbox,
        source_block=source_block,
        score=0.9,
        score_details={},
        constructed=PageNumber(bbox=bbox, value=1) if has_constructed else None,
    )


def test_create_drawable_items_empty():
    """Test creating drawable items with no blocks or elements."""
    page_data = PageData(page_number=1, bbox=BBox(0, 0, 100, 100), blocks=[])
    result = ClassificationResult(page_data=page_data)
    items = _create_drawable_items(
        result, draw_blocks=False, draw_elements=False, draw_deleted=False
    )
    assert items == []


def test_create_drawable_items_blocks_only():
    """Test creating drawable items with only blocks."""
    block1 = Text(id=1, bbox=BBox(0, 0, 10, 10), text="Hello")
    block2 = Drawing(id=2, bbox=BBox(20, 20, 30, 30))
    result = ClassificationResult(
        page_data=PageData(
            page_number=1, bbox=BBox(0, 0, 200, 200), blocks=[block1, block2]
        )
    )

    items = _create_drawable_items(
        result, draw_blocks=True, draw_elements=False, draw_deleted=False
    )

    assert len(items) == 2
    assert all(isinstance(item, DrawableItem) for item in items)
    assert items[0].bbox == block1.bbox
    assert items[1].bbox == block2.bbox
    assert not items[0].is_element
    assert not items[1].is_element


def test_create_drawable_items_elements_only():
    """Test creating drawable items with only elements."""
    block = Text(id=1, bbox=BBox(0, 0, 10, 10), text="1")
    result = ClassificationResult(
        page_data=PageData(page_number=1, bbox=BBox(0, 0, 200, 200), blocks=[block])
    )

    # Create an element candidate with a source block
    page_number = PageNumber(bbox=block.bbox, value=1)
    candidate = Candidate(
        label="PageNumber",
        bbox=block.bbox,
        source_block=block,
        score=0.9,
        score_details={},
        constructed=page_number,
    )
    result.add_candidate("PageNumber", candidate)

    # Create a Page that includes this PageNumber
    page = Page(bbox=BBox(0, 0, 200, 200), page_number=page_number)
    page_candidate = Candidate(
        label="page",
        bbox=BBox(0, 0, 200, 200),
        source_block=None,
        score=1.0,
        score_details={},
        constructed=page,
    )
    result.add_candidate("page", page_candidate)

    items = _create_drawable_items(
        result, draw_blocks=False, draw_elements=True, draw_deleted=False
    )

    assert len(items) == 2  # PageNumber + Page
    assert any(item.label == "[PageNumber]" for item in items)
    assert items[0].is_element
    assert items[0].bbox == block.bbox
    assert "[PageNumber]" in items[0].label


def test_create_drawable_items_no_duplicate_blocks():
    """Test that blocks used as element sources are not drawn twice."""
    block = Text(id=1, bbox=BBox(0, 0, 10, 10), text="1")
    result = ClassificationResult(
        page_data=PageData(page_number=1, bbox=BBox(0, 0, 200, 200), blocks=[block])
    )

    # Create an element candidate with this block as source
    page_number = PageNumber(bbox=block.bbox, value=1)
    candidate = Candidate(
        label="PageNumber",
        bbox=block.bbox,
        source_block=block,
        score=0.9,
        score_details={},
        constructed=page_number,
    )
    result.add_candidate("PageNumber", candidate)

    # Create a Page that includes this PageNumber
    page = Page(bbox=BBox(0, 0, 200, 200), page_number=page_number)
    page_candidate = Candidate(
        label="page",
        bbox=BBox(0, 0, 200, 200),
        source_block=None,
        score=1.0,
        score_details={},
        constructed=page,
    )
    result.add_candidate("page", page_candidate)

    # Request both blocks and elements
    items = _create_drawable_items(
        result, draw_blocks=True, draw_elements=True, draw_deleted=False
    )

    # Should have 2 items (Page + PageNumber element), block should be filtered out
    assert len(items) == 2
    assert sum(item.is_element for item in items) == 2
    assert "[PageNumber]" in items[0].label


def test_create_drawable_items_element_without_source_block():
    """Test creating drawable items for elements without source blocks."""
    result = ClassificationResult(
        page_data=PageData(page_number=1, bbox=BBox(0, 0, 200, 200), blocks=[])
    )

    # Create a Page element (has no source block)
    page = Page(bbox=BBox(5, 5, 15, 15))
    candidate = Candidate(
        label="page",
        bbox=BBox(5, 5, 15, 15),
        source_block=None,
        score=1.0,
        score_details={},
        constructed=page,
    )
    result.add_candidate("page", candidate)

    items = _create_drawable_items(
        result, draw_blocks=False, draw_elements=True, draw_deleted=False
    )

    assert len(items) == 1
    assert items[0].label == "[page]"
    assert items[0].is_element
    assert items[0].bbox == candidate.bbox


def test_create_drawable_items_filter_removed_blocks():
    """Test that removed blocks are filtered when draw_deleted=False."""
    block1 = Text(id=1, bbox=BBox(0, 0, 10, 10), text="Keep")
    block2 = Text(id=2, bbox=BBox(20, 20, 30, 30), text="Remove")
    result = ClassificationResult(
        page_data=PageData(
            page_number=1, bbox=BBox(0, 0, 200, 200), blocks=[block1, block2]
        )
    )
    reason = RemovalReason(reason_type="test", target_block=block1)
    result.mark_removed(block2, reason=reason)

    items = _create_drawable_items(
        result, draw_blocks=True, draw_elements=False, draw_deleted=False
    )

    assert len(items) == 1
    assert items[0].bbox == block1.bbox
    assert "[REMOVED]" not in items[0].label


def test_create_drawable_items_include_removed_blocks():
    """Test that removed blocks are included when draw_deleted=True."""
    block1 = Text(id=1, bbox=BBox(0, 0, 10, 10), text="Keep")
    block2 = Text(id=2, bbox=BBox(20, 20, 30, 30), text="Remove")
    result = ClassificationResult(
        page_data=PageData(
            page_number=1, bbox=BBox(0, 0, 200, 200), blocks=[block1, block2]
        )
    )
    reason = RemovalReason(reason_type="test", target_block=block1)
    result.mark_removed(block2, reason=reason)

    items = _create_drawable_items(
        result, draw_blocks=True, draw_elements=False, draw_deleted=True
    )

    assert len(items) == 2
    assert "[REMOVED]" in items[1].label


def test_create_drawable_items_filter_non_winner_elements():
    """Test that non-winner elements are filtered when draw_deleted=False."""
    block = Text(id=1, bbox=BBox(0, 0, 10, 10), text="1")
    result = ClassificationResult(
        page_data=PageData(page_number=1, bbox=BBox(0, 0, 200, 200), blocks=[block])
    )

    # Create a winner candidate (will be added to the Page)
    page_number = PageNumber(bbox=block.bbox, value=1)
    winner_candidate = Candidate(
        label="PageNumber",
        bbox=block.bbox,
        source_block=block,
        score=0.9,
        score_details={},
        constructed=page_number,
    )

    # Create a non-winner candidate (constructed but not in the Page)
    step_number_elem = PageNumber(
        bbox=block.bbox, value=2
    )  # Using PageNumber for simplicity
    non_winner_candidate = Candidate(
        label="StepNumber",
        bbox=block.bbox,
        source_block=block,
        score=0.8,
        score_details={},
        constructed=step_number_elem,
    )

    result.add_candidate("PageNumber", winner_candidate)
    result.add_candidate("StepNumber", non_winner_candidate)

    # Build a Page that only includes the winner candidate's constructed element
    # This simulates the page builder choosing only the PageNumber
    page = Page(bbox=BBox(0, 0, 200, 200), page_number=page_number)
    page_candidate = Candidate(
        label="page",
        bbox=BBox(0, 0, 200, 200),
        source_block=None,
        score=1.0,
        score_details={},
        constructed=page,
    )
    result.add_candidate("page", page_candidate)

    items = _create_drawable_items(
        result, draw_blocks=False, draw_elements=True, draw_deleted=False
    )

    # Only the winners should be included (Page + PageNumber)
    assert len(items) == 2
    assert any("[PageNumber]" in item.label for item in items)
    assert not any("[StepNumber]" in item.label for item in items)
    assert all("[NOT WINNER]" not in item.label for item in items)


def test_create_drawable_items_include_non_winner_elements():
    """Test that non-winner elements are included when draw_deleted=True."""
    block = Text(id=1, bbox=BBox(0, 0, 10, 10), text="1")
    result = ClassificationResult(
        page_data=PageData(page_number=1, bbox=BBox(0, 0, 200, 200), blocks=[block])
    )

    # Create a winner candidate (will be added to the Page)
    page_number = PageNumber(bbox=block.bbox, value=1)
    winner_candidate = Candidate(
        label="PageNumber",
        bbox=block.bbox,
        source_block=block,
        score=0.9,
        score_details={},
        constructed=page_number,
    )

    # Create a non-winner candidate (constructed but not in the Page)
    step_number_elem = PageNumber(
        bbox=block.bbox, value=2
    )  # Using PageNumber for simplicity
    non_winner_candidate = Candidate(
        label="StepNumber",
        bbox=block.bbox,
        source_block=block,
        score=0.8,
        score_details={},
        constructed=step_number_elem,
    )

    result.add_candidate("PageNumber", winner_candidate)
    result.add_candidate("StepNumber", non_winner_candidate)

    # Build a Page that only includes the winner candidate's constructed element
    page = Page(bbox=BBox(0, 0, 200, 200), page_number=page_number)
    page_candidate = Candidate(
        label="page",
        bbox=BBox(0, 0, 200, 200),
        source_block=None,
        score=1.0,
        score_details={},
        constructed=page,
    )
    result.add_candidate("page", page_candidate)

    items = _create_drawable_items(
        result, draw_blocks=False, draw_elements=True, draw_deleted=True
    )

    # Both winner and non-winner should be included (Page + PageNumber + StepNumber)
    assert len(items) == 3
    assert any("[PageNumber]" in item.label for item in items)
    # Check for non-winner StepNumber element
    assert any(
        "[NOT WINNER]" in item.label and "[StepNumber]" in item.label for item in items
    )


def test_drawable_item_has_bbox():
    """Test that DrawableItem has a bbox attribute (for hierarchy building)."""
    bbox = BBox(0, 0, 10, 10)
    item = DrawableItem(
        bbox=bbox,
        label="Test",
        is_element=False,
        is_winner=True,
        is_removed=False,
    )
    assert hasattr(item, "bbox")
    assert item.bbox == bbox


def test_draw_and_save_bboxes_smoke_test():
    """Smoke test for draw_and_save_bboxes with a real PDF page."""
    # Create a simple PDF with one page
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)

    # Create some blocks
    block1 = Text(id=1, bbox=BBox(10, 10, 50, 30), text="Hello")
    block2 = Image(id=2, bbox=BBox(60, 60, 120, 100))
    result = ClassificationResult(
        page_data=PageData(
            page_number=1, bbox=BBox(0, 0, 200, 200), blocks=[block1, block2]
        )
    )

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_output.png"

        # Should not raise an exception
        draw_and_save_bboxes(
            page,
            result,
            output_path,
            draw_blocks=True,
            draw_elements=False,
            draw_deleted=False,
        )

        # Check that file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    doc.close()


def test_draw_and_save_bboxes_with_elements():
    """Smoke test for draw_and_save_bboxes with elements."""
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)

    block = Text(id=1, bbox=BBox(10, 10, 50, 30), text="1")
    result = ClassificationResult(
        page_data=PageData(page_number=1, bbox=BBox(0, 0, 200, 200), blocks=[block])
    )

    candidate = _make_candidate("PageNumber", block.bbox, block, True)
    result.add_candidate("PageNumber", candidate)

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_output_elements.png"

        # Should not raise an exception
        draw_and_save_bboxes(
            page,
            result,
            output_path,
            draw_blocks=False,
            draw_elements=True,
            draw_deleted=False,
        )

        # Check that file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    doc.close()


def test_draw_and_save_bboxes_with_nested_bboxes():
    """Test drawing with nested bounding boxes for hierarchy depth calculation."""
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)

    # Create nested blocks (parent contains child)
    parent = Drawing(id=1, bbox=BBox(10, 10, 100, 100))
    child = Text(id=2, bbox=BBox(20, 20, 50, 50), text="Nested")
    result = ClassificationResult(
        page_data=PageData(
            page_number=1, bbox=BBox(0, 0, 200, 200), blocks=[parent, child]
        )
    )

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_nested.png"

        # Should not raise an exception and should handle depth correctly
        draw_and_save_bboxes(
            page,
            result,
            output_path,
            draw_blocks=True,
            draw_elements=False,
            draw_deleted=False,
        )

        assert output_path.exists()

    doc.close()
