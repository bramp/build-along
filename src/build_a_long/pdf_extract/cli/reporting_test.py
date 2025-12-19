"""Tests for the reporting module."""

import io
from contextlib import redirect_stdout

from build_a_long.pdf_extract.cli.reporting import (
    print_page_hierarchy,
)
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.extractor import PageData
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Diagram,
    InstructionContent,
    Page,
    PageNumber,
    Part,
    PartCount,
    PartNumber,
    PartsList,
    Step,
    StepNumber,
)


def test_print_page_hierarchy_with_part_numbers() -> None:
    """Test that print_page_hierarchy handles Part with PartNumber correctly."""
    # Create a part with a PartNumber
    part_with_number = Part(
        bbox=BBox(0, 0, 10, 10),
        count=PartCount(bbox=BBox(0, 0, 5, 5), count=2),
        number=PartNumber(bbox=BBox(0, 5, 5, 10), element_id="6208370"),
    )

    # Create a part without a PartNumber
    part_without_number = Part(
        bbox=BBox(10, 0, 20, 10),
        count=PartCount(bbox=BBox(10, 0, 15, 5), count=3),
    )

    parts_list = PartsList(
        bbox=BBox(0, 0, 20, 10), parts=[part_with_number, part_without_number]
    )

    step = Step(
        bbox=BBox(0, 0, 25, 30),
        step_number=StepNumber(bbox=BBox(0, 10, 5, 15), value=1),
        parts_list=parts_list,
        diagram=Diagram(bbox=BBox(0, 15, 20, 25)),
    )

    page = Page(
        bbox=BBox(0, 0, 25, 35),
        pdf_page_number=1,
        page_number=PageNumber(bbox=BBox(0, 25, 5, 30), value=1),
        instruction=InstructionContent(steps=[step]),
    )

    page_data = PageData(page_number=1, bbox=BBox(0, 0, 25, 35), blocks=[])

    # Capture the output
    output = io.StringIO()
    with redirect_stdout(output):
        print_page_hierarchy(page_data, page)

    result = output.getvalue()

    # Verify the output contains the expected information
    assert "Page 1:" in result
    assert "✓ Page Number: 1" in result
    assert "✓ Steps: 1" in result
    assert "Step 1 (2 parts)" in result
    assert "Parts List:" in result
    assert "2x (6208370)" in result  # Part with element_id
    assert "3x (no number)" in result  # Part without number


def test_print_page_hierarchy_empty_parts_list() -> None:
    """Test that print_page_hierarchy handles empty parts list correctly."""
    parts_list = PartsList(bbox=BBox(0, 0, 20, 10), parts=[])

    step = Step(
        bbox=BBox(0, 0, 25, 30),
        step_number=StepNumber(bbox=BBox(0, 10, 5, 15), value=1),
        parts_list=parts_list,
        diagram=Diagram(bbox=BBox(0, 15, 20, 25)),
    )

    page = Page(
        bbox=BBox(0, 0, 25, 35),
        pdf_page_number=1,
        page_number=PageNumber(bbox=BBox(0, 25, 5, 30), value=1),
        instruction=InstructionContent(steps=[step]),
    )

    page_data = PageData(page_number=1, bbox=BBox(0, 0, 25, 35), blocks=[])

    output = io.StringIO()
    with redirect_stdout(output):
        print_page_hierarchy(page_data, page)

    result = output.getvalue()

    assert "Parts List: (none)" in result
