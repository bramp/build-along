"""Tests for LEGO page elements serialization and deserialization."""

from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Manual,
    Page,
    PageNumber,
    Part,
    PartCount,
    PartImage,
    PartsList,
    Step,
    StepNumber,
)



# ============================================================================
# Manual tests
# ============================================================================


def _make_sample_manual() -> Manual:
    """Create a sample manual for testing."""
    bbox = BBox(x0=0, y0=0, x1=100, y1=100)

    page1 = Page(
        bbox=bbox,
        pdf_page_number=1,
        categories={Page.PageType.INSTRUCTION},
        page_number=PageNumber(bbox=bbox, value=1),
        steps=[
            Step(
                bbox=bbox,
                step_number=StepNumber(bbox=bbox, value=1),
                parts_list=PartsList(
                    bbox=bbox,
                    parts=[
                        Part(bbox=bbox, count=PartCount(bbox=bbox, count=2)),
                        Part(bbox=bbox, count=PartCount(bbox=bbox, count=3)),
                    ],
                ),
            ),
            Step(
                bbox=bbox,
                step_number=StepNumber(bbox=bbox, value=2),
                parts_list=PartsList(
                    bbox=bbox,
                    parts=[
                        Part(bbox=bbox, count=PartCount(bbox=bbox, count=1)),
                    ],
                ),
            ),
        ],
    )

    page2 = Page(
        bbox=bbox,
        pdf_page_number=2,
        categories={Page.PageType.INSTRUCTION},
        page_number=PageNumber(bbox=bbox, value=2),
        steps=[
            Step(
                bbox=bbox,
                step_number=StepNumber(bbox=bbox, value=3),
                parts_list=PartsList(
                    bbox=bbox,
                    parts=[
                        Part(bbox=bbox, count=PartCount(bbox=bbox, count=4)),
                    ],
                ),
            ),
        ],
    )

    page3 = Page(
        bbox=bbox,
        pdf_page_number=180,
        categories={Page.PageType.CATALOG},
        page_number=PageNumber(bbox=bbox, value=180),
        catalog=[
            Part(bbox=bbox, count=PartCount(bbox=bbox, count=5)),
            Part(bbox=bbox, count=PartCount(bbox=bbox, count=10)),
        ],
    )

    return Manual(pages=[page1, page2, page3], set_number="75375", name="Test Set")


def test_manual_get_page_by_number():
    """Test getting a page by its page number."""
    manual = _make_sample_manual()

    page1 = manual.get_page(1)
    assert page1 is not None
    assert page1.page_number is not None
    assert page1.page_number.value == 1

    page180 = manual.get_page(180)
    assert page180 is not None
    assert page180.page_number is not None
    assert page180.page_number.value == 180

    # Non-existent page
    assert manual.get_page(999) is None


def test_manual_instruction_pages():
    """Test filtering instruction pages."""
    manual = _make_sample_manual()
    instruction_pages = manual.instruction_pages
    assert len(instruction_pages) == 2
    assert all(p.is_instruction for p in instruction_pages)


def test_manual_catalog_pages():
    """Test filtering catalog pages."""
    manual = _make_sample_manual()
    catalog_pages = manual.catalog_pages
    assert len(catalog_pages) == 1
    assert all(p.is_catalog for p in catalog_pages)


def test_manual_catalog_parts():
    """Test getting all parts from catalog pages."""
    manual = _make_sample_manual()
    catalog_parts = manual.catalog_parts
    assert len(catalog_parts) == 2
    assert catalog_parts[0].count.count == 5
    assert catalog_parts[1].count.count == 10


def test_manual_all_steps():
    """Test getting all steps from instruction pages."""
    manual = _make_sample_manual()
    all_steps = manual.all_steps
    assert len(all_steps) == 3
    assert [s.step_number.value for s in all_steps] == [1, 2, 3]


def test_manual_total_parts_count():
    """Test total parts count across all steps."""
    manual = _make_sample_manual()
    # Step 1: 2+3=5, Step 2: 1, Step 3: 4 => Total: 10
    assert manual.total_parts_count == 10


def test_manual_str_representation():
    """Test string representation."""
    manual = _make_sample_manual()
    str_repr = str(manual)
    assert "75375" in str_repr
    assert "Test Set" in str_repr
    assert "pages=3" in str_repr



def test_manual_empty():
    """Test an empty manual."""
    manual = Manual(pages=[])
    assert len(manual.pages) == 0
    assert len(manual.instruction_pages) == 0
    assert len(manual.catalog_pages) == 0
    assert len(manual.all_steps) == 0
    assert len(manual.catalog_parts) == 0
    assert manual.total_parts_count == 0
    assert manual.get_page(1) is None
