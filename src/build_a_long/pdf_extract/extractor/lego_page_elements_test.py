"""Tests for LEGO page elements serialization and deserialization."""

import json

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


def test_page_number_serialization_with_tag():
    """Test that PageNumber serializes with __tag__ field."""
    page_num = PageNumber(bbox=BBox(0, 0, 10, 10), value=5)

    # Serialize with by_alias=True to get __tag__ instead of tag
    json_data = page_num.to_dict()

    assert json_data["__tag__"] == "PageNumber"
    assert json_data["value"] == 5
    assert "bbox" in json_data
    assert "tag" not in json_data  # Should use alias __tag__


def test_step_number_serialization_with_tag():
    """Test that StepNumber serializes with __tag__ field."""
    step_num = StepNumber(bbox=BBox(0, 0, 10, 10), value=7)

    json_data = step_num.to_dict()

    assert json_data["__tag__"] == "StepNumber"
    assert json_data["value"] == 7
    assert "bbox" in json_data
    assert "tag" not in json_data


def test_part_count_serialization_with_tag():
    """Test that PartCount serializes with __tag__ field."""
    part_count = PartCount(bbox=BBox(0, 0, 10, 10), count=2)

    json_data = part_count.to_dict()

    assert json_data["__tag__"] == "PartCount"
    assert json_data["count"] == 2
    assert "bbox" in json_data
    assert "tag" not in json_data


def test_part_serialization_with_nested_elements():
    """Test that Part serializes correctly with nested PartCount and PartImage."""
    count = PartCount(bbox=BBox(0, 0, 5, 5), count=3)
    diagram = PartImage(bbox=BBox(10, 10, 20, 20))
    part = Part(
        bbox=BBox(0, 0, 25, 25),
        count=count,
        diagram=diagram,
    )

    json_data = part.to_dict()

    assert json_data["__tag__"] == "Part"
    assert "number" not in json_data  # None values excluded
    assert json_data["count"]["__tag__"] == "PartCount"
    assert json_data["count"]["count"] == 3
    assert json_data["diagram"]["__tag__"] == "PartImage"


def test_parts_list_serialization():
    """Test that PartsList serializes correctly with list of parts."""
    part1 = Part(
        bbox=BBox(0, 0, 10, 10),
        count=PartCount(bbox=BBox(0, 0, 5, 5), count=2),
    )
    part2 = Part(
        bbox=BBox(0, 10, 10, 20),
        count=PartCount(bbox=BBox(0, 10, 5, 15), count=5),
    )
    parts_list = PartsList(bbox=BBox(0, 0, 100, 100), parts=[part1, part2])

    json_data = parts_list.to_dict()

    assert json_data["__tag__"] == "PartsList"
    assert len(json_data["parts"]) == 2
    assert json_data["parts"][0]["__tag__"] == "Part"
    assert json_data["parts"][1]["__tag__"] == "Part"


def test_page_number_deserialization():
    """Test that PageNumber can be deserialized from JSON."""
    json_str = '{"bbox": {"x0": 0, "y0": 0, "x1": 10, "y1": 10}, "value": 7}'

    page_num = PageNumber.model_validate_json(json_str)

    assert isinstance(page_num, PageNumber)
    assert page_num.value == 7
    assert page_num.bbox == BBox(0, 0, 10, 10)


def test_step_number_deserialization():
    """Test that StepNumber can be deserialized from JSON."""
    json_str = '{"bbox": {"x0": 0, "y0": 0, "x1": 10, "y1": 10}, "value": 2}'

    step_num = StepNumber.model_validate_json(json_str)

    assert isinstance(step_num, StepNumber)
    assert step_num.value == 2


def test_part_count_deserialization():
    """Test that PartCount can be deserialized from JSON."""
    json_str = '{"bbox": {"x0": 0, "y0": 0, "x1": 10, "y1": 10}, "count": 4}'

    part_count = PartCount.model_validate_json(json_str)

    assert isinstance(part_count, PartCount)
    assert part_count.count == 4


def test_part_deserialization_with_nested_elements():
    """Test that Part can be deserialized from JSON with nested elements."""
    json_str = """
    {
        "bbox": {"x0": 0, "y0": 0, "x1": 25, "y1": 25},
        "count": {
            "bbox": {"x0": 0, "y0": 0, "x1": 5, "y1": 5},
            "count": 3
        },
        "diagram": {
            "__tag__": "PartImage",
            "bbox": {"x0": 10, "y0": 10, "x1": 20, "y1": 20}
        },
        "number": null
    }
    """

    part = Part.model_validate_json(json_str)

    assert isinstance(part, Part)
    assert part.number is None
    assert part.count.count == 3
    assert isinstance(part.diagram, PartImage)


def test_lego_element_round_trip_serialization():
    """Test that elements can be serialized and deserialized without data loss."""
    original = StepNumber(bbox=BBox(1.5, 2.5, 10.5, 20.5), value=42)

    # Serialize with alias
    json_str = original.to_json()

    # Deserialize
    restored = StepNumber.model_validate_json(json_str)

    assert restored.bbox == original.bbox
    assert restored.value == original.value


def test_parts_list_round_trip():
    """Test PartsList round-trip serialization."""
    original = PartsList(
        bbox=BBox(0, 0, 100, 100),
        parts=[
            Part(
                bbox=BBox(0, 0, 10, 10),
                count=PartCount(bbox=BBox(0, 0, 5, 5), count=2),
            )
        ],
    )

    json_str = original.to_json()
    restored = PartsList.model_validate_json(json_str)

    assert restored.bbox == original.bbox
    assert len(restored.parts) == 1
    assert restored.parts[0].count.count == 2


def test_to_dict_excludes_none_and_uses_tag():
    """Test that to_dict() helper enforces by_alias=True and exclude_none=True."""
    # Part with None fields
    part = Part(
        bbox=BBox(0, 0, 10, 10),
        count=PartCount(bbox=BBox(0, 0, 5, 5), count=3),
    )

    # Use the helper method
    data = part.to_dict()

    # Should use __tag__ not tag
    assert "__tag__" in data
    assert data["__tag__"] == "Part"
    assert "tag" not in data

    # None fields should be excluded
    assert "diagram" not in data
    assert "number" not in data
    assert "length" not in data

    # Nested elements should also use __tag__
    assert data["count"]["__tag__"] == "PartCount"


def test_to_json_excludes_none_and_uses_tag():
    """Test that to_json() helper enforces by_alias=True and exclude_none=True."""
    # StepNumber with no None fields
    step = StepNumber(bbox=BBox(0, 0, 10, 10), value=5)

    # Use the helper method
    json_str = step.to_json()
    data = json.loads(json_str)

    # Should use __tag__ not tag
    assert "__tag__" in data
    assert data["__tag__"] == "StepNumber"
    assert "tag" not in data
    assert data["value"] == 5


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


def test_manual_to_dict():
    """Test serialization to dict."""
    manual = _make_sample_manual()
    data = manual.to_dict()
    assert data["__tag__"] == "Manual"
    assert data["set_number"] == "75375"
    assert data["name"] == "Test Set"
    assert len(data["pages"]) == 3


def test_manual_to_json():
    """Test serialization to JSON."""
    manual = _make_sample_manual()
    json_str = manual.to_json()
    assert '"__tag__":"Manual"' in json_str
    assert '"set_number":"75375"' in json_str


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
