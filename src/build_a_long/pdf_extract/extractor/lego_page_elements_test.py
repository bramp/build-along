"""Tests for LEGO page elements serialization and deserialization."""

from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    PageNumber,
    Part,
    PartCount,
    PartsList,
    StepNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import Drawing


def test_page_number_serialization_with_tag():
    """Test that PageNumber serializes with __tag__ field."""
    page_num = PageNumber(bbox=BBox(0, 0, 10, 10), value=5)

    # Serialize with by_alias=True to get __tag__ instead of tag
    json_data = page_num.model_dump(by_alias=True)

    assert json_data["__tag__"] == "PageNumber"
    assert json_data["value"] == 5
    assert "bbox" in json_data
    assert "tag" not in json_data  # Should use alias __tag__


def test_step_number_serialization_with_tag():
    """Test that StepNumber serializes with __tag__ field."""
    step_num = StepNumber(bbox=BBox(0, 0, 10, 10), value=3)

    json_data = step_num.model_dump(by_alias=True)

    assert json_data["__tag__"] == "StepNumber"
    assert json_data["value"] == 3
    assert "bbox" in json_data
    assert "tag" not in json_data


def test_part_count_serialization_with_tag():
    """Test that PartCount serializes with __tag__ field."""
    part_count = PartCount(bbox=BBox(0, 0, 10, 10), count=2)

    json_data = part_count.model_dump(by_alias=True)

    assert json_data["__tag__"] == "PartCount"
    assert json_data["count"] == 2
    assert "bbox" in json_data
    assert "tag" not in json_data


def test_part_serialization_with_nested_elements():
    """Test that Part serializes correctly with nested PartCount and Drawing."""
    count = PartCount(bbox=BBox(0, 0, 5, 5), count=3)
    diagram = Drawing(bbox=BBox(10, 10, 20, 20), id=1, image_id="img_123")
    part = Part(
        bbox=BBox(0, 0, 25, 25),
        count=count,
        diagram=diagram,
        name="Brick 2x4",
        number="3001",
    )

    json_data = part.model_dump(by_alias=True)

    assert json_data["__tag__"] == "Part"
    assert json_data["name"] == "Brick 2x4"
    assert json_data["number"] == "3001"
    assert json_data["count"]["__tag__"] == "PartCount"
    assert json_data["count"]["count"] == 3
    assert json_data["diagram"]["__tag__"] == "Drawing"
    assert json_data["diagram"]["image_id"] == "img_123"


def test_parts_list_serialization():
    """Test that PartsList serializes correctly with list of parts."""
    part1 = Part(
        bbox=BBox(0, 0, 10, 10),
        count=PartCount(bbox=BBox(0, 0, 5, 5), count=2),
        name="Brick",
        number="3001",
    )
    part2 = Part(
        bbox=BBox(0, 10, 10, 20),
        count=PartCount(bbox=BBox(0, 10, 5, 15), count=5),
        name="Plate",
        number="3020",
    )
    parts_list = PartsList(bbox=BBox(0, 0, 100, 100), parts=[part1, part2])

    json_data = parts_list.model_dump(by_alias=True)

    assert json_data["__tag__"] == "PartsList"
    assert len(json_data["parts"]) == 2
    assert json_data["parts"][0]["__tag__"] == "Part"
    assert json_data["parts"][0]["name"] == "Brick"
    assert json_data["parts"][1]["__tag__"] == "Part"
    assert json_data["parts"][1]["name"] == "Plate"


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
            "__tag__": "Drawing",
            "bbox": {"x0": 10, "y0": 10, "x1": 20, "y1": 20},
            "id": 1,
            "image_id": "img_123"
        },
        "name": "Brick 2x4",
        "number": "3001"
    }
    """

    part = Part.model_validate_json(json_str)

    assert isinstance(part, Part)
    assert part.name == "Brick 2x4"
    assert part.number == "3001"
    assert part.count.count == 3
    assert isinstance(part.diagram, Drawing)
    assert part.diagram.image_id == "img_123"


def test_lego_element_round_trip_serialization():
    """Test that elements can be serialized and deserialized without data loss."""
    original = StepNumber(bbox=BBox(1.5, 2.5, 10.5, 20.5), value=42)

    # Serialize with alias
    json_str = original.model_dump_json(by_alias=True)

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
                name="Test Part",
                number="12345",
            )
        ],
    )

    json_str = original.model_dump_json(by_alias=True)
    restored = PartsList.model_validate_json(json_str)

    assert restored.bbox == original.bbox
    assert len(restored.parts) == 1
    assert restored.parts[0].name == "Test Part"
    assert restored.parts[0].count.count == 2
