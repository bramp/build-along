from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Part,
    PartCount,
    PartsList,
    StepNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Drawing,
    Image,
    Text,
)


def test_step_number():
    sn = StepNumber(bbox=BBox(0, 0, 10, 10), value=3)
    assert sn.value == 3
    assert isinstance(sn.bbox, BBox)


def test_drawing_optional_id():
    d = Drawing(bbox=BBox(1, 1, 100, 100), id=0)
    assert d.image_id is None
    d2 = Drawing(bbox=BBox(1, 1, 100, 100), image_id="img_1", id=1)
    assert d2.image_id == "img_1"


def test_part_and_count():
    cnt = PartCount(bbox=BBox(5, 5, 7, 7), count=2)
    p = Part(bbox=BBox(0, 0, 10, 10), count=cnt)
    assert p.count.count == 2
    assert p.bbox.x1 == 10


def test_parts_list_total_items():
    p1 = Part(
        bbox=BBox(0, 0, 10, 10),
        count=PartCount(bbox=BBox(8, 0, 10, 2), count=2),
    )
    p2 = Part(
        bbox=BBox(0, 10, 10, 20),
        count=PartCount(bbox=BBox(8, 10, 10, 12), count=5),
    )
    pl = PartsList(bbox=BBox(0, 0, 100, 200), parts=[p1, p2])
    assert pl.total_items == 7


def test_partcount_non_negative():
    PartCount(bbox=BBox(0, 0, 1, 1), count=0)  # ok
    try:
        # TODO Is this the correct way to test for ValueError?
        PartCount(bbox=BBox(0, 0, 1, 1), count=-1)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_text_serialization_with_tag():
    """Test that Text blocks serialize with __tag__ field."""
    text = Text(bbox=BBox(0, 0, 10, 10), id=0, text="Hello World")

    # Serialize with by_alias=True to get __tag__ instead of tag
    json_data = text.model_dump(by_alias=True)

    assert json_data["__tag__"] == "Text"
    assert json_data["text"] == "Hello World"
    assert json_data["id"] == 0
    assert "tag" not in json_data  # Should use alias __tag__


def test_drawing_serialization_with_tag():
    """Test that Drawing blocks serialize with __tag__ field."""
    drawing = Drawing(bbox=BBox(0, 0, 10, 10), id=1, image_id="img_123")

    json_data = drawing.model_dump(by_alias=True)

    assert json_data["__tag__"] == "Drawing"
    assert json_data["image_id"] == "img_123"
    assert json_data["id"] == 1


def test_image_serialization_with_tag():
    """Test that Image blocks serialize with __tag__ field."""
    image = Image(bbox=BBox(0, 0, 10, 10), id=2, image_id="img_456")

    json_data = image.model_dump(by_alias=True)

    assert json_data["__tag__"] == "Image"
    assert json_data["image_id"] == "img_456"
    assert json_data["id"] == 2


def test_text_deserialization_from_tagged_json():
    """Test that Text blocks can be deserialized from JSON with __tag__."""
    json_str = (
        '{"__tag__": "Text", "bbox": {"x0": 0, "y0": 0, "x1": 10, "y1": 10}, '
        '"id": 0, "text": "Test"}'
    )

    text = Text.model_validate_json(json_str)

    assert isinstance(text, Text)
    assert text.text == "Test"
    assert text.id == 0


def test_drawing_deserialization_from_tagged_json():
    """Test that Drawing blocks can be deserialized from JSON with __tag__."""
    json_str = (
        '{"__tag__": "Drawing", "bbox": {"x0": 0, "y0": 0, "x1": 10, "y1": 10}, '
        '"id": 1}'
    )

    drawing = Drawing.model_validate_json(json_str)

    assert isinstance(drawing, Drawing)
    assert drawing.id == 1
    assert drawing.image_id is None


def test_image_deserialization_from_tagged_json():
    """Test that Image blocks can be deserialized from JSON with __tag__."""
    json_str = (
        '{"__tag__": "Image", "bbox": {"x0": 0, "y0": 0, "x1": 10, "y1": 10}, '
        '"id": 2, "image_id": "test"}'
    )

    image = Image.model_validate_json(json_str)

    assert isinstance(image, Image)
    assert image.id == 2
    assert image.image_id == "test"


def test_block_round_trip_serialization():
    """Test that blocks can be serialized and deserialized without data loss."""
    original_text = Text(
        bbox=BBox(1.5, 2.5, 10.5, 20.5),
        id=42,
        text="Round trip test",
        font_name="Arial",
        font_size=12.0,
    )

    # Serialize with alias
    json_str = original_text.model_dump_json(by_alias=True)

    # Deserialize
    restored = Text.model_validate_json(json_str)

    assert restored.bbox == original_text.bbox
    assert restored.id == original_text.id
    assert restored.text == original_text.text
    assert restored.font_name == original_text.font_name
    assert restored.font_size == original_text.font_size
