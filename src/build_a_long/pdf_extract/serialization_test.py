"""Consolidated serialization tests for all data models.

This module provides a centralized place to verify that all data models:
1. Can be serialized to JSON and deserialized back without data loss (round-trip).
2. Conform to the expected JSON schema (e.g., include "__tag__").
3. Handle optional fields and nested structures correctly.
"""

import pytest
from pydantic import BaseModel

from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Arrow,
    ArrowHead,
    Background,
    BagNumber,
    Diagram,
    Divider,
    LoosePartSymbol,
    Manual,
    OpenBag,
    Page,
    PageNumber,
    Part,
    PartCount,
    PartImage,
    PartNumber,
    PartsList,
    PieceLength,
    Preview,
    ProgressBar,
    ProgressBarIndicator,
    RotationSymbol,
    Scale,
    Shine,
    Step,
    StepCount,
    StepNumber,
    SubAssembly,
    SubStep,
    TriviaText,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Drawing,
    Image,
    Text,
)

# Shared BBox for test instances
BBOX = BBox(0, 0, 10, 10)


def _make_text() -> Text:
    return Text(
        bbox=BBOX,
        id=1,
        text="Hello",
        font_name="Arial",
        font_size=12.0,
        color=0xFF0000,
    )


def _make_image() -> Image:
    return Image(
        bbox=BBOX,
        id=2,
        image_id="img_1",
        width=100,
        height=100,
        xref=5,
        smask=6,
    )


def _make_drawing() -> Drawing:
    return Drawing(
        bbox=BBOX,
        id=3,
        fill_color=(1.0, 0.0, 0.0),
        line_width=2.0,
    )


def _make_part() -> Part:
    return Part(
        bbox=BBOX,
        count=PartCount(bbox=BBOX, count=2),
        diagram=PartImage(bbox=BBox(5, 5, 10, 10), image_id="p1"),
        length=PieceLength(bbox=BBOX, value=4),
    )


def _make_step() -> Step:
    return Step(
        bbox=BBOX,
        step_number=StepNumber(bbox=BBOX, value=5),
        parts_list=PartsList(
            bbox=BBOX,
            parts=[
                Part(bbox=BBOX, count=PartCount(bbox=BBOX, count=1)),
            ],
        ),
        diagram=Diagram(bbox=BBOX),
        rotation_symbol=RotationSymbol(bbox=BBOX),
        arrows=[
            Arrow(
                bbox=BBOX,
                heads=[ArrowHead(tip=(10, 10), direction=90)],
                tail=(0, 0),
            )
        ],
    )


def _make_page() -> Page:
    return Page(
        bbox=BBOX,
        pdf_page_number=10,
        categories={Page.PageType.INSTRUCTION},
        page_number=PageNumber(bbox=BBOX, value=10),
        steps=[_make_step()],
        dividers=[Divider(bbox=BBOX, orientation=Divider.Orientation.VERTICAL)],
    )


def _make_manual() -> Manual:
    return Manual(
        set_number="12345",
        name="Test Set",
        pages=[_make_page()],
        source_hash="abc123hash",
    )


# List of model instances to test
# Each entry is a tuple: (model_instance, expected_tag)
TEST_CASES = [
    (_make_text(), "Text"),
    (_make_image(), "Image"),
    (_make_drawing(), "Drawing"),
    (PageData(page_number=1, bbox=BBOX, blocks=[]), None),  # PageData has no tag
    (PageNumber(bbox=BBOX, value=1), "PageNumber"),
    (StepNumber(bbox=BBOX, value=10), "StepNumber"),
    (PartCount(bbox=BBOX, count=5), "PartCount"),
    (StepCount(bbox=BBOX, count=2), "StepCount"),
    (PartNumber(bbox=BBOX, element_id="3001"), "PartNumber"),
    (PieceLength(bbox=BBOX, value=6), "PieceLength"),
    (Shine(bbox=BBOX), "Shine"),
    (Scale(bbox=BBOX, length=PieceLength(bbox=BBOX, value=1)), "Scale"),
    (PartImage(bbox=BBOX, image_id="img"), "PartImage"),
    (ProgressBarIndicator(bbox=BBOX), "ProgressBarIndicator"),
    (ProgressBar(bbox=BBOX, full_width=100, progress=0.5), "ProgressBar"),
    (Divider(bbox=BBOX, orientation=Divider.Orientation.HORIZONTAL), "Divider"),
    (Background(bbox=BBOX, fill_color=(0.9, 0.9, 0.9)), "Background"),
    (TriviaText(bbox=BBOX, text_lines=["Did you know?"]), "TriviaText"),
    (RotationSymbol(bbox=BBOX), "RotationSymbol"),
    (Arrow(bbox=BBOX, heads=[ArrowHead(tip=(0, 0), direction=0)]), "Arrow"),
    (_make_part(), "Part"),
    (PartsList(bbox=BBOX, parts=[_make_part()]), "PartsList"),
    (BagNumber(bbox=BBOX, value=1), "BagNumber"),
    (LoosePartSymbol(bbox=BBOX), "LoosePartSymbol"),
    (OpenBag(bbox=BBOX, number=BagNumber(bbox=BBOX, value=1)), "OpenBag"),
    (Diagram(bbox=BBOX), "Diagram"),
    (Preview(bbox=BBOX), "Preview"),
    (
        SubStep(
            bbox=BBOX,
            step_number=StepNumber(bbox=BBOX, value=1),
            diagram=Diagram(bbox=BBOX),
        ),
        "SubStep",
    ),
    (SubAssembly(bbox=BBOX, diagram=Diagram(bbox=BBOX)), "SubAssembly"),
    (_make_step(), "Step"),
    (_make_page(), "Page"),
    (_make_manual(), "Manual"),
]


@pytest.mark.parametrize("instance, expected_tag", TEST_CASES)
def test_round_trip_serialization(instance: BaseModel, expected_tag: str | None):
    """Test that the model can be serialized and deserialized without data loss."""
    # Serialize
    to_json = getattr(instance, "to_json", None)
    if to_json is not None:
        json_str = to_json()
    else:
        # Fallback for standard Pydantic models (like PageData)
        json_str = instance.model_dump_json(by_alias=True, exclude_none=True)

    # Deserialize
    model_class = type(instance)
    restored = model_class.model_validate_json(json_str)

    # Assert equality
    # Note: We compare model_dump() output to handle float precision and nested objects
    # consistently, rather than direct object equality which might be strict on types.
    assert restored.model_dump() == instance.model_dump()


@pytest.mark.parametrize("instance, expected_tag", TEST_CASES)
def test_json_structure(instance: BaseModel, expected_tag: str | None):
    """Test that the serialized JSON has the expected structure (tags, aliases)."""
    to_dict = getattr(instance, "to_dict", None)
    if to_dict is not None:
        data = to_dict()
    else:
        data = instance.model_dump(by_alias=True, exclude_none=True)

    # Check for __tag__ if expected
    if expected_tag:
        assert "__tag__" in data
        assert data["__tag__"] == expected_tag
        assert "tag" not in data  # Should use alias

    # Check that None fields are excluded (simple check on a known optional field)
    # This relies on the instances constructed above having some None fields.
    # For example, _make_text() has no 'origin', so 'origin' should not be in data.
    if isinstance(instance, Text):
        assert "origin" not in data


def test_manual_serialization_custom_logic():
    """Test specific serialization logic for Manual (e.g. sorted pages)."""
    # Create pages in unsorted order
    page1 = Page(bbox=BBOX, pdf_page_number=1)
    page2 = Page(bbox=BBOX, pdf_page_number=2)

    manual = Manual(pages=[page2, page1])

    # Manual validator should sort them
    assert manual.pages[0].pdf_page_number == 1
    assert manual.pages[1].pdf_page_number == 2

    # Check serialization
    data = manual.to_dict()
    assert data["pages"][0]["pdf_page_number"] == 1


def test_polymorphic_deserialization():
    """Verify that a list of mixed types deserializes correctly."""
    # Create a Page with mixed elements (Step, Divider)
    page = _make_page()

    json_str = page.to_json()
    restored_page = Page.model_validate_json(json_str)

    assert len(restored_page.steps) == 1
    assert isinstance(restored_page.steps[0], Step)
    assert len(restored_page.dividers) == 1
    assert isinstance(restored_page.dividers[0], Divider)
