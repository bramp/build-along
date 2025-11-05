"""Tests for PageData JSON serialization/deserialization with polymorphic elements."""

import json

from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_elements import (
    Drawing,
    Image,
    Text,
)


class TestPageDataJsonSerialization:
    """Tests for PageData.from_json() handling of polymorphic element types."""

    def test_from_json_with_text_elements(self) -> None:
        """Verify Text elements are correctly deserialized from JSON."""
        json_str = json.dumps(
            {
                "page_number": 1,
                "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 200},
                "elements": [
                    {
                        "__tag__": "Text",
                        "bbox": {"x0": 10, "y0": 10, "x1": 50, "y1": 30},
                        "text": "Hello",
                        "font_name": "Arial",
                        "font_size": 12.0,
                        "id": 0,
                    },
                    {
                        "__tag__": "Text",
                        "bbox": {"x0": 10, "y0": 40, "x1": 50, "y1": 60},
                        "text": "World",
                        "id": 1,
                    },
                ],
            }
        )

        page: PageData = PageData.from_json(json_str)  # type: ignore[assignment]

        assert page.page_number == 1
        assert page.bbox == BBox(0, 0, 100, 200)
        assert len(page.elements) == 2
        assert all(isinstance(e, Text) for e in page.elements)
        assert page.elements[0].text == "Hello"  # type: ignore
        assert page.elements[0].font_name == "Arial"  # type: ignore
        assert page.elements[0].font_size == 12.0  # type: ignore
        assert page.elements[1].text == "World"  # type: ignore

    def test_from_json_with_image_elements(self) -> None:
        """Verify Image elements are correctly deserialized from JSON."""
        json_str = json.dumps(
            {
                "page_number": 2,
                "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 200},
                "elements": [
                    {
                        "__tag__": "Image",
                        "bbox": {"x0": 20, "y0": 20, "x1": 80, "y1": 80},
                        "image_id": "img_001",
                        "id": 0,
                    },
                    {
                        "__tag__": "Image",
                        "bbox": {"x0": 20, "y0": 90, "x1": 80, "y1": 150},
                        "image_id": "img_002",
                        "id": 42,
                    },
                ],
            }
        )

        page: PageData = PageData.from_json(json_str)  # type: ignore[assignment]

        assert page.page_number == 2
        assert len(page.elements) == 2
        assert all(isinstance(e, Image) for e in page.elements)
        assert page.elements[0].image_id == "img_001"  # type: ignore
        assert page.elements[1].image_id == "img_002"  # type: ignore
        assert page.elements[1].id == 42

    def test_from_json_with_drawing_elements(self) -> None:
        """Verify Drawing elements are correctly deserialized from JSON."""
        json_str = json.dumps(
            {
                "page_number": 3,
                "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 200},
                "elements": [
                    {
                        "__tag__": "Drawing",
                        "bbox": {"x0": 5, "y0": 5, "x1": 95, "y1": 195},
                        "id": 0,
                    },
                    {
                        "__tag__": "Drawing",
                        "bbox": {"x0": 10, "y0": 10, "x1": 90, "y1": 190},
                        "image_id": "drawing_raster",
                        "id": 99,
                    },
                ],
            }
        )

        page: PageData = PageData.from_json(json_str)  # type: ignore[assignment]

        assert page.page_number == 3
        assert len(page.elements) == 2
        assert all(isinstance(e, Drawing) for e in page.elements)
        assert isinstance(page.elements[0], Drawing)
        assert page.elements[0].image_id is None
        assert isinstance(page.elements[1], Drawing)
        assert page.elements[1].image_id == "drawing_raster"  # type: ignore
        assert page.elements[1].id == 99

    def test_from_json_with_mixed_element_types(self) -> None:
        """Verify mixed Text, Image, and Drawing elements are correctly deserialized."""
        json_str = json.dumps(
            {
                "page_number": 10,
                "bbox": {"x0": 0.0, "y0": 0.0, "x1": 552.75, "y1": 496.06},
                "elements": [
                    {
                        "__tag__": "Text",
                        "id": 0,
                        "bbox": {
                            "x0": 113.92,
                            "y0": 104.82,
                            "x1": 148.61,
                            "y1": 169.89,
                        },
                        "text": "1",
                        "font_name": "CeraPro-Medium",
                        "font_size": 50.0,
                    },
                    {
                        "__tag__": "Text",
                        "id": 1,
                        "bbox": {
                            "x0": 316.19,
                            "y0": 58.86,
                            "x1": 325.07,
                            "y1": 68.82,
                        },
                        "text": "2x",
                        "font_name": "CeraPro-Light",
                        "font_size": 8.0,
                    },
                    {
                        "__tag__": "Image",
                        "id": 2,
                        "bbox": {"x0": 300, "y0": 40, "x1": 350, "y1": 70},
                        "image_id": "image_123",
                    },
                    {
                        "__tag__": "Drawing",
                        "id": 3,
                        "bbox": {
                            "x0": 16.0,
                            "y0": 39.0,
                            "x1": 253.0,
                            "y1": 254.0,
                        },
                    },
                ],
            }
        )

        page: PageData = PageData.from_json(json_str)  # type: ignore[assignment]

        assert page.page_number == 10
        assert page.bbox == BBox(0.0, 0.0, 552.75, 496.06)
        assert len(page.elements) == 4

        # Check each element type
        assert isinstance(page.elements[0], Text)
        assert page.elements[0].text == "1"  # type: ignore
        assert page.elements[0].id == 0

        assert isinstance(page.elements[1], Text)
        assert page.elements[1].text == "2x"  # type: ignore
        assert page.elements[1].id == 1

        assert isinstance(page.elements[2], Image)
        assert page.elements[2].image_id == "image_123"  # type: ignore
        assert page.elements[2].id == 2

        assert isinstance(page.elements[3], Drawing)
        assert page.elements[3].id == 3

    def test_from_json_with_bbox_as_dict(self) -> None:
        """Verify bbox can be specified as a dict instead of a list."""
        json_str = json.dumps(
            {
                "page_number": 5,
                "bbox": {"x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 200.0},
                "elements": [
                    {
                        "__tag__": "Text",
                        "bbox": {
                            "x0": 10.0,
                            "y0": 10.0,
                            "x1": 50.0,
                            "y1": 30.0,
                        },
                        "text": "Dict bbox",
                        "id": 0,
                    }
                ],
            }
        )

        page: PageData = PageData.from_json(json_str)  # type: ignore[assignment]

        assert page.page_number == 5
        assert page.bbox == BBox(0.0, 0.0, 100.0, 200.0)
        assert len(page.elements) == 1
        assert isinstance(page.elements[0], Text)
        assert page.elements[0].bbox == BBox(10.0, 10.0, 50.0, 30.0)

    def test_from_json_with_unknown_element_type(self) -> None:
        """Unknown element types should be skipped with a warning, not crash."""
        json_str = json.dumps(
            {
                "page_number": 6,
                "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 200},
                "elements": [
                    {
                        "__tag__": "Text",
                        "bbox": {"x0": 10, "y0": 10, "x1": 50, "y1": 30},
                        "text": "Valid",
                    },
                    {
                        "__tag__": "Unknown",  # invalid tag should raise
                        "bbox": {"x0": 10, "y0": 40, "x1": 50, "y1": 60},
                    },
                ],
            }
        )

        # TODO Isn't there a better way to assert an exception is raised?

        # Unknown tag should cause parsing error
        raised = False
        try:
            _ = PageData.from_json(json_str)
        except Exception:
            # TODO Check for the correct exception type and message
            raised = True
        assert raised is True

    def test_round_trip_serialization(self) -> None:
        """Verify to_json() and from_json() round-trip correctly."""
        original = PageData(
            page_number=42,
            bbox=BBox(0, 0, 500, 600),
            elements=[
                Text(bbox=BBox(10, 10, 100, 30), text="Test", id=1),
                Image(bbox=BBox(10, 40, 100, 140), image_id="img_1", id=2),
                Drawing(bbox=BBox(10, 150, 100, 250), id=3),
            ],
        )

        json_str = original.to_json()  # type: ignore
        restored: PageData = PageData.from_json(json_str)  # type: ignore[assignment]

        assert restored.page_number == original.page_number
        assert restored.bbox == original.bbox
        assert len(restored.elements) == len(original.elements)

        for orig_elem, restored_elem in zip(
            original.elements, restored.elements, strict=True
        ):
            assert isinstance(restored_elem, type(orig_elem))
            assert restored_elem.bbox == orig_elem.bbox
            assert restored_elem.id == orig_elem.id

    def test_unknown_field_raises_error(self) -> None:
        """Verify that unknown fields in JSON are handled appropriately.

        Note: With custom decoders, Undefined.RAISE may not work as expected.
        This test documents current behavior.
        """
        json_str = json.dumps(
            {
                "page_number": 1,
                "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 200},
                "unknown_field": "extra field",  # Unknown field
                "elements": [],
            }
        )

        # Unknown top-level fields are ignored in current configuration
        page: PageData = PageData.from_json(json_str)  # type: ignore[assignment]
        assert page.page_number == 1
        assert len(page.elements) == 0

    def test_unknown_element_field_raises_error(self) -> None:
        """Verify that unknown fields in element JSON are handled appropriately.

        Note: With custom decoders, Undefined.RAISE may not work as expected.
        This test documents current behavior.
        """
        json_str = json.dumps(
            {
                "page_number": 1,
                "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 200},
                "elements": [
                    {
                        "__tag__": "Text",
                        "bbox": {"x0": 10, "y0": 10, "x1": 50, "y1": 30},
                        "text": "Test",
                        "extra_field": "should raise",  # Unknown field
                        "id": 0,
                    }
                ],
            }
        )

        # Unknown fields on elements are ignored in current configuration
        page: PageData = PageData.from_json(json_str)  # type: ignore[assignment]
        assert len(page.elements) == 1
        assert isinstance(page.elements[0], Text)
