from unittest.mock import MagicMock

from build_a_long.pdf_extract.extractor import (
    extract_bounding_boxes,
)
from build_a_long.pdf_extract.extractor.extractor import (
    _extract_text_elements,
    _extract_image_elements,
    _extract_drawing_elements,
    _warn_unknown_block_types,
)
from build_a_long.pdf_extract.extractor.page_elements import (
    Drawing,
    Image,
    Text,
)


class TestBoundingBoxExtractor:
    def test_extract_bounding_boxes_basic(self):
        # Build a fake document with 1 page and simple rawdict content
        fake_page = MagicMock()
        fake_page.get_text.return_value = {
            "blocks": [
                {  # text block representing a numeric instruction number
                    "type": 0,
                    "bbox": [10, 20, 30, 40],
                    "lines": [
                        {
                            "spans": [
                                {"text": "1", "bbox": [10, 20, 30, 40]},
                            ]
                        }
                    ],
                },
                {  # image block
                    "type": 1,
                    "bbox": [50, 60, 150, 200],
                },
            ]
        }
        fake_page.get_drawings.return_value = []
        fake_page.rect = MagicMock(x0=0, y0=0, x1=612, y1=792)

        fake_doc = MagicMock()
        fake_doc.__len__.return_value = 1

        def _getitem(idx):
            assert idx == 0
            return fake_page

        fake_doc.__getitem__.side_effect = _getitem

        # Call the function with document instead of path
        result = extract_bounding_boxes(fake_doc)

        # Validate typed elements structure (no artificial Root element)
        assert len(result) == 1
        page_data = result[0]
        assert page_data.page_number == 1
        elements = page_data.elements
        # elements now only include actual content: 1 Text + 1 Image
        assert len(elements) == 2
        assert isinstance(elements[0], Text)
        assert elements[0].text == "1"
        assert elements[0].bbox.x0 == 10.0 and elements[0].bbox.y0 == 20.0
        assert isinstance(elements[1], Image)

    def test_extract_bounding_boxes_with_output(self):
        """Test that extractor does NOT write images/json (that's in main.py now)."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "bbox": [10, 20, 30, 40],
                    "lines": [
                        {
                            "spans": [
                                {"text": "1", "bbox": [10, 20, 30, 40]},
                            ]
                        }
                    ],
                }
            ]
        }
        mock_page.get_drawings.return_value = []
        mock_page.rect = MagicMock(x0=0, y0=0, x1=612, y1=792)

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page

        result = extract_bounding_boxes(mock_doc)

        # Typed elements exist (no Root)
        assert len(result) == 1
        page_data = result[0]
        elements = page_data.elements
        assert len(elements) == 1  # Only 1 Text element
        assert isinstance(elements[0], Text)
        assert elements[0].text == "1"

    def test_extract_text_elements(self):
        """Test that regular text is extracted as Text elements with content."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "bbox": [10, 20, 100, 40],
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Build Step Instructions",
                                    "bbox": [10, 20, 100, 40],
                                },
                            ]
                        }
                    ],
                }
            ]
        }
        mock_page.get_drawings.return_value = []
        mock_page.rect = MagicMock(x0=0, y0=0, x1=612, y1=792)

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page

        result = extract_bounding_boxes(mock_doc)

        # Validate text element (no Root)
        assert len(result) == 1
        page_data = result[0]
        elements = page_data.elements
        assert len(elements) == 1  # Only 1 Text element
        assert isinstance(elements[0], Text)
        assert elements[0].text == "Build Step Instructions"
        assert elements[0].bbox.x0 == 10.0


class TestExtractedMethods:
    """Tests for the smaller extracted helper methods."""

    def test_extract_text_elements(self):
        """Test _extract_text_elements extracts text from blocks."""
        from typing import Any

        blocks: list[Any] = [
            {
                "type": 0,
                "number": 1,
                "lines": [
                    {
                        "spans": [
                            {"text": "Hello", "bbox": [10.0, 20.0, 30.0, 40.0]},
                            {"text": "World", "bbox": [35.0, 20.0, 55.0, 40.0]},
                        ]
                    }
                ],
            },
            {
                "type": 1,  # image block, should be skipped
                "number": 2,
                "bbox": [100.0, 200.0, 150.0, 250.0],
            },
        ]

        result = _extract_text_elements(blocks)

        assert len(result) == 2
        assert isinstance(result[0], Text)
        assert result[0].text == "Hello"
        assert result[0].bbox.x0 == 10.0
        assert isinstance(result[1], Text)
        assert result[1].text == "World"
        assert result[1].bbox.x0 == 35.0

    def test_extract_image_elements(self):
        """Test _extract_image_elements extracts images from blocks."""
        from typing import Any

        blocks: list[Any] = [
            {
                "type": 0,  # text block, should be skipped
                "number": 1,
                "lines": [{"spans": [{"text": "Text", "bbox": [10, 20, 30, 40]}]}],
            },
            {
                "type": 1,
                "number": 2,
                "bbox": [100.0, 200.0, 150.0, 250.0],
            },
            {
                "type": 1,
                "number": 3,
                "bbox": [200.0, 300.0, 250.0, 350.0],
            },
        ]

        result = _extract_image_elements(blocks)

        assert len(result) == 2
        assert isinstance(result[0], Image)
        assert result[0].image_id == "image_2"
        assert result[0].bbox.x0 == 100.0
        assert isinstance(result[1], Image)
        assert result[1].image_id == "image_3"
        assert result[1].bbox.x0 == 200.0

    def test_extract_drawing_elements(self):
        """Test _extract_drawing_elements extracts drawings from page.get_drawings()."""
        # Create mock rectangle objects
        rect1 = MagicMock()
        rect1.x0 = 10.0
        rect1.y0 = 20.0
        rect1.x1 = 30.0
        rect1.y1 = 40.0

        rect2 = MagicMock()
        rect2.x0 = 50.0
        rect2.y0 = 60.0
        rect2.x1 = 70.0
        rect2.y1 = 80.0

        drawings = [
            {"rect": rect1},
            {"rect": rect2},
        ]

        result = _extract_drawing_elements(drawings)

        assert len(result) == 2
        assert isinstance(result[0], Drawing)
        assert result[0].bbox.x0 == 10.0
        assert result[0].bbox.y0 == 20.0
        assert isinstance(result[1], Drawing)
        assert result[1].bbox.x0 == 50.0
        assert result[1].bbox.y0 == 60.0

    def test_warn_unknown_block_types_valid(self):
        """Test _warn_unknown_block_types returns True for valid blocks."""
        blocks = [
            {"type": 0, "number": 1},
            {"type": 1, "number": 2},
            {"type": 0, "number": 3},
        ]

        result = _warn_unknown_block_types(blocks)

        assert result is True

    def test_warn_unknown_block_types_invalid(self):
        """Test _warn_unknown_block_types returns False and logs warning for unknown types."""
        blocks = [
            {"type": 0, "number": 1},
            {"type": 99, "number": 2},  # Unknown type
        ]

        result = _warn_unknown_block_types(blocks)

        assert result is False

    def test_extract_text_elements_empty_blocks(self):
        """Test _extract_text_elements handles empty blocks list."""
        result = _extract_text_elements([])
        assert result == []

    def test_extract_image_elements_empty_blocks(self):
        """Test _extract_image_elements handles empty blocks list."""
        result = _extract_image_elements([])
        assert result == []

    def test_extract_drawing_elements_empty_drawings(self):
        """Test _extract_drawing_elements handles empty drawings list."""
        result = _extract_drawing_elements([])
        assert result == []
