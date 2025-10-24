from unittest.mock import MagicMock, patch

from build_a_long.bounding_box_extractor.extractor import (
    extract_bounding_boxes,
)
from build_a_long.bounding_box_extractor.extractor.extractor import (
    _extract_text_elements,
    _extract_image_elements,
    _extract_drawing_elements,
    _create_root_element,
    _warn_unknown_block_types,
)
from build_a_long.bounding_box_extractor.extractor.page_elements import (
    Drawing,
    Image,
    Root,
    Text,
)


class TestBoundingBoxExtractor:
    @patch("build_a_long.bounding_box_extractor.extractor.extractor.pymupdf.open")
    def test_extract_bounding_boxes_basic(self, mock_pymupdf_open):
        # Create a dummy PDF path for testing
        dummy_pdf_path = "/path/to/dummy.pdf"

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

        fake_doc = MagicMock()
        fake_doc.__len__.return_value = 1

        def _getitem(idx):
            assert idx == 0
            return fake_page

        fake_doc.__getitem__.side_effect = _getitem
        mock_pymupdf_open.return_value = fake_doc

        # Call the function (no output_dir parameter after refactor)
        result = extract_bounding_boxes(dummy_pdf_path)

        # Validate typed elements structure
        assert len(result.pages) == 1
        page_data = result.pages[0]
        assert page_data.page_number == 1
        elements = page_data.elements
        assert len(elements) == 2
        assert isinstance(elements[0], Text)
        assert elements[0].content == "1"
        assert elements[0].bbox.x0 == 10.0 and elements[0].bbox.y0 == 20.0
        assert isinstance(elements[1], Image)

    @patch("build_a_long.bounding_box_extractor.extractor.extractor.pymupdf.open")
    def test_extract_bounding_boxes_with_output(
        self,
        mock_pymupdf_open,
    ):
        """Test that extractor does NOT write images/json (that's in main.py now)."""
        dummy_pdf_path = "/path/to/dummy.pdf"

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

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_pymupdf_open.return_value = mock_doc

        result = extract_bounding_boxes(dummy_pdf_path)

        # Typed elements exist
        assert len(result.pages) == 1
        page_data = result.pages[0]
        elements = page_data.elements
        assert len(elements) == 1
        assert isinstance(elements[0], Text)
        assert elements[0].content == "1"

    @patch("build_a_long.bounding_box_extractor.extractor.extractor.pymupdf.open")
    def test_extract_text_elements(self, mock_pymupdf_open):
        """Test that regular text is extracted as Text elements with content."""
        dummy_pdf_path = "/path/to/dummy.pdf"

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

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_pymupdf_open.return_value = mock_doc

        result = extract_bounding_boxes(dummy_pdf_path)

        # Validate text element
        assert len(result.pages) == 1
        page_data = result.pages[0]
        elements = page_data.elements
        assert len(elements) == 1
        assert isinstance(elements[0], Text)
        assert elements[0].content == "Build Step Instructions"
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
        assert result[0].content == "Hello"
        assert result[0].bbox.x0 == 10.0
        assert isinstance(result[1], Text)
        assert result[1].content == "World"
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

    def test_create_root_element(self):
        """Test _create_root_element creates Root with page bounds."""
        mock_page = MagicMock()
        mock_rect = MagicMock()
        mock_rect.x0 = 0.0
        mock_rect.y0 = 0.0
        mock_rect.x1 = 612.0  # Letter size width
        mock_rect.y1 = 792.0  # Letter size height
        mock_page.rect = mock_rect

        result = _create_root_element(mock_page)

        assert isinstance(result, Root)
        assert result.bbox.x0 == 0.0
        assert result.bbox.y0 == 0.0
        assert result.bbox.x1 == 612.0
        assert result.bbox.y1 == 792.0

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
