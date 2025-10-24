from unittest.mock import MagicMock, patch

from build_a_long.bounding_box_extractor.extractor import (
    extract_bounding_boxes,
)
from build_a_long.bounding_box_extractor.extractor.page_elements import (
    Drawing,
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
        assert "pages" in result
        assert len(result["pages"]) == 1
        elements = result["pages"][0]["elements"]
        assert len(elements) == 2
        assert isinstance(elements[0], Text)
        assert elements[0].content == "1"
        assert elements[0].bbox.x0 == 10.0 and elements[0].bbox.y0 == 20.0
        assert isinstance(elements[1], Drawing)

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
        assert "pages" in result
        assert len(result["pages"]) == 1
        elements = result["pages"][0]["elements"]
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
        assert "pages" in result
        assert len(result["pages"]) == 1
        elements = result["pages"][0]["elements"]
        assert len(elements) == 1
        assert isinstance(elements[0], Text)
        assert elements[0].content == "Build Step Instructions"
        assert elements[0].bbox.x0 == 10.0
