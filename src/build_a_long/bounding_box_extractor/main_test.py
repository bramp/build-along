from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest

from build_a_long.bounding_box_extractor.main import (
    extract_bounding_boxes,
    parse_page_range,
)
from build_a_long.bounding_box_extractor.page_elements import StepNumber, Drawing


class TestBoundingBoxExtractor:
    @patch("build_a_long.bounding_box_extractor.main.fitz.open")
    def test_extract_bounding_boxes_basic(self, mock_fitz_open):
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
                                {"text": "1"},
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

        fake_doc = MagicMock()
        fake_doc.__len__.return_value = 1

        def _getitem(idx):
            assert idx == 0
            return fake_page

        fake_doc.__getitem__.side_effect = _getitem
        mock_fitz_open.return_value = fake_doc

        # Call the function
        result = extract_bounding_boxes(dummy_pdf_path, output_dir=None)

        # Validate typed elements structure
        assert "pages" in result
        assert len(result["pages"]) == 1
        elements = result["pages"][0]["elements"]
        assert len(elements) == 2
        assert isinstance(elements[0], StepNumber)
        assert elements[0].bbox.x0 == 10.0 and elements[0].bbox.y0 == 20.0
        assert isinstance(elements[1], Drawing)

    @patch("build_a_long.bounding_box_extractor.main.fitz.open")
    @patch("build_a_long.bounding_box_extractor.main.Image.frombytes")
    @patch("pathlib.Path.mkdir")
    def test_extract_bounding_boxes_with_image_output(
        self,
        mock_path_mkdir,
        mock_image_frombytes,
        mock_fitz_open,
    ):
        dummy_pdf_path = "/path/to/dummy.pdf"
        dummy_output_dir = Path("/tmp/output")

        # Mock the pixmap and image objects
        mock_pixmap = MagicMock()
        mock_pixmap.width = 100
        mock_pixmap.height = 100
        mock_pixmap.samples = b"dummy_samples"

        mock_image = MagicMock()
        mock_image_frombytes.return_value = mock_image

        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "bbox": [10, 20, 30, 40],
                    "lines": [
                        {
                            "spans": [
                                {"text": "1"},
                            ]
                        }
                    ],
                }
            ]
        }

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        result = extract_bounding_boxes(dummy_pdf_path, output_dir=dummy_output_dir)

        # Assert that image saving was attempted
        mock_image.save.assert_called_once_with(dummy_output_dir / "page_001.png")

        # Ensure that the output directory was created
        mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Typed elements exist
        assert "pages" in result
        assert len(result["pages"]) == 1
        elements = result["pages"][0]["elements"]
        assert len(elements) == 1
        assert isinstance(elements[0], StepNumber)


class TestParsePageRange:
    """Test parse_page_range() function with various input formats."""

    def test_single_page(self):
        """Test parsing a single page number."""
        assert parse_page_range("5") == (5, 5)
        assert parse_page_range("1") == (1, 1)
        assert parse_page_range("100") == (100, 100)

    def test_single_page_with_whitespace(self):
        """Test parsing a single page with leading/trailing whitespace."""
        assert parse_page_range("  5  ") == (5, 5)
        assert parse_page_range("\t10\n") == (10, 10)

    def test_explicit_range(self):
        """Test parsing an explicit page range (e.g., '5-10')."""
        assert parse_page_range("5-10") == (5, 10)
        assert parse_page_range("1-3") == (1, 3)
        assert parse_page_range("10-100") == (10, 100)

    def test_explicit_range_with_whitespace(self):
        """Test parsing ranges with whitespace around numbers."""
        assert parse_page_range(" 5 - 10 ") == (5, 10)
        assert parse_page_range("1-  3") == (1, 3)

    def test_same_start_and_end(self):
        """Test range where start equals end."""
        assert parse_page_range("5-5") == (5, 5)

    def test_open_end_range(self):
        """Test 'from page X to end' format (e.g., '10-')."""
        assert parse_page_range("10-") == (10, None)
        assert parse_page_range("1-") == (1, None)
        assert parse_page_range("  5  -  ") == (5, None)

    def test_open_start_range(self):
        """Test 'from start to page X' format (e.g., '-5')."""
        assert parse_page_range("-5") == (None, 5)
        assert parse_page_range("-1") == (None, 1)
        assert parse_page_range("  -  10  ") == (None, 10)

    def test_invalid_empty_string(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Page range cannot be empty"):
            parse_page_range("")
        with pytest.raises(ValueError, match="Page range cannot be empty"):
            parse_page_range("   ")

    def test_invalid_double_dash(self):
        """Test that '-' alone raises ValueError."""
        with pytest.raises(ValueError, match="At least one page number required"):
            parse_page_range("-")

    def test_invalid_non_numeric(self):
        """Test that non-numeric input raises ValueError."""
        with pytest.raises(ValueError, match="Invalid page number"):
            parse_page_range("abc")
        with pytest.raises(ValueError, match="Invalid end page number"):
            parse_page_range("5-abc")
        with pytest.raises(ValueError, match="Invalid start page number"):
            parse_page_range("abc-10")
        with pytest.raises(ValueError, match="Invalid end page number"):
            parse_page_range("-abc")

    def test_invalid_negative_numbers(self):
        """Test that negative page numbers raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_page_range("0")
        with pytest.raises(ValueError, match="must be >= 1"):
            # Double dash creates empty start_str and "-1" as end_str
            parse_page_range("--1")
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_page_range("5-0")
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_page_range("0-5")
        with pytest.raises(ValueError, match="must be >= 1"):
            parse_page_range("-0")

    def test_invalid_start_greater_than_end(self):
        """Test that start > end raises ValueError."""
        with pytest.raises(ValueError, match="cannot be greater than end page"):
            parse_page_range("10-5")
        with pytest.raises(ValueError, match="cannot be greater than end page"):
            parse_page_range("100-1")

    def test_invalid_float(self):
        """Test that float numbers raise ValueError."""
        with pytest.raises(ValueError, match="Invalid page"):
            parse_page_range("5.5")
        with pytest.raises(ValueError, match="Invalid end page number"):
            parse_page_range("5-10.5")
