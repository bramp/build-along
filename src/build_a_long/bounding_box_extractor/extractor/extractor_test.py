from pathlib import Path
from unittest.mock import MagicMock, patch

from build_a_long.bounding_box_extractor.extractor.extractor import (
    extract_bounding_boxes,
)
from build_a_long.bounding_box_extractor.extractor.page_elements import (
    Drawing,
    StepNumber,
)


class TestBoundingBoxExtractor:
    @patch("build_a_long.bounding_box_extractor.extractor.extractor.fitz.open")
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

    @patch("build_a_long.bounding_box_extractor.extractor.extractor.fitz.open")
    @patch("build_a_long.bounding_box_extractor.drawing.drawing.Image.frombytes")
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
