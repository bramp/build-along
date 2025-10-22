from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

from build_a_long.bounding_box_extractor.main import extract_bounding_boxes


class TestBoundingBoxExtractor:
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("build_a_long.bounding_box_extractor.main.fitz.open")
    def test_extract_bounding_boxes_basic(
        self, mock_fitz_open, mock_json_dump, mock_file_open
    ):
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
        # __getitem__ for index 0 returns our fake page

        def _getitem(idx):
            assert idx == 0
            return fake_page

        fake_doc.__getitem__.side_effect = _getitem
        mock_fitz_open.return_value = fake_doc

        # Call the function
        extract_bounding_boxes(dummy_pdf_path, output_dir=None)

        # Assert that json.dump was called with the expected structure
        mock_json_dump.assert_called_once()
        args, kwargs = mock_json_dump.call_args
        extracted_data = args[0]

        assert "pages" in extracted_data
        assert len(extracted_data["pages"]) == 1
        elements = extracted_data["pages"][0]["elements"]
        # Expect two elements: a text classified as instruction_number and an image
        assert len(elements) == 2
        assert elements[0]["type"] == "instruction_number"
        assert elements[0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
        assert elements[1]["type"] == "image"

        # Assert that the output file was attempted to be opened with 'w'
        expected_output_filename = dummy_pdf_path.replace(".pdf", ".json")
        mock_file_open.assert_called_once_with(expected_output_filename, "w")

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("build_a_long.bounding_box_extractor.main.fitz.open")
    @patch("build_a_long.bounding_box_extractor.main.Image.frombytes")
    @patch("pathlib.Path.mkdir")
    def test_extract_bounding_boxes_with_image_output(
        self,
        mock_path_mkdir,
        mock_image_frombytes,
        mock_fitz_open,
        mock_json_dump,
        mock_file_open,
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

        extract_bounding_boxes(dummy_pdf_path, output_dir=dummy_output_dir)

        # Assert that image saving was attempted
        mock_image.save.assert_called_once_with(dummy_output_dir / "page_001.png")

        # Ensure that the output directory was created
        mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=True)
