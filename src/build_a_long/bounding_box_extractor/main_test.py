from unittest.mock import patch, mock_open, MagicMock

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
        extract_bounding_boxes(dummy_pdf_path)

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
