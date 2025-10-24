from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from build_a_long.bounding_box_extractor.main import main


class TestMain:
    @patch("build_a_long.bounding_box_extractor.main.pymupdf.open")
    @patch("build_a_long.bounding_box_extractor.main.extract_bounding_boxes")
    @patch("build_a_long.bounding_box_extractor.main.draw_and_save_bboxes")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    @patch("sys.argv", ["main.py", "/path/to/test.pdf"])
    def test_main_writes_json_and_images(
        self,
        mock_file_open,
        mock_mkdir,
        mock_exists,
        mock_draw_and_save_bboxes,
        mock_extract_bounding_boxes,
        mock_pymupdf_open,
    ):
        """Test that main.py writes JSON and PNG files using extracted data."""
        mock_exists.return_value = True

        # Mock the extractor to return structured data
        from build_a_long.bounding_box_extractor.extractor.bbox import BBox
        from build_a_long.bounding_box_extractor.extractor.hierarchy import (
            ElementNode,
        )
        from build_a_long.bounding_box_extractor.extractor.page_elements import (
            StepNumber,
        )

        step_element = StepNumber(bbox=BBox(10.0, 20.0, 30.0, 40.0), value=1)
        root_node = ElementNode(element=step_element, children=())

        mock_extract_bounding_boxes.return_value = {
            "pages": [
                {
                    "page_number": 1,
                    "elements": [step_element],
                    "hierarchy": [root_node],
                }
            ]
        }

        # Mock the PDF document
        mock_page = MagicMock()
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_pymupdf_open.return_value = mock_doc

        # Run main
        result = main()

        # Assert success
        assert result == 0

        # Assert JSON was written
        mock_file_open.assert_called_once()
        call_args = mock_file_open.call_args
        json_path = call_args[0][0]
        assert str(json_path).endswith("test.json")
        assert call_args[0][1] == "w"

        # Assert draw_and_save_bboxes was called
        mock_draw_and_save_bboxes.assert_called_once()
        draw_call_args = mock_draw_and_save_bboxes.call_args
        assert draw_call_args[0][0] == mock_page
        assert draw_call_args[0][2] == Path("/path/to")
        assert draw_call_args[0][3] == 1

    @patch("pathlib.Path.exists")
    @patch("sys.argv", ["main.py", "/nonexistent/file.pdf"])
    def test_main_file_not_found(self, mock_exists):
        """Test that main returns error code when PDF does not exist."""
        mock_exists.return_value = False

        result = main()

        assert result == 2
