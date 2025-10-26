from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from build_a_long.pdf_extract.main import main

from build_a_long.pdf_extract.extractor.hierarchy import (
    ElementTree,
)
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_elements import (
    Text,
)
from build_a_long.pdf_extract.extractor import PageData


class TestMain:
    @patch("build_a_long.pdf_extract.main.pymupdf.open")
    @patch("build_a_long.pdf_extract.main.extract_bounding_boxes")
    @patch("build_a_long.pdf_extract.main.draw_and_save_bboxes")
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

        step_element = Text(bbox=BBox(10.0, 20.0, 30.0, 40.0), text="1")
        page_bbox = BBox(0.0, 0.0, 100.0, 100.0)

        page_data = PageData(
            page_number=1,
            elements=[step_element],
            bbox=page_bbox,
        )

        # extract_bounding_boxes now returns List[PageData]
        mock_extract_bounding_boxes.return_value = [page_data]

        # Mock the PDF document
        mock_page = MagicMock()
        mock_doc = MagicMock()
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_pymupdf_open.return_value = mock_doc

        # Run main
        result = main()

        # Assert success
        assert result == 0

        # Assert extract_bounding_boxes was called with the document
        mock_extract_bounding_boxes.assert_called_once()
        call_args = mock_extract_bounding_boxes.call_args
        assert call_args[0][0] == mock_doc  # First arg is the document

        # Assert JSON was written
        mock_file_open.assert_called_once()
        call_args = mock_file_open.call_args
        json_path = call_args[0][0]
        assert str(json_path).endswith("test.json")
        assert call_args[0][1] == "w"

        # Assert draw_and_save_bboxes was called with correct arguments
        mock_draw_and_save_bboxes.assert_called_once()
        draw_call_args = mock_draw_and_save_bboxes.call_args
        # page: pymupdf.Page, hierarchy: ElementTree, output_path: Path
        assert draw_call_args[0][0] == mock_page  # page object
        # hierarchy is now an ElementTree, check that it has roots

        hierarchy = draw_call_args[0][1]
        assert isinstance(hierarchy, ElementTree)
        assert isinstance(draw_call_args[0][2], Path)  # output_path
        assert draw_call_args[0][2].name == "page_001.png"

    @patch("pathlib.Path.exists")
    @patch("sys.argv", ["main.py", "/nonexistent/file.pdf"])
    def test_main_file_not_found(self, mock_exists):
        """Test that main returns error code when PDF does not exist."""
        mock_exists.return_value = False

        result = main()

        assert result == 2
