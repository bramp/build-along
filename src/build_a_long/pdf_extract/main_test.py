from pathlib import Path
from unittest.mock import ANY, MagicMock, mock_open, patch

from build_a_long.pdf_extract.main import main

from build_a_long.pdf_extract.extractor.hierarchy import (
    ElementTree,
)
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_elements import (
    Text,
    Image,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.main import save_raw_json
import json


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
        mock_draw_and_save_bboxes.assert_called_once_with(
            mock_page, ANY, ANY, draw_deleted=False
        )
        draw_call_args = mock_draw_and_save_bboxes.call_args
        assert isinstance(draw_call_args.args[1], ElementTree)
        assert isinstance(draw_call_args.args[2], Path)
        assert draw_call_args.args[2].name == "page_001.png"

    @patch("pathlib.Path.exists")
    @patch("sys.argv", ["main.py", "/nonexistent/file.pdf"])
    def test_main_file_not_found(self, mock_exists):
        """Test that main returns error code when PDF does not exist."""
        mock_exists.return_value = False

        result = main()

        assert result == 2


def test_save_raw_json_prunes_fields(tmp_path: Path) -> None:
    """save_raw_json should exclude deleted=False, label=None, and empty label_scores."""
    # Element with defaults that should be pruned
    text = Text(
        bbox=BBox(0.0, 0.0, 10.0, 10.0),
        text="hello",
    )
    # Element with explicit values that must be preserved
    img = Image(
        bbox=BBox(1.0, 1.0, 5.0, 5.0),
        image_id="img1",
        label="page_number",
        label_scores={"page_number": 0.9},
        deleted=True,
    )

    page = PageData(
        page_number=1, elements=[text, img], bbox=BBox(0.0, 0.0, 100.0, 100.0)
    )

    pdf_path = tmp_path / "my.pdf"
    pdf_path.write_bytes(b"")

    save_raw_json([page], tmp_path, pdf_path)

    out = tmp_path / "page_001_raw.json"
    assert out.exists()
    data = json.loads(out.read_text())

    assert data["page_number"] == 1
    assert isinstance(data["elements"], list) and len(data["elements"]) == 2

    e0 = data["elements"][0]
    assert "deleted" not in e0
    assert "label" not in e0
    assert "label_scores" not in e0

    e1 = data["elements"][1]
    assert e1.get("deleted") is True
    assert e1.get("label") == "page_number"
    assert e1.get("label_scores") == {"page_number": 0.9}
