"""Tests for main.py using minimal mocking and real temp files."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.cli import ProcessingConfig
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Text
from build_a_long.pdf_extract.main import (
    _load_json_pages,
    _parse_page_selection,
    _process_json,
    _process_pdf,
    _validate_pdf_path,
    main,
)
from build_a_long.pdf_extract.parser.page_ranges import PageRanges


def _make_config(**overrides) -> ProcessingConfig:
    """Helper to create ProcessingConfig with defaults."""
    defaults = {
        "pdf_paths": [],
        "output_dir": None,
        "include_types": {"text", "image", "drawing"},
        "page_ranges": None,
        "save_debug_json": False,
        "compress_json": False,
        "save_summary": False,
        "summary_detailed": False,
        "print_histogram": False,
        "print_font_hints": False,
        "debug_classification": False,
        "debug_candidates": False,
        "debug_candidates_label": None,
        "draw_blocks": False,
        "draw_elements": False,
        "draw_deleted": False,
    }
    defaults.update(overrides)
    return ProcessingConfig(**defaults)


class TestValidatePdfPath:
    """Test _validate_pdf_path function."""

    def test_exists(self, tmp_path):
        """Test path validation with existing file."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        assert _validate_pdf_path(pdf_file) is True

    def test_not_exists(self):
        """Test path validation with non-existent file."""
        assert _validate_pdf_path(Path("/nonexistent/file.pdf")) is False


class TestLoadJsonPages:
    """Test _load_json_pages function."""

    def test_valid(self, tmp_path):
        """Test loading valid JSON extraction data."""
        extraction_data = {
            "pages": [
                {
                    "page_number": 1,
                    "blocks": [],
                    "bbox": {"x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 100.0},
                },
                {
                    "page_number": 2,
                    "blocks": [],
                    "bbox": {"x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 100.0},
                },
            ]
        }

        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(extraction_data))

        pages = _load_json_pages(json_file)
        assert len(pages) == 2
        assert pages[0].page_number == 1
        assert pages[1].page_number == 2

    def test_empty(self, tmp_path):
        """Test loading JSON with no pages."""
        extraction_data = {"pages": []}

        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(extraction_data))

        pages = _load_json_pages(json_file)
        assert pages == []


class TestParsePageSelection:
    """Test _parse_page_selection function."""

    def test_all(self):
        """Test parsing 'all pages' selection."""
        result = _parse_page_selection(None, 100)
        assert result is not None
        assert result == PageRanges.all()
        assert list(result.page_numbers(100)) == list(range(1, 101))

    def test_range(self):
        """Test parsing page range."""
        result = _parse_page_selection("5-10", 100)
        assert result is not None
        assert list(result.page_numbers(100)) == [5, 6, 7, 8, 9, 10]

    def test_multiple_segments(self):
        """Test parsing comma-separated page ranges."""
        result = _parse_page_selection("1-3,5,10-12", 100)
        assert result is not None
        assert list(result.page_numbers(100)) == [1, 2, 3, 5, 10, 11, 12]

    def test_overlapping_segments(self):
        """Test parsing overlapping comma-separated page ranges."""
        result = _parse_page_selection("1-5,2,4-6", 6)
        assert result is not None
        assert list(result.page_numbers(6)) == [1, 2, 3, 4, 5, 6]

    def test_invalid(self):
        """Test parsing invalid page range."""
        result = _parse_page_selection("invalid", 100)
        assert result is None


class TestProcessJson:
    """Test JSON processing using approach #2 (real temp files)."""

    def test_process_json_success(self, tmp_path):
        """Test successful JSON processing."""
        extraction_data = {
            "pages": [
                {
                    "page_number": 1,
                    "blocks": [],
                    "bbox": {"x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 100.0},
                }
            ]
        }

        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(extraction_data))

        config = _make_config(pdf_paths=[json_file])

        result = _process_json(config, json_file)
        assert result == 0

    def test_process_json_empty_pages(self, tmp_path):
        """Test JSON processing with empty pages."""
        extraction_data = {"pages": []}

        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(extraction_data))

        config = _make_config(pdf_paths=[json_file])

        result = _process_json(config, json_file)
        assert result == 2


class TestProcessPdf:
    """Test PDF processing with minimal mocking (approach #1 + #3)."""

    @patch("build_a_long.pdf_extract.main.pymupdf.open")
    @patch("build_a_long.pdf_extract.main.extract_bounding_boxes")
    def test_process_pdf_basic(
        self, mock_extract_bounding_boxes, mock_pymupdf_open, tmp_path
    ):
        """Test basic PDF processing with real temp directory."""
        # Create test data
        page_data = PageData(
            page_number=1,
            blocks=[Text(id=0, bbox=BBox(10.0, 20.0, 30.0, 40.0), text="1")],
            bbox=BBox(0.0, 0.0, 100.0, 100.0),
        )
        mock_extract_bounding_boxes.return_value = [page_data]

        # Mock PDF document
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_pymupdf_open.return_value = mock_doc

        pdf_path = Path("/fake/test.pdf")
        config = _make_config(
            pdf_paths=[pdf_path],
            output_dir=tmp_path,
            save_debug_json=True,
        )

        result = _process_pdf(config, pdf_path, tmp_path)

        assert result == 0
        # Verify JSON file was created
        json_file = tmp_path / "test_raw.json"
        assert json_file.exists()

        # Verify content
        saved_data = json.loads(json_file.read_text())
        assert len(saved_data["pages"]) == 1
        assert saved_data["pages"][0]["page_number"] == 1

    @patch("build_a_long.pdf_extract.main.pymupdf.open")
    @patch("build_a_long.pdf_extract.main.extract_bounding_boxes")
    def test_process_pdf_filters_pages(
        self, mock_extract_bounding_boxes, mock_pymupdf_open, tmp_path
    ):
        """Test that --pages argument filters output correctly."""

        def _mk_page(n: int) -> PageData:
            return PageData(page_number=n, blocks=[], bbox=BBox(0.0, 0.0, 1.0, 1.0))

        # All pages extracted for font hints
        all_pages = [_mk_page(i) for i in range(1, 201)]
        mock_extract_bounding_boxes.return_value = all_pages

        # Mock PDF document
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 200
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_pymupdf_open.return_value = mock_doc

        pdf_path = Path("/fake/test.pdf")
        config = _make_config(
            pdf_paths=[pdf_path],
            output_dir=tmp_path,
            page_ranges="10-12,15",  # Filter to specific pages
            save_debug_json=True,
        )

        result = _process_pdf(config, pdf_path, tmp_path)

        assert result == 0

        # Verify all pages were extracted for font hints
        assert mock_extract_bounding_boxes.call_count == 1
        pages_arg = mock_extract_bounding_boxes.call_args[0][1]
        assert pages_arg == list(range(1, 201))

        # Verify only filtered pages in output (per-page files when filtered)
        # When specific pages are selected, raw JSON is saved per-page
        page_files = sorted(tmp_path.glob("test_page_*_raw.json"))
        assert len(page_files) == 4
        saved_page_numbers = []
        for page_file in page_files:
            page_data = json.loads(page_file.read_text())
            saved_page_numbers.append(page_data["pages"][0]["page_number"])
        assert saved_page_numbers == [10, 11, 12, 15]

    @patch("build_a_long.pdf_extract.main.pymupdf.open")
    @patch("build_a_long.pdf_extract.main.extract_bounding_boxes")
    @patch("build_a_long.pdf_extract.cli.io.draw_and_save_bboxes")
    def test_process_pdf_with_drawing(
        self,
        mock_draw_and_save_bboxes,
        mock_extract_bounding_boxes,
        mock_pymupdf_open,
        tmp_path,
    ):
        """Test PDF processing with image drawing enabled."""
        page_data = PageData(
            page_number=1,
            blocks=[],
            bbox=BBox(0.0, 0.0, 100.0, 100.0),
        )
        mock_extract_bounding_boxes.return_value = [page_data]

        # Mock PDF document with page
        mock_page = MagicMock()
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_pymupdf_open.return_value = mock_doc

        pdf_path = Path("/fake/test.pdf")
        config = _make_config(
            pdf_paths=[pdf_path],
            output_dir=tmp_path,
            draw_blocks=True,
            draw_elements=True,
        )

        result = _process_pdf(config, pdf_path, tmp_path)

        assert result == 0
        # Verify draw function was called
        mock_draw_and_save_bboxes.assert_called_once()
        call_args = mock_draw_and_save_bboxes.call_args
        assert call_args.kwargs["draw_blocks"] is True
        assert call_args.kwargs["draw_elements"] is True
        assert call_args.kwargs["draw_deleted"] is False
        assert isinstance(call_args.args[1], ClassificationResult)


class TestMainIntegration:
    """Integration tests for main() entry point (approach #2)."""

    def test_main_with_json_file(self, tmp_path):
        """Test main() with JSON file input."""
        extraction_data = {
            "pages": [
                {
                    "page_number": 1,
                    "blocks": [],
                    "bbox": {"x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 100.0},
                }
            ]
        }

        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(extraction_data))

        with patch("sys.argv", ["main.py", str(json_file)]):
            result = main()
        assert result == 0

    def test_main_file_not_found(self):
        """Test main() with non-existent file."""
        with patch("sys.argv", ["main.py", "/nonexistent/file.pdf"]):
            result = main()
        assert result == 2

    def test_main_invalid_json(self, tmp_path):
        """Test main() with malformed JSON."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("invalid json{")

        with patch("sys.argv", ["main.py", str(json_file)]):
            # Invalid JSON causes pydantic.ValidationError - expected behavior
            try:
                main()
                raise AssertionError("Expected exception for invalid JSON")
            except Exception as e:
                # Expected - pydantic validation fails on malformed JSON
                # Check it's not our AssertionError
                if isinstance(e, AssertionError):
                    raise

    def test_main_empty_json_pages(self, tmp_path):
        """Test main() with JSON containing no pages."""
        extraction_data = {"pages": []}

        json_file = tmp_path / "empty.json"
        json_file.write_text(json.dumps(extraction_data))

        with patch("sys.argv", ["main.py", str(json_file)]):
            result = main()
        assert result == 2

    @patch("build_a_long.pdf_extract.main.pymupdf.open")
    @patch("build_a_long.pdf_extract.main.extract_bounding_boxes")
    def test_main_with_pdf_and_output_dir(
        self, mock_extract_bounding_boxes, mock_pymupdf_open, tmp_path
    ):
        """Test main() with PDF file and custom output directory."""
        page_data = PageData(
            page_number=1,
            blocks=[],
            bbox=BBox(0.0, 0.0, 100.0, 100.0),
        )
        mock_extract_bounding_boxes.return_value = [page_data]

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_pymupdf_open.return_value = mock_doc

        pdf_file = tmp_path / "input.pdf"
        pdf_file.touch()

        output_dir = tmp_path / "output"

        with patch(
            "sys.argv",
            ["main.py", str(pdf_file), "--output-dir", str(output_dir)],
        ):
            result = main()

        assert result == 0
        # Verify output was created
        assert (output_dir / "input.json").exists()

    @patch("build_a_long.pdf_extract.main.pymupdf.open")
    @patch("build_a_long.pdf_extract.main.extract_bounding_boxes")
    def test_main_multiple_pdfs(
        self, mock_extract_bounding_boxes, mock_pymupdf_open, tmp_path
    ):
        """Test main() processing multiple PDF files."""
        page_data = PageData(
            page_number=1,
            blocks=[],
            bbox=BBox(0.0, 0.0, 100.0, 100.0),
        )
        mock_extract_bounding_boxes.return_value = [page_data]

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_pymupdf_open.return_value = mock_doc

        pdf1 = tmp_path / "test1.pdf"
        pdf2 = tmp_path / "test2.pdf"
        pdf1.touch()
        pdf2.touch()

        with patch("sys.argv", ["main.py", str(pdf1), str(pdf2)]):
            result = main()

        assert result == 0
        # Verify both were processed
        assert mock_pymupdf_open.call_count == 2
        assert (tmp_path / "test1.json").exists()
        assert (tmp_path / "test2.json").exists()
