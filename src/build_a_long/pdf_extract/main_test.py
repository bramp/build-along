"""Tests for main.py using minimal mocking and real temp files."""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from build_a_long.pdf_extract.classifier import ClassificationResult
from build_a_long.pdf_extract.cli import ProcessingConfig
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Image, Text
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
    @patch("build_a_long.pdf_extract.extractor.extractor.extract_page_data")
    def test_process_pdf_basic(
        self, mock_extract_page_data, mock_pymupdf_open, tmp_path
    ):
        """Test basic PDF processing with real temp directory."""
        # Create test data
        page_data = PageData(
            page_number=1,
            blocks=[Text(id=0, bbox=BBox(10.0, 20.0, 30.0, 40.0), text="1")],
            bbox=BBox(0.0, 0.0, 100.0, 100.0),
        )
        mock_extract_page_data.return_value = [page_data]

        # Mock PDF document
        mock_page = MagicMock()
        mock_page.get_text.return_value = {
            "blocks": []
        }  # Configure get_text to return a dictionary
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_pymupdf_open.return_value = mock_doc

        # Create a dummy PDF file in tmp_path for stat() and read_bytes()
        pdf_path = tmp_path / "test.pdf"
        pdf_content = b"dummy pdf content"
        pdf_path.write_bytes(pdf_content)

        config = _make_config(
            pdf_paths=[pdf_path],
            output_dir=tmp_path,
            save_debug_json=True,
        )

        result = _process_pdf(config, pdf_path, tmp_path)

        assert result == 0
        # Verify JSON file was created
        json_file = tmp_path / "test.json"
        assert json_file.exists()

        # Verify content
        saved_data = json.loads(json_file.read_text())
        assert saved_data["source_pdf"] == "test.pdf"
        assert saved_data["source_size"] == len(pdf_content)
        assert saved_data["source_hash"] == hashlib.sha256(pdf_content).hexdigest()

    @patch("build_a_long.pdf_extract.main.pymupdf.open")
    @patch("build_a_long.pdf_extract.main.extract_page_data")
    def test_process_pdf_filters_pages(
        self, mock_extract_page_data, mock_pymupdf_open, tmp_path
    ):
        """Test that --pages argument filters output correctly."""

        def _mk_page(n: int) -> PageData:
            return PageData(page_number=n, blocks=[], bbox=BBox(0.0, 0.0, 1.0, 1.0))

        # All pages for font hints (first call)
        all_pages = [_mk_page(i) for i in range(1, 201)]
        # Filtered pages for actual processing (second call)
        filtered_pages = [_mk_page(i) for i in [10, 11, 12, 15]]

        # Return different values for each call:
        # - First call: all pages for font hints
        # - Second call: only filtered pages for processing
        mock_extract_page_data.side_effect = [all_pages, filtered_pages]

        # Mock PDF document
        mock_page = MagicMock()
        mock_page.get_text.return_value = {
            "blocks": []
        }  # Configure get_text to return a dictionary
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 200
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_pymupdf_open.return_value = mock_doc

        # Create a dummy PDF file in tmp_path for stat() and read_bytes()
        pdf_path = tmp_path / "test.pdf"
        pdf_content = b"dummy pdf content"
        pdf_path.write_bytes(pdf_content)

        config = _make_config(
            pdf_paths=[pdf_path],
            output_dir=tmp_path,
            page_ranges="10-12,15",  # Filter to specific pages
            save_debug_json=True,
        )

        result = _process_pdf(config, pdf_path, tmp_path)

        assert result == 0

        # Verify extract_page_data was called twice
        assert mock_extract_page_data.call_count == 2

        # First call: all pages were extracted for font hints
        first_call_pages_arg = mock_extract_page_data.call_args_list[0][0][1]
        assert first_call_pages_arg == list(range(1, 201))

        # Second call: only filtered pages were extracted for actual processing
        second_call_pages_arg = mock_extract_page_data.call_args_list[1][0][1]
        assert second_call_pages_arg == [10, 11, 12, 15]

        # Verify that output files were created
        # Main output JSON
        assert (tmp_path / "test.json").exists()
        # Debug JSON when save_debug_json=True
        assert (tmp_path / "test_debug.json").exists()

        # Verify filtered pages are in the debug output
        debug_data = json.loads((tmp_path / "test_debug.json").read_text())
        saved_page_numbers = [
            p["page_data"]["page_number"] for p in debug_data["pages"]
        ]
        assert saved_page_numbers == [10, 11, 12, 15]

    @patch("build_a_long.pdf_extract.main.pymupdf.open")
    @patch("build_a_long.pdf_extract.extractor.extractor.extract_page_data")
    @patch("build_a_long.pdf_extract.cli.io.draw_and_save_bboxes")
    def test_process_pdf_with_drawing(
        self,
        mock_draw_and_save_bboxes,
        mock_extract_page_data,
        mock_pymupdf_open,
        tmp_path,
    ):
        """Test PDF processing with image drawing enabled."""
        page_data = PageData(
            page_number=1,
            blocks=[],
            bbox=BBox(0.0, 0.0, 100.0, 100.0),
        )
        mock_extract_page_data.return_value = [page_data]

        # Mock PDF document with page
        mock_page = MagicMock()
        mock_page.get_text.return_value = {
            "blocks": []
        }  # Configure get_text to return a dictionary
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_pymupdf_open.return_value = mock_doc

        # Create a dummy PDF file in tmp_path for stat() and read_bytes()
        pdf_path = tmp_path / "test.pdf"
        pdf_content = b"dummy pdf content"
        pdf_path.write_bytes(pdf_content)

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

    @patch("build_a_long.pdf_extract.main.pymupdf.open")
    @patch("build_a_long.pdf_extract.extractor.extractor.extract_page_data")
    def test_process_pdf_skips_full_page_image_pdfs(
        self, mock_extract_page_data, mock_pymupdf_open, tmp_path
    ):
        """Test that _process_pdf skips PDFs dominated by full-page images."""
        # Create page data where most pages are full-page images
        full_page_img_page = PageData(
            page_number=1,
            blocks=[Image(id=0, bbox=BBox(0, 0, 100, 200))],  # 100% image
            bbox=BBox(0, 0, 100, 200),
        )
        normal_page = PageData(
            page_number=2,
            blocks=[Text(id=0, bbox=BBox(10, 10, 20, 20), text="text")],
            bbox=BBox(0, 0, 100, 200),
        )
        mock_extract_page_data.return_value = [
            full_page_img_page,
            full_page_img_page,
            normal_page,
            full_page_img_page,
        ]  # 3/4 pages are full-page images (75%)

        # Mock PDF document
        mock_page = MagicMock()
        mock_page.get_text.return_value = {
            "blocks": []
        }  # Configure get_text to return a dictionary
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 4
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_pymupdf_open.return_value = mock_doc

        # Create a dummy PDF file
        pdf_path = tmp_path / "test_full_page_image.pdf"
        pdf_content = b"dummy pdf content for full page image test"
        pdf_path.write_bytes(pdf_content)

        config = _make_config(
            pdf_paths=[pdf_path],
            output_dir=tmp_path,
        )

        result = _process_pdf(config, pdf_path, tmp_path)

        assert result == 0
        # Verify unsupported JSON was created
        manual_json_path = tmp_path / "test_full_page_image.json"
        assert manual_json_path.exists()
        saved_data = json.loads(manual_json_path.read_text())
        assert saved_data.get("unsupported_reason") is None
        assert saved_data["source_pdf"] == pdf_path.name
        assert saved_data["source_size"] == len(pdf_content)
        assert saved_data["source_hash"] == hashlib.sha256(pdf_content).hexdigest()

    @patch("build_a_long.pdf_extract.main.pymupdf.open")
    @patch("build_a_long.pdf_extract.extractor.extractor.extract_page_data")
    def test_process_pdf_does_not_skip_normal_pdfs(
        self, mock_extract_page_data, mock_pymupdf_open, tmp_path
    ):
        """Test that _process_pdf does not skip normal PDFs (less than 50% full-page images)."""
        # Create page data where less than 50% of pages are full-page images
        full_page_img_page = PageData(
            page_number=1,
            blocks=[Image(id=0, bbox=BBox(0, 0, 100, 200))],  # 100% image
            bbox=BBox(0, 0, 100, 200),
        )
        normal_page = PageData(
            page_number=2,
            blocks=[Text(id=0, bbox=BBox(10, 10, 20, 20), text="text")],
            bbox=BBox(0, 0, 100, 200),
        )
        mock_extract_page_data.return_value = [
            full_page_img_page,
            normal_page,
            normal_page,
            normal_page,
        ]  # 1/4 pages are full-page images (25%)

        # Mock PDF document
        mock_page = MagicMock()
        mock_page.get_text.return_value = {
            "blocks": []
        }  # Configure get_text to return a dictionary
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 4
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_pymupdf_open.return_value = mock_doc

        # Create a dummy PDF file
        pdf_path = tmp_path / "test_normal.pdf"
        pdf_content = b"dummy pdf content for normal pdf test"
        pdf_path.write_bytes(pdf_content)

        config = _make_config(
            pdf_paths=[pdf_path],
            output_dir=tmp_path,
        )

        result = _process_pdf(config, pdf_path, tmp_path)

        assert result == 0
        # Verify unsupported JSON was NOT created
        manual_json_path = tmp_path / "test_normal.json"
        assert manual_json_path.exists()
        saved_data = json.loads(manual_json_path.read_text())
        assert saved_data.get("unsupported_reason") is None
        assert saved_data["source_pdf"] == pdf_path.name
        assert saved_data["source_size"] == len(pdf_content)
        assert saved_data["source_hash"] == hashlib.sha256(pdf_content).hexdigest()


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
    @patch("build_a_long.pdf_extract.extractor.extractor.extract_page_data")
    def test_main_with_pdf_and_output_dir(
        self, mock_extract_page_data, mock_pymupdf_open, tmp_path
    ):
        """Test main() with PDF file and custom output directory."""
        page_data = PageData(
            page_number=1,
            blocks=[],
            bbox=BBox(0.0, 0.0, 100.0, 100.0),
        )
        mock_extract_page_data.return_value = [page_data]

        mock_page = MagicMock()
        mock_page.get_text.return_value = {
            "blocks": []
        }  # Configure get_text to return a dictionary
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_pymupdf_open.return_value = mock_doc

        # Create a dummy PDF file in tmp_path for stat() and read_bytes()
        pdf_file = tmp_path / "input.pdf"
        pdf_content = b"dummy pdf content"
        pdf_file.write_bytes(pdf_content)

        output_dir = tmp_path / "output"

        with patch(
            "sys.argv",
            ["main.py", str(pdf_file), "--output-dir", str(output_dir)],
        ):
            result = main()

        assert result == 0
        # Verify output was created
        assert (output_dir / "input.json").exists()
        manual_data = json.loads((output_dir / "input.json").read_text())
        assert manual_data["source_pdf"] == "input.pdf"
        assert manual_data["source_size"] == len(pdf_content)
        assert manual_data["source_hash"] == hashlib.sha256(pdf_content).hexdigest()

    @patch("build_a_long.pdf_extract.main.pymupdf.open")
    @patch("build_a_long.pdf_extract.extractor.extractor.extract_page_data")
    def test_main_multiple_pdfs(
        self, mock_extract_page_data, mock_pymupdf_open, tmp_path
    ):
        """Test main() processing multiple PDF files."""
        page_data = PageData(
            page_number=1,
            blocks=[],
            bbox=BBox(0.0, 0.0, 100.0, 100.0),
        )
        mock_extract_page_data.return_value = [page_data]

        mock_page = MagicMock()
        mock_page.get_text.return_value = {
            "blocks": []
        }  # Configure get_text to return a dictionary
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__enter__.return_value = mock_doc
        mock_doc.__exit__.return_value = None
        mock_pymupdf_open.return_value = mock_doc

        pdf1 = tmp_path / "test1.pdf"
        pdf2 = tmp_path / "test2.pdf"
        pdf_content1 = b"dummy pdf content 1"
        pdf_content2 = b"dummy pdf content 2"
        pdf1.write_bytes(pdf_content1)
        pdf2.write_bytes(pdf_content2)

        with patch("sys.argv", ["main.py", str(pdf1), str(pdf2)]):
            result = main()

        assert result == 0
        # Verify both were processed
        assert mock_pymupdf_open.call_count == 2
        assert (tmp_path / "test1.json").exists()
        assert (tmp_path / "test2.json").exists()

        manual1 = json.loads((tmp_path / "test1.json").read_text())
        assert manual1["source_pdf"] == "test1.pdf"
        assert manual1["source_size"] == len(pdf_content1)
        assert manual1["source_hash"] == hashlib.sha256(pdf_content1).hexdigest()

        manual2 = json.loads((tmp_path / "test2.json").read_text())
        assert manual2["source_pdf"] == "test2.pdf"
        assert manual2["source_size"] == len(pdf_content2)
        assert manual2["source_hash"] == hashlib.sha256(pdf_content2).hexdigest()
