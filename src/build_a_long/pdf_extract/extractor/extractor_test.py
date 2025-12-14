from typing import cast
from unittest.mock import MagicMock

import pymupdf

from build_a_long.pdf_extract.extractor import (
    extract_page_data,
)
from build_a_long.pdf_extract.extractor.extractor import Extractor
from build_a_long.pdf_extract.extractor.page_blocks import (
    Drawing,
    Image,
    Text,
)
from build_a_long.pdf_extract.extractor.pymupdf_types import DrawingDict


def make_texttrace_span(
    text: str,
    bbox: tuple[float, float, float, float],
    seqno: int,
    font: str = "Arial",
    size: float = 12.0,
) -> dict:
    """Helper to create a texttrace span dict for testing."""
    # Each char is (unicode, glyph_id, origin, bbox)
    chars = [(ord(c), i, (bbox[0], bbox[3] - 5), bbox) for i, c in enumerate(text)]
    return {
        "bbox": bbox,
        "font": font,
        "size": size,
        "seqno": seqno,
        "chars": chars,
    }


def make_rawdict_span(
    text: str,
    bbox: tuple[float, float, float, float],
    font: str = "Arial",
    size: float = 12.0,
) -> dict:
    """Helper to create a rawdict span dict for testing.

    The rawdict format has chars as list of dicts with 'c' (char) and 'bbox' keys.
    """
    chars = [
        {"c": c, "bbox": (bbox[0] + i * 4, bbox[1], bbox[0] + (i + 1) * 4, bbox[3])}
        for i, c in enumerate(text)
    ]
    return {
        "bbox": bbox,
        "font": font,
        "size": size,
        "chars": chars,
        "origin": (bbox[0], bbox[3] - 5),
    }


def make_rawdict(spans: list[dict]) -> dict:
    """Helper to create a rawdict structure from a list of span dicts.

    The rawdict structure is: {"blocks": [{"type": 0, "lines": [{"spans": [...]}]}]}
    Each span becomes its own line for simplicity.
    """
    lines = [{"spans": [span], "bbox": span["bbox"]} for span in spans]
    return {
        "blocks": [
            {
                "type": 0,  # text block
                "lines": lines,
            }
        ]
    }


class TestBoundingBoxExtractor:
    def test_extract_page_data_basic(self):
        # Build a fake document with 1 page and simple rawdict content
        fake_page = MagicMock()
        # Mock get_text for rawdict extraction (default path)
        fake_page.get_text.return_value = make_rawdict(
            [
                make_rawdict_span("1", (10.0, 20.0, 30.0, 40.0)),
            ]
        )
        fake_page.get_bboxlog.return_value = [
            ("fill-text", (10.0, 20.0, 30.0, 40.0)),
            ("fill-image", (50.0, 60.0, 150.0, 200.0)),
        ]
        fake_page.get_drawings.return_value = []
        # Mock get_image_info for the new image extraction approach
        fake_page.get_image_info.return_value = [
            {
                "number": 1,
                "bbox": (50.0, 60.0, 150.0, 200.0),
                "width": 100,
                "height": 140,
                "colorspace": 3,
                "xres": 96,
                "yres": 96,
                "bpc": 8,
                "size": 5000,
                "transform": (1.0, 0.0, 0.0, 1.0, 50.0, 60.0),
                "xref": 10,
            },
        ]
        fake_page.get_images.return_value = [
            (10, 0, 100, 140, 8, 3, "", "Im1", "DCTDecode", 0),
        ]
        fake_page.rect = MagicMock(x0=0, y0=0, x1=612, y1=792)

        fake_doc = MagicMock()
        fake_doc.__len__.return_value = 1

        def _getitem(idx):
            assert idx == 0
            return fake_page

        fake_doc.__getitem__.side_effect = _getitem

        # Call the function with document instead of path
        result = extract_page_data(fake_doc)

        # Validate typed elements structure (no artificial Root element)
        assert len(result) == 1
        page_data = result[0]
        assert page_data.page_number == 1
        blocks = page_data.blocks
        # blocks now only include actual content: 1 Text + 1 Image
        assert len(blocks) == 2
        assert isinstance(blocks[0], Text)
        assert blocks[0].text == "1"
        assert blocks[0].bbox.x0 == 10.0 and blocks[0].bbox.y0 == 20.0
        assert isinstance(blocks[1], Image)
        assert blocks[1].xref == 10  # Verify xref is extracted

    def test_extract_page_data_with_output(self):
        """Test that extractor does NOT write images/json (that's in main.py now)."""
        mock_page = MagicMock()
        # Mock get_text for rawdict extraction (default path)
        mock_page.get_text.return_value = make_rawdict(
            [
                make_rawdict_span("1", (10.0, 20.0, 30.0, 40.0)),
            ]
        )
        mock_page.get_bboxlog.return_value = [
            ("fill-text", (10.0, 20.0, 30.0, 40.0)),
        ]
        mock_page.get_drawings.return_value = []
        mock_page.get_image_info.return_value = []  # No images
        mock_page.get_images.return_value = []
        mock_page.rect = MagicMock(x0=0, y0=0, x1=612, y1=792)

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page

        result = extract_page_data(mock_doc)

        # Typed elements exist (no Root)
        assert len(result) == 1
        page_data = result[0]
        blocks = page_data.blocks
        assert len(blocks) == 1  # Only 1 Text block
        assert isinstance(blocks[0], Text)
        assert blocks[0].text == "1"

    def test_extract_text_blocks(self):
        """Test that regular text is extracted as Text elements with content."""
        mock_page = MagicMock()
        # Mock get_text for rawdict extraction (default path)
        text = "Build Step Instructions"
        mock_page.get_text.return_value = make_rawdict(
            [
                make_rawdict_span(text, (10.0, 20.0, 100.0, 40.0)),
            ]
        )
        mock_page.get_bboxlog.return_value = [
            ("fill-text", (10.0, 20.0, 100.0, 40.0)),
        ]
        mock_page.get_drawings.return_value = []
        mock_page.get_image_info.return_value = []  # No images
        mock_page.get_images.return_value = []
        mock_page.rect = MagicMock(x0=0, y0=0, x1=612, y1=792)

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page

        result = extract_page_data(mock_doc)

        # Validate text element (no Root)
        assert len(result) == 1
        page_data = result[0]
        blocks = page_data.blocks
        assert len(blocks) == 1  # Only 1 Text block
        assert isinstance(blocks[0], Text)
        assert blocks[0].text == "Build Step Instructions"
        assert blocks[0].bbox.x0 == 10.0

    def test_sequential_id_assignment(self):
        """Test that IDs are assigned sequentially within a page."""
        mock_page = MagicMock()
        # Mock get_text for rawdict extraction (default path)
        mock_page.get_text.return_value = make_rawdict(
            [
                make_rawdict_span("First", (10.0, 20.0, 30.0, 40.0)),
                make_rawdict_span("Second", (35.0, 20.0, 55.0, 40.0)),
            ]
        )
        mock_page.get_bboxlog.return_value = [
            ("fill-text", (10.0, 20.0, 30.0, 40.0)),
            ("fill-text", (35.0, 20.0, 55.0, 40.0)),
            ("fill-image", (50.0, 60.0, 150.0, 200.0)),
            ("fill-path", (100.0, 100.0, 200.0, 200.0)),
        ]
        mock_page.get_drawings.return_value = [
            {"rect": pymupdf.Rect(100, 100, 200, 200), "seqno": 3},
        ]
        # Mock get_image_info for the new image extraction approach
        mock_page.get_image_info.return_value = [
            {
                "number": 1,
                "bbox": (50.0, 60.0, 150.0, 200.0),
                "width": 100,
                "height": 140,
                "colorspace": 3,
                "xres": 96,
                "yres": 96,
                "bpc": 8,
                "size": 5000,
                "transform": (1.0, 0.0, 0.0, 1.0, 50.0, 60.0),
                "xref": 10,
            },
        ]
        mock_page.get_images.return_value = [
            (10, 0, 100, 140, 8, 3, "", "Im1", "DCTDecode", 0),
        ]
        mock_page.rect = MagicMock(x0=0, y0=0, x1=612, y1=792)
        mock_page.transformation_matrix = pymupdf.Identity

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page

        result = extract_page_data(mock_doc)

        # Validate IDs are sequential starting from 0
        assert len(result) == 1
        page_data = result[0]
        blocks = page_data.blocks
        assert len(blocks) == 4  # 2 Text + 1 Image + 1 Drawing

        # Check that IDs are sequential: 0, 1, 2, 3
        for i, block in enumerate(blocks):
            assert block.id == i, f"Block {i} has id {block.id}, expected {i}"

    def test_id_reset_across_pages(self):
        """Test that IDs reset to 0 for each new page."""
        mock_page1 = MagicMock()
        # Mock get_text for rawdict extraction (default path)
        mock_page1.get_text.return_value = make_rawdict(
            [
                make_rawdict_span("Page 1", (10.0, 20.0, 30.0, 40.0)),
            ]
        )
        mock_page1.get_bboxlog.return_value = [
            ("fill-text", (10.0, 20.0, 30.0, 40.0)),
        ]
        mock_page1.get_drawings.return_value = []
        mock_page1.get_image_info.return_value = []  # No images
        mock_page1.get_images.return_value = []
        mock_page1.rect = MagicMock(x0=0, y0=0, x1=612, y1=792)

        mock_page2 = MagicMock()
        # Mock get_text for rawdict extraction (default path)
        mock_page2.get_text.return_value = make_rawdict(
            [
                make_rawdict_span("Page 2", (10.0, 20.0, 30.0, 40.0)),
            ]
        )
        mock_page2.get_bboxlog.return_value = [
            ("fill-text", (10.0, 20.0, 30.0, 40.0)),
        ]
        mock_page2.get_drawings.return_value = []
        mock_page2.get_image_info.return_value = []  # No images
        mock_page2.get_images.return_value = []
        mock_page2.rect = MagicMock(x0=0, y0=0, x1=612, y1=792)

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 2

        def _getitem(idx):
            if idx == 0:
                return mock_page1
            elif idx == 1:
                return mock_page2
            raise IndexError()

        mock_doc.__getitem__.side_effect = _getitem

        result = extract_page_data(mock_doc)

        # Both pages should have elements with ID starting from 0
        assert len(result) == 2
        assert result[0].blocks[0].id == 0
        assert isinstance(result[0].blocks[0], Text)
        assert result[0].blocks[0].text == "Page 1"
        assert result[1].blocks[0].id == 0  # Reset to 0 for second page
        assert isinstance(result[1].blocks[0], Text)
        assert result[1].blocks[0].text == "Page 2"


class TestExtractor:
    """Tests for the Extractor class."""

    def test_extractor_sequential_ids(self):
        """Test that Extractor assigns sequential IDs."""
        # Create a minimal mock page
        mock_page = MagicMock()
        extractor = Extractor(page=mock_page, page_num=1)

        texttrace = [
            make_texttrace_span("Hello", (10.0, 20.0, 30.0, 40.0), seqno=0),
            make_texttrace_span("World", (35.0, 20.0, 55.0, 40.0), seqno=1),
        ]

        result = extractor._extract_text_blocks_from_texttrace(texttrace)

        assert len(result) == 2
        assert result[0].id == 0
        assert result[1].id == 1

    def test_extractor_ids_across_types(self):
        """Test that IDs increment across different element types."""
        drawings = cast(
            list[DrawingDict], [{"rect": pymupdf.Rect(10.0, 20.0, 30.0, 40.0)}]
        )

        # Create mock page with identity transformation matrix
        mock_page = MagicMock()
        mock_page.transformation_matrix = pymupdf.Identity
        mock_page.get_bboxlog.return_value = [
            ("fill-image", (100.0, 200.0, 150.0, 250.0)),
        ]
        mock_page.get_image_info.return_value = [
            {
                "number": 2,
                "bbox": (100.0, 200.0, 150.0, 250.0),
                "width": 50,
                "height": 50,
                "colorspace": 3,
                "xres": 96,
                "yres": 96,
                "bpc": 8,
                "size": 1000,
                "transform": (1.0, 0.0, 0.0, 1.0, 100.0, 200.0),
                "xref": 10,
            },
        ]
        mock_page.get_images.return_value = [
            (10, 0, 50, 50, 8, 3, "", "Im1", "DCTDecode", 0),
        ]
        mock_page.get_drawings.return_value = drawings

        extractor = Extractor(page=mock_page, page_num=1)

        texttrace = [
            make_texttrace_span("Text1", (10.0, 20.0, 30.0, 40.0), seqno=0),
        ]

        # Extract in order and verify IDs increment
        texts = extractor._extract_text_blocks_from_texttrace(texttrace)
        images = extractor.extract_image_blocks()
        draw = extractor.extract_drawing_blocks()

        assert texts[0].id == 0
        assert images[0].id == 1
        assert draw[0].id == 2


class TestExtractorMethods:
    """Tests for the Extractor class methods."""

    def test_extract_text_blocks_from_texttrace(self):
        """Test Extractor._extract_text_blocks_from_texttrace extracts text."""
        mock_page = MagicMock()
        extractor = Extractor(page=mock_page, page_num=1)
        texttrace = [
            make_texttrace_span("Hello", (10.0, 20.0, 30.0, 40.0), seqno=0),
            make_texttrace_span("World", (35.0, 20.0, 55.0, 40.0), seqno=1),
        ]

        result = extractor._extract_text_blocks_from_texttrace(texttrace)

        assert len(result) == 2
        assert isinstance(result[0], Text)
        assert result[0].text == "Hello"
        assert result[0].bbox.x0 == 10.0
        assert result[0].id == 0
        assert result[0].draw_order == 0
        assert isinstance(result[1], Text)
        assert result[1].text == "World"
        assert result[1].bbox.x0 == 35.0
        assert result[1].id == 1
        assert result[1].draw_order == 1

    def test_extract_image_blocks(self):
        """Test Extractor.extract_image_blocks extracts images from page."""
        # Create mock page with get_image_info and get_images
        mock_page = MagicMock()
        mock_page.get_bboxlog.return_value = [
            ("fill-image", (100.0, 200.0, 150.0, 250.0)),
            ("fill-image", (200.0, 300.0, 250.0, 350.0)),
        ]
        mock_page.get_image_info.return_value = [
            {
                "number": 2,
                "bbox": (100.0, 200.0, 150.0, 250.0),
                "width": 50,
                "height": 50,
                "colorspace": 3,
                "xres": 96,
                "yres": 96,
                "bpc": 8,
                "size": 1000,
                "transform": (1.0, 0.0, 0.0, 1.0, 100.0, 200.0),
                "xref": 10,
            },
            {
                "number": 3,
                "bbox": (200.0, 300.0, 250.0, 350.0),
                "width": 50,
                "height": 50,
                "colorspace": 3,
                "xres": 96,
                "yres": 96,
                "bpc": 8,
                "size": 1200,
                "transform": (1.0, 0.0, 0.0, 1.0, 200.0, 300.0),
                "xref": 11,
            },
        ]
        # get_images returns: (xref, smask, width, height, bpc, colorspace, ...)
        mock_page.get_images.return_value = [
            (10, 0, 50, 50, 8, 3, "", "Im1", "DCTDecode", 0),  # no smask
            (11, 15, 50, 50, 8, 3, "", "Im2", "DCTDecode", 0),  # has smask=15
        ]

        # Test without smask extraction (default)
        extractor = Extractor(page=mock_page, page_num=1)
        result = extractor.extract_image_blocks()

        assert len(result) == 2
        assert isinstance(result[0], Image)
        assert result[0].image_id == "image_2"
        assert result[0].bbox.x0 == 100.0
        assert result[0].id == 0
        assert result[0].xref == 10
        assert result[0].smask is None  # no smask when include_smask=False
        assert isinstance(result[1], Image)
        assert result[1].image_id == "image_3"
        assert result[1].bbox.x0 == 200.0
        assert result[1].id == 1
        assert result[1].xref == 11
        assert result[1].smask is None  # no smask when include_smask=False

        # Test with smask extraction enabled
        extractor_with_smask = Extractor(page=mock_page, page_num=1, include_smask=True)
        result_with_smask = extractor_with_smask.extract_image_blocks()

        assert len(result_with_smask) == 2
        assert result_with_smask[0].smask is None  # no smask for first image
        assert result_with_smask[1].smask == 15  # has smask

    def test_extract_drawing_blocks(self):
        """Test Extractor.extract_drawing_blocks extracts drawings from
        page.get_drawings()."""
        # Create real rectangle objects for PyMuPDF compatibility
        drawings = cast(
            list[DrawingDict],
            [
                {"rect": pymupdf.Rect(10.0, 20.0, 30.0, 40.0)},
                {"rect": pymupdf.Rect(50.0, 60.0, 70.0, 80.0)},
            ],
        )

        # Create mock page with identity transformation matrix
        mock_page = MagicMock()
        mock_page.transformation_matrix = pymupdf.Identity
        mock_page.get_drawings.return_value = drawings

        extractor = Extractor(page=mock_page, page_num=1)
        result = extractor.extract_drawing_blocks()

        assert len(result) == 2
        assert isinstance(result[0], Drawing)
        assert result[0].bbox.x0 == 10.0
        assert result[0].bbox.y0 == 20.0
        assert result[0].id == 0
        assert isinstance(result[1], Drawing)
        assert result[1].bbox.x0 == 50.0
        assert result[1].bbox.y0 == 60.0
        assert result[1].id == 1

    def test_extract_text_blocks_from_texttrace_empty(self):
        """Test Extractor._extract_text_blocks_from_texttrace handles empty list."""
        mock_page = MagicMock()
        extractor = Extractor(page=mock_page, page_num=1)
        result = extractor._extract_text_blocks_from_texttrace([])
        assert result == []

    def test_extract_image_blocks_empty_page(self):
        """Test Extractor.extract_image_blocks handles page with no images."""
        # Create mock page that returns no images
        mock_page = MagicMock()
        mock_page.get_image_info.return_value = []
        mock_page.get_images.return_value = []

        extractor = Extractor(page=mock_page, page_num=1)
        result = extractor.extract_image_blocks()
        assert result == []

    def test_extract_drawing_blocks_empty_drawings(self):
        """Test Extractor.extract_drawing_blocks handles empty drawings list."""
        # Create mock page with identity transformation matrix
        mock_page = MagicMock()
        mock_page.transformation_matrix = pymupdf.Identity
        mock_page.get_drawings.return_value = []

        extractor = Extractor(page=mock_page, page_num=1)
        result = extractor.extract_drawing_blocks()
        assert result == []
