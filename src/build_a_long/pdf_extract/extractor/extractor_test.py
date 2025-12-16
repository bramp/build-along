from unittest.mock import MagicMock

from build_a_long.pdf_extract.extractor.extractor import (
    Extractor,
    extract_page_data,
)
from build_a_long.pdf_extract.extractor.page_blocks import (
    Drawing,
    Image,
    Text,
)
from build_a_long.pdf_extract.extractor.pymupdf_types import (
    TexttraceChar,
    TexttraceSpanDict,
)
from build_a_long.pdf_extract.extractor.testing_utils import PageBuilder


def make_texttrace_span(
    text: str,
    bbox: tuple[float, float, float, float],
    seqno: int,
    font: str = "Arial",
    size: float = 12.0,
) -> TexttraceSpanDict:
    """Helper to create a texttrace span dict for testing."""
    # Each char is (unicode, glyph_id, origin, bbox)
    chars: list[TexttraceChar] = [
        (ord(c), i, (bbox[0], bbox[3] - 5), bbox) for i, c in enumerate(text)
    ]
    return {
        "bbox": bbox,
        "font": font,
        "size": size,
        "seqno": seqno,
        "chars": chars,
    }


class TestBoundingBoxExtractor:
    def test_extract_page_data_basic(self):
        page = (
            PageBuilder()
            .add_text("1", (10.0, 20.0, 30.0, 40.0))
            .add_image(
                bbox=(50.0, 60.0, 150.0, 200.0),
                xref=10,
                number=1,
                image_id="Im1",
            )
            .build_mock_page()
        )

        fake_doc = MagicMock()
        fake_doc.__len__.return_value = 1
        fake_doc.__getitem__.return_value = page

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
        page = PageBuilder().add_text("1", (10.0, 20.0, 30.0, 40.0)).build_mock_page()

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = page

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
        page = (
            PageBuilder()
            .add_text("Build Step Instructions", (10.0, 20.0, 100.0, 40.0))
            .build_mock_page()
        )

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = page

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
        page = (
            PageBuilder()
            .add_text("First", (10.0, 20.0, 30.0, 40.0))
            .add_text("Second", (35.0, 20.0, 55.0, 40.0))
            .add_image(
                bbox=(50.0, 60.0, 150.0, 200.0),
                xref=10,
                number=1,
                image_id="Im1",
            )
            .add_drawing(
                bbox=(100.0, 100.0, 200.0, 200.0),
                seqno=3,
            )
            .build_mock_page()
        )

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = page

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
        page1 = (
            PageBuilder().add_text("Page 1", (10.0, 20.0, 30.0, 40.0)).build_mock_page()
        )
        page2 = (
            PageBuilder().add_text("Page 2", (10.0, 20.0, 30.0, 40.0)).build_mock_page()
        )

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 2

        def _getitem(idx):
            if idx == 0:
                return page1
            elif idx == 1:
                return page2
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
        # NOTE: Extractor.extract_image_blocks relies on BBoxLogTracker which
        # needs matching bboxlog entries. PageBuilder handles this automatically.
        page = (
            PageBuilder()
            # Text will be injected via texttrace manually in this test
            .add_image(
                bbox=(100.0, 200.0, 150.0, 250.0),
                xref=10,
                number=2,
                image_id="Im1",
                width=50,
                height=50,
            )
            .add_drawing(
                bbox=(10.0, 20.0, 30.0, 40.0),
            )
            .build_mock_page()
        )

        extractor = Extractor(page=page, page_num=1)

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
        page = (
            PageBuilder()
            .add_image(
                bbox=(100.0, 200.0, 150.0, 250.0),
                xref=10,
                number=2,
                image_id="Im1",
                width=50,
                height=50,
            )
            .add_image(
                bbox=(200.0, 300.0, 250.0, 350.0),
                xref=11,
                number=3,
                image_id="Im2",
                width=50,
                height=50,
            )
            .build_mock_page()
        )

        # Manually fix get_images to simulate smask since PageBuilder doesn't
        # support detailed smask config yet (it defaults to smask=0)
        # We need to override the return value of get_images
        page.get_images.return_value = [
            (10, 0, 50, 50, 8, 3, "", "Im1", "DCTDecode", 0),  # no smask
            (11, 15, 50, 50, 8, 3, "", "Im2", "DCTDecode", 0),  # has smask=15
        ]

        # Test without smask extraction (default)
        extractor = Extractor(page=page, page_num=1)
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
        extractor_with_smask = Extractor(page=page, page_num=1, include_smask=True)
        result_with_smask = extractor_with_smask.extract_image_blocks()

        assert len(result_with_smask) == 2
        assert result_with_smask[0].smask is None  # no smask for first image
        assert result_with_smask[1].smask == 15  # has smask

    def test_extract_drawing_blocks(self):
        """Test Extractor.extract_drawing_blocks extracts drawings from
        page.get_drawings()."""
        page = (
            PageBuilder()
            .add_drawing(bbox=(10.0, 20.0, 30.0, 40.0))
            .add_drawing(bbox=(50.0, 60.0, 70.0, 80.0))
            .build_mock_page()
        )

        extractor = Extractor(page=page, page_num=1)
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
        page = PageBuilder().build_mock_page()
        extractor = Extractor(page=page, page_num=1)
        result = extractor.extract_image_blocks()
        assert result == []

    def test_extract_drawing_blocks_empty_drawings(self):
        """Test Extractor.extract_drawing_blocks handles empty drawings list."""
        page = PageBuilder().build_mock_page()
        extractor = Extractor(page=page, page_num=1)
        result = extractor.extract_drawing_blocks()
        assert result == []
