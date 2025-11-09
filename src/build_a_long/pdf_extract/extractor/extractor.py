import logging
from collections.abc import Sequence
from typing import Any

import pymupdf
from pydantic import BaseModel

from build_a_long.pdf_extract.extractor.bbox import BBox

# Note: We intentionally do not build hierarchy here to avoid syncing issues
from build_a_long.pdf_extract.extractor.page_blocks import (
    Block,
    Drawing,
    Image,
    Text,
)
from build_a_long.pdf_extract.extractor.pymupdf_types import (
    BBoxTuple,
    BlockDict,
    ImageBlockDict,
    RawDict,
    TextBlockDict,
)

logger = logging.getLogger("extractor")


class PageData(BaseModel):
    """Data extracted from a single PDF page.

    Attributes:
        page_number: The page number (1-indexed) from the PDF metadata.
        bbox: The bounding box of the entire page (page coordinate space).
        blocks: Flat list of all blocks on the page
    """

    page_number: int
    bbox: BBox
    blocks: list[Block]


class ExtractionResult(BaseModel):
    """Top-level container for extracted PDF data."""

    pages: list[PageData]


class Extractor:
    """Handles extraction of page elements with sequential ID assignment.

    Each Extractor instance should be used for a single page to ensure
    IDs are sequential and reset for each page.
    """

    def __init__(self) -> None:
        """Initialize the extractor with a fresh ID counter."""
        self._next_id = 0

    def _get_next_id(self) -> int:
        """Get the next sequential ID and increment the counter."""
        current_id = self._next_id
        self._next_id += 1
        return current_id

    def _extract_text_blocks(self, blocks: list[BlockDict]) -> list[Text]:
        """Extract text blocks from a page's raw dictionary blocks.

        Args:
            blocks: List of block dictionaries from page.get_text("rawdict")["blocks"]

        Returns:
            List of Text blocks with assigned IDs
        """
        text_blocks: list[Text] = []

        for b in blocks:
            assert isinstance(b, dict)

            bi: int | None = b.get("number")
            btype: int | None = b.get("type")  # 0=text, 1=image

            if btype != 0:  # Skip non-text blocks
                continue

            # Now we know b is a text block
            text_block: TextBlockDict = b  # type: ignore[assignment]

            for line in text_block.get("lines", []):
                for span in line.get("spans", []):
                    sbbox: BBoxTuple = span.get("bbox", (0.0, 0.0, 0.0, 0.0))
                    nbbox = BBox.from_tuple(sbbox)

                    text: str = span.get("text", None)
                    if text is None or text == "":
                        chars = span.get("chars", [])
                        text = "".join(c["c"] for c in chars)

                    font_size: float = span.get("size", 0.0)
                    font_name: str = span.get("font", "unknown")

                    logger.debug(
                        "Found text %s %r with bbox %s, font %s, size %s",
                        bi,
                        text,
                        nbbox,
                        font_name,
                        font_size,
                    )

                    text_blocks.append(
                        Text(
                            bbox=nbbox,
                            text=text,
                            font_name=font_name,
                            font_size=font_size,
                            id=self._get_next_id(),
                        )
                    )

        return text_blocks

    def _extract_image_blocks(self, blocks: list[BlockDict]) -> list[Image]:
        """Extract image blocks from a page's raw dictionary blocks.

        Args:
            blocks: List of block dictionaries from page.get_text("rawdict")["blocks"]

        Returns:
            List of Image blocks with assigned IDs
        """
        image_blocks: list[Image] = []

        for b in blocks:
            assert isinstance(b, dict)

            bi: int | None = b.get("number")
            btype: int | None = b.get("type")  # 0=text, 1=image

            if btype != 1:  # Skip non-image blocks
                continue

            # Now we know b is an image block
            image_block: ImageBlockDict = b  # type: ignore[assignment]

            bbox: BBoxTuple = image_block.get("bbox", (0.0, 0.0, 0.0, 0.0))
            nbbox = BBox.from_tuple(bbox)

            image_blocks.append(
                Image(
                    bbox=nbbox,
                    image_id=f"image_{bi}",
                    id=self._get_next_id(),
                )
            )
            logger.debug("Found image %s with %s", bi, nbbox)

        return image_blocks

    def _extract_drawing_blocks(self, drawings: list[Any]) -> list[Drawing]:
        """Extract drawing (vector path) blocks from a page.

        Args:
            drawings: List of drawing dictionaries from page.get_drawings()

        Returns:
            List of Drawing blocks with assigned IDs
        """
        drawing_blocks: list[Drawing] = []

        for d in drawings:
            drect = d["rect"]
            nbbox = BBox.from_tuple((drect.x0, drect.y0, drect.x1, drect.y1))
            drawing_blocks.append(Drawing(bbox=nbbox, id=self._get_next_id()))
            logger.debug("Found drawing with %s", nbbox)

        return drawing_blocks

    def _warn_unknown_block_types(self, blocks: list[Any]) -> bool:
        """Log warnings for blocks with unsupported types.

        Args:
            blocks: List of block dictionaries from page.get_text("rawdict")["blocks"]

        Returns:
            True if all blocks are valid types, False if any unknown types were found
        """
        for b in blocks:
            assert isinstance(b, dict)
            bi: int | None = b.get("number")
            btype: int | None = b.get("type")  # 0=text, 1=image

            if btype not in (0, 1):
                logger.warning(
                    "Skipping block with unsupported type %s at index %s", btype, bi
                )
                return False
        return True

    def extract_page_blocks(
        self, page: pymupdf.Page, page_num: int, include_types: set[str]
    ) -> PageData:
        """Extract all blocks from a single page.

        Args:
            page: PyMuPDF page object
            page_num: Page number (1-indexed)
            include_types: Set of element types to include

        Returns:
            PageData with all extracted blocks (with sequential IDs)
        """
        logger.debug("Processing page %s", page_num)

        # Get raw dictionary with text and image blocks
        raw: RawDict = page.get_text("rawdict")  # type: ignore[assignment]
        assert isinstance(raw, dict)

        blocks = raw.get("blocks", [])
        assert self._warn_unknown_block_types(blocks)

        # Extract blocks by type (IDs are assigned during creation)
        typed_blocks: list[Block] = []
        if "text" in include_types:
            typed_blocks.extend(self._extract_text_blocks(blocks))
        if "image" in include_types:
            typed_blocks.extend(self._extract_image_blocks(blocks))
        if "drawing" in include_types:
            drawings = page.get_drawings()
            typed_blocks.extend(self._extract_drawing_blocks(drawings))

        page_rect = page.rect
        page_bbox = BBox.from_tuple(
            (page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1)
        )

        return PageData(
            page_number=page_num,
            blocks=typed_blocks,
            bbox=page_bbox,
        )


def extract_bounding_boxes(
    doc: pymupdf.Document,
    page_numbers: Sequence[int] | None = None,
    include_types: set[str] | None = None,
) -> list[PageData]:
    """
    Extract bounding boxes for the selected pages of a PDF document.

    Args:
        doc: PyMuPDF Document object
        page_numbers: A sequence of 1-indexed page numbers to process. If None or
            empty, all pages are processed.
        include_types: Set of element types to include ("text", "image", "drawing").
            If None, defaults to all types.

    Returns:
        List of PageData containing all pages with their blocks
    """
    if include_types is None:
        include_types = {"text", "image", "drawing"}

    pages: list[PageData] = []

    num_pages = len(doc)

    # If not provided, process all pages (1-indexed numbers)
    if not page_numbers:
        page_numbers = [i + 1 for i in range(num_pages)]

    for page_num in page_numbers:
        page_index = page_num - 1  # convert to 0-indexed
        if page_index < 0 or page_index >= num_pages:
            # Out of bounds; skip defensively
            continue
        page = doc[page_index]

        # Create a new Extractor for each page to reset IDs
        extractor = Extractor()
        page_data = extractor.extract_page_blocks(page, page_num, include_types)
        pages.append(page_data)

    return pages
