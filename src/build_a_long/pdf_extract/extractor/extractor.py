import logging
from dataclasses import dataclass
from typing import Any, List, Set

import pymupdf
from dataclasses_json import DataClassJsonMixin

from build_a_long.pdf_extract.extractor.bbox import BBox

# Note: We intentionally do not build hierarchy here to avoid syncing issues
from build_a_long.pdf_extract.extractor.page_elements import (
    Drawing,
    Image,
    Text,
    PageElement,
)
from build_a_long.pdf_extract.extractor.pymupdf_types import (
    BBoxTuple,
    BlockDict,
    ImageBlockDict,
    RawDict,
    TextBlockDict,
)

logger = logging.getLogger("extractor")


@dataclass
class PageData(DataClassJsonMixin):
    """Data extracted from a single PDF page.

    Attributes:
        page_number: The page number (1-indexed)
        elements: Flat list of all elements on the page
        bbox: The bounding box of the entire page (page coordinate space).
    """

    page_number: int
    elements: List[PageElement]
    bbox: BBox


@dataclass
class ExtractionResult(DataClassJsonMixin):
    """Top-level container for extracted PDF data.

    This wraps the pages array and a schema version, making JSON IO trivial
    and future-proofing the format for additional metadata.
    """

    pages: List[PageData]
    schema_version: int = 1


def _extract_text_elements(blocks: List[BlockDict]) -> List[PageElement]:
    """Extract text elements from a page's raw dictionary blocks.

    Args:
        blocks: List of block dictionaries from page.get_text("rawdict")["blocks"]

    Returns:
        List of Text elements
    """
    elements: List[PageElement] = []

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

                elements.append(
                    Text(
                        bbox=nbbox,
                        text=text,
                        font_name=font_name,
                        font_size=font_size,
                    )
                )

    return elements


def _extract_image_elements(blocks: List[BlockDict]) -> List[PageElement]:
    """Extract image elements from a page's raw dictionary blocks.

    Args:
        blocks: List of block dictionaries from page.get_text("rawdict")["blocks"]

    Returns:
        List of Image elements
    """
    elements: List[PageElement] = []

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

        elements.append(Image(bbox=nbbox, image_id=f"image_{bi}"))
        logger.debug("Found image %s with %s", bi, nbbox)

    return elements


def _extract_drawing_elements(drawings: List[Any]) -> List[PageElement]:
    """Extract drawing (vector path) elements from a page.

    Args:
        drawings: List of drawing dictionaries from page.get_drawings()

    Returns:
        List of Drawing elements
    """
    elements: List[PageElement] = []

    for d in drawings:
        drect = d["rect"]
        nbbox = BBox.from_tuple((drect.x0, drect.y0, drect.x1, drect.y1))
        elements.append(Drawing(bbox=nbbox))
        logger.debug("Found drawing with %s", nbbox)

    return elements


def _warn_unknown_block_types(blocks: List[Any]) -> bool:
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


def _extract_page_elements(
    page: pymupdf.Page, page_num: int, include_types: Set[str]
) -> PageData:
    """Extract all elements from a single page.

    Args:
        page: PyMuPDF page object
        page_num: Page number (1-indexed)
        include_types: Set of element types to include

    Returns:
        PageData with all extracted elements
    """
    logger.info("Processing page %s", page_num)

    # Get raw dictionary with text and image blocks
    raw: RawDict = page.get_text("rawdict")  # type: ignore[assignment]
    assert isinstance(raw, dict)

    blocks = raw.get("blocks", [])
    assert _warn_unknown_block_types(blocks)

    # Extract elements by type
    typed_elements: List[PageElement] = []
    if "text" in include_types:
        typed_elements.extend(_extract_text_elements(blocks))
    if "image" in include_types:
        typed_elements.extend(_extract_image_elements(blocks))
    if "drawing" in include_types:
        drawings = page.get_drawings()
        typed_elements.extend(_extract_drawing_elements(drawings))

    # Assign sequential IDs to elements within this page
    for i, element in enumerate(typed_elements):
        element.id = i

    page_rect = page.rect
    page_bbox = BBox.from_tuple(
        (page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1)
    )

    return PageData(
        page_number=page_num,
        elements=typed_elements,
        bbox=page_bbox,
    )


def extract_bounding_boxes(
    doc: pymupdf.Document,
    start_page: int | None = None,
    end_page: int | None = None,
    include_types: Set[str] = {"text", "image", "drawing"},
) -> List[PageData]:
    """
    Extract bounding boxes for instruction numbers, parts lists, and build steps
    from a given PDF document using PyMuPDF.

    Args:
        doc: PyMuPDF Document object
        start_page: First page to process (1-indexed), None for first page
        end_page: Last page to process (1-indexed, inclusive), None for last page
        include_types: Set of element types to include ("text", "image", "drawing")

    Returns:
        List of PageData containing all pages with their elements
    """
    pages: List[PageData] = []

    num_pages = len(doc)

    # Determine page range (convert to 0-indexed)
    first_page = (start_page - 1) if start_page is not None else 0
    last_page = (end_page - 1) if end_page is not None else (num_pages - 1)

    # Validate and clamp page range
    first_page = max(0, min(first_page, num_pages - 1))
    last_page = max(0, min(last_page, num_pages - 1))

    if first_page > last_page:
        logger.warning("Invalid page range %s-%s", str(start_page), str(end_page))
        return []

    logger.info(
        "Processing pages %s-%s of %s", first_page + 1, last_page + 1, num_pages
    )

    for page_index in range(first_page, last_page + 1):
        page = doc[page_index]
        page_num = page_index + 1

        page_data = _extract_page_elements(page, page_num, include_types)
        pages.append(page_data)

    return pages
