import logging
from dataclasses import dataclass, replace
from typing import Any, List, Set, Sequence

import pymupdf
from dataclass_wizard import JSONPyWizard

from build_a_long.pdf_extract.extractor.bbox import BBox

# Note: We intentionally do not build hierarchy here to avoid syncing issues
from build_a_long.pdf_extract.extractor.page_elements import (
    Drawing,
    Image,
    Text,
    Element,
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
class PageData(JSONPyWizard):
    """Data extracted from a single PDF page.

    Attributes:
        page_number: The page number (1-indexed)
        bbox: The bounding box of the entire page (page coordinate space).
        elements: Flat list of all elements on the page
    """

    class _(JSONPyWizard.Meta):
        # Enable auto-tagging for polymorphic Union types
        auto_assign_tags = True
        # Do not raise on unknown JSON keys to avoid conflicts with tag keys on
        # nested elements
        # TODO Change this to true.
        raise_on_unknown_json_key = False

    page_number: int
    bbox: BBox
    elements: List[Element]


@dataclass
class ExtractionResult(JSONPyWizard):
    """Top-level container for extracted PDF data."""

    class _(JSONPyWizard.Meta):
        auto_assign_tags = True
        # Do not raise on unknown JSON keys
        # TODO Change this to true.
        raise_on_unknown_json_key = False

    pages: List[PageData]


def _extract_text_elements(blocks: List[BlockDict]) -> List[Text]:
    """Extract text elements from a page's raw dictionary blocks.

    Args:
        blocks: List of block dictionaries from page.get_text("rawdict")["blocks"]

    Returns:
        List of Text elements
    """
    elements: List[Text] = []

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


def _extract_image_elements(blocks: List[BlockDict]) -> List[Image]:
    """Extract image elements from a page's raw dictionary blocks.

    Args:
        blocks: List of block dictionaries from page.get_text("rawdict")["blocks"]

    Returns:
        List of Image elements
    """
    elements: List[Image] = []

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


def _extract_drawing_elements(drawings: List[Any]) -> List[Drawing]:
    """Extract drawing (vector path) elements from a page.

    Args:
        drawings: List of drawing dictionaries from page.get_drawings()

    Returns:
        List of Drawing elements
    """
    elements: List[Drawing] = []

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
    typed_elements: List[Element] = []
    if "text" in include_types:
        typed_elements.extend(_extract_text_elements(blocks))
    if "image" in include_types:
        typed_elements.extend(_extract_image_elements(blocks))
    if "drawing" in include_types:
        drawings = page.get_drawings()
        typed_elements.extend(_extract_drawing_elements(drawings))

    # Assign sequential IDs to elements within this page. PageElement dataclasses
    # are frozen, so we create new instances with the assigned id using
    # dataclasses.replace.
    for i, element in enumerate(typed_elements):
        typed_elements[i] = replace(element, id=i)

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
    page_numbers: Sequence[int] | None = None,
    include_types: Set[str] = {"text", "image", "drawing"},
) -> List[PageData]:
    """
    Extract bounding boxes for the selected pages of a PDF document.

    Args:
        doc: PyMuPDF Document object
        page_numbers: A sequence of 1-indexed page numbers to process. If None or
            empty, all pages are processed.
        include_types: Set of element types to include ("text", "image", "drawing")

    Returns:
        List of PageData containing all pages with their elements
    """
    pages: List[PageData] = []

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

        page_data = _extract_page_elements(page, page_num, include_types)
        pages.append(page_data)

    return pages
