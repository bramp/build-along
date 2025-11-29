import logging
from collections.abc import Sequence
from typing import Any

import pymupdf
from pydantic import BaseModel

from build_a_long.pdf_extract.extractor.bbox import BBox

# Note: We intentionally do not build hierarchy here to avoid syncing issues
from build_a_long.pdf_extract.extractor.page_blocks import (
    Blocks,
    Drawing,
    Image,
    Text,
)
from build_a_long.pdf_extract.extractor.pymupdf_types import (
    DrawingDict,
    ImageBlockDict,
    ImageInfoDict,
    PointLikeTuple,
    RawDict,
    RectLikeTuple,
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
    blocks: list[Blocks]


class ExtractionResult(BaseModel):
    """Top-level container for extracted PDF data."""

    pages: list[PageData]


class Extractor:
    """Handles extraction of page elements with sequential ID assignment.

    Each Extractor instance should be used for a single page to ensure
    IDs are sequential and reset for each page.
    """

    def __init__(self, *, include_metadata: bool = False) -> None:
        """Initialize the extractor with a fresh ID counter.

        Args:
            include_metadata: If True, extract additional metadata (colors, fonts,
                dimensions, etc.). If False, only extract core fields.
        """
        self._next_id = 0
        self._include_metadata = include_metadata

    def _get_next_id(self) -> int:
        """Get the next sequential ID and increment the counter."""
        current_id = self._next_id
        self._next_id += 1
        return current_id

    def _extract_text_blocks(
        self, blocks: list[TextBlockDict | ImageBlockDict]
    ) -> list[Text]:
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
                    sbbox: RectLikeTuple = span.get("bbox", (0.0, 0.0, 0.0, 0.0))
                    nbbox = BBox.from_tuple(sbbox)

                    text: str = span.get("text", None)
                    if text is None or text == "":
                        chars = span.get("chars", [])
                        text = "".join(c["c"] for c in chars)

                    font_size: float = span.get("size", 0.0)
                    font_name: str = span.get("font", "unknown")

                    # Extract additional metadata if requested
                    font_flags: int | None = None
                    color: int | None = None
                    ascender: float | None = None
                    descender: float | None = None
                    origin: PointLikeTuple | None = None

                    if self._include_metadata:
                        font_flags = span.get("flags")
                        color = span.get("color")
                        ascender = span.get("ascender")
                        descender = span.get("descender")
                        origin = span.get("origin")

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
                            font_flags=font_flags,
                            color=color,
                            ascender=ascender,
                            descender=descender,
                            origin=origin,
                            id=self._get_next_id(),
                        )
                    )

        return text_blocks

    def _extract_image_blocks(self, page: pymupdf.Page) -> list[Image]:
        """Extract image blocks from a page using get_image_info.

        This method uses page.get_image_info(xrefs=True) which provides:
        - xref: PDF cross-reference number for identifying unique/reused images
        - digest: MD5 hash for duplicate detection
        - Full bounding box and transform information

        Args:
            page: PyMuPDF page object

        Returns:
            List of Image blocks with assigned IDs
        """
        image_blocks: list[Image] = []

        # Use get_image_info with xrefs=True to get xref and digest (MD5 hash)
        # This is more comprehensive than extracting from rawdict
        image_infos: list[ImageInfoDict] = page.get_image_info(xrefs=True)  # type: ignore[assignment]

        for img_info in image_infos:
            bbox: RectLikeTuple = img_info.get("bbox", (0.0, 0.0, 0.0, 0.0))
            nbbox = BBox.from_tuple(bbox)

            # Get the xref - 0 means inline image (no xref)
            xref: int | None = img_info.get("xref")
            if xref == 0:
                xref = None  # Inline images have no xref

            # Get smask (soft mask) xref if present
            # We need to look this up from page.get_images() for the full info
            smask: int | None = None

            # Extract additional metadata if requested
            width: int | None = None
            height: int | None = None
            colorspace: int | None = None
            xres: int | None = None
            yres: int | None = None
            bpc: int | None = None
            size: int | None = None
            transform: tuple[float, float, float, float, float, float] | None = None
            digest: bytes | None = None

            if self._include_metadata:
                width = img_info.get("width")
                height = img_info.get("height")
                colorspace = img_info.get("colorspace")
                xres = img_info.get("xres")
                yres = img_info.get("yres")
                bpc = img_info.get("bpc")
                size = img_info.get("size")
                transform = img_info.get("transform")
                digest = img_info.get("digest")

            bi = img_info.get("number")

            image_blocks.append(
                Image(
                    bbox=nbbox,
                    image_id=f"image_{bi}",
                    width=width,
                    height=height,
                    colorspace=colorspace,
                    xres=xres,
                    yres=yres,
                    bpc=bpc,
                    size=size,
                    transform=transform,
                    xref=xref,
                    smask=smask,
                    digest=digest,
                    id=self._get_next_id(),
                )
            )
            logger.debug("Found image %s with %s, xref=%s", bi, nbbox, xref)

        # Now get smask info from page.get_images() and update the image blocks
        # page.get_images() returns: (xref, smask, width, height, bpc, colorspace, ...)
        if image_blocks:
            page_images = page.get_images(full=True)
            xref_to_smask: dict[int, int] = {}
            for img_tuple in page_images:
                img_xref = img_tuple[0]
                img_smask = img_tuple[1]
                if img_smask > 0:
                    xref_to_smask[img_xref] = img_smask

            # Update image blocks with smask info
            updated_blocks: list[Image] = []
            for img in image_blocks:
                if img.xref is not None and img.xref in xref_to_smask:
                    # Create a new Image with smask set (since Image is frozen)
                    img = img.model_copy(update={"smask": xref_to_smask[img.xref]})
                updated_blocks.append(img)
            image_blocks = updated_blocks

        return image_blocks

    def _convert_drawing_items(
        self, items: list[tuple] | None
    ) -> tuple[tuple, ...] | None:
        """Convert PyMuPDF objects in drawing items to JSON-serializable tuples.

        Drawing items can contain Rect, Point, and Quad objects which need
        to be converted to tuples for JSON serialization.

        Args:
            items: List of drawing command tuples from PyMuPDF

        Returns:
            Tuple of tuples with PyMuPDF objects converted to tuples, or None
        """
        if items is None:
            return None

        converted_items: list[tuple] = []
        for item in items:
            if not isinstance(item, tuple) or len(item) == 0:
                converted_items.append(item)
                continue

            # Convert tuple elements that are PyMuPDF objects
            converted_elements: list = [item[0]]  # Keep the command string
            for element in item[1:]:
                # Check if element has x0, y0, x1, y1 (Rect-like)
                if hasattr(element, "x0"):
                    converted_elements.append(
                        (element.x0, element.y0, element.x1, element.y1)
                    )
                # Check if element has x, y (Point-like)
                elif hasattr(element, "x") and hasattr(element, "y"):
                    converted_elements.append((element.x, element.y))
                # Check if it's a Quad (has ul, ur, ll, lr)
                elif hasattr(element, "ul"):
                    converted_elements.append(
                        (
                            (element.ul.x, element.ul.y),
                            (element.ur.x, element.ur.y),
                            (element.ll.x, element.ll.y),
                            (element.lr.x, element.lr.y),
                        )
                    )
                else:
                    # Keep as-is if not a recognized PyMuPDF object
                    converted_elements.append(element)

            converted_items.append(tuple(converted_elements))

        return tuple(converted_items)

    def _compute_visible_bbox(
        self,
        bbox: BBox,
        level: int,
        drawings: list[DrawingDict],
        current_index: int,
    ) -> BBox:
        """Compute visible bbox by intersecting with applicable clip paths.

        Args:
            bbox: The original bounding box
            level: The hierarchy level of this drawing
            drawings: Full list of drawings (with extended=True)
            current_index: Index of current drawing in the list

        Returns:
            The visible bbox after applying all relevant clips
        """
        visible = bbox

        # Build the clip stack by walking backwards
        # A clip at level L applies to drawings at level > L
        # A clip's scope ends when we see a non-clip at level <= L
        clip_stack: dict[int, BBox] = {}  # level -> clip bbox

        for i in range(current_index - 1, -1, -1):
            prev = drawings[i]
            prev_level = prev.get("level", 0)
            prev_type = prev.get("type")

            # If we see a non-clip at or below our level, we can stop
            # (we've exited all relevant clip scopes)
            if prev_type != "clip" and prev_level < level:
                logger.debug("  Stop at drawing %d (L%d non-clip)", i, prev_level)
                break

            # If it's a clip at a level less than ours, it applies
            if prev_type == "clip" and prev_level < level:
                # Only add if we haven't seen a clip at this level yet
                # (we're walking backwards, so first encountered is most recent)
                if prev_level not in clip_stack:
                    scissor = prev.get("scissor")
                    if scissor:
                        # Skip inverted/invalid clip rectangles - they don't make
                        # geometric sense as clipping regions and are likely PDF
                        # artifacts
                        if scissor.x0 > scissor.x1 or scissor.y0 > scissor.y1:
                            logger.debug(
                                "  Skipping inverted clip from drawing %d (L%d): %s",
                                i,
                                prev_level,
                                scissor,
                            )
                            continue

                        clip_bbox = BBox.from_tuple(
                            (scissor.x0, scissor.y0, scissor.x1, scissor.y1)
                        )
                        clip_stack[prev_level] = clip_bbox
                        logger.debug(
                            "  Adding clip from drawing %d (L%d): %s",
                            i,
                            prev_level,
                            clip_bbox,
                        )

        # Apply all clips by intersecting
        for clip_bbox in clip_stack.values():
            visible = visible.intersect(clip_bbox)

        return visible

    def _extract_drawing_blocks(
        self, drawings: list[DrawingDict], page: pymupdf.Page
    ) -> list[Drawing]:
        """Extract drawing (vector path) blocks from a page.

        Args:
            drawings: List of drawing dictionaries from page.get_drawings(extended=True)
            page: The PyMuPDF page object (needed for coordinate transformation)

        Returns:
            List of Drawing blocks with assigned IDs
        """
        drawing_blocks: list[Drawing] = []
        for idx, d in enumerate(drawings):
            assert isinstance(d, dict)

            # Skip clip paths - they're not visible drawings, just clipping regions
            if d.get("type") == "clip":
                continue

            drect = d.get("rect")
            if not drect:
                logger.warning(
                    "Drawing at index %d has no 'rect' field, skipping: %s", idx, d
                )
                continue

            nbbox = BBox.from_tuple(
                (
                    drect.x0,
                    drect.y0,
                    drect.x1,
                    drect.y1,
                )
            )

            # Extract additional metadata if requested
            fill_color: tuple[float, ...] | None = None
            stroke_color: tuple[float, ...] | None = None
            line_width: float | None = None
            fill_opacity: float | None = None
            stroke_opacity: float | None = None
            path_type: str | None = None
            dashes: str | None = None
            even_odd: bool | None = None
            items: tuple[tuple, ...] | None = None

            if self._include_metadata:
                fill_color = d.get("fill")
                stroke_color = d.get("color")
                line_width = d.get("width")
                fill_opacity = d.get("fill_opacity")
                stroke_opacity = d.get("stroke_opacity")
                path_type = d.get("type")
                dashes = d.get("dashes")
                even_odd = d.get("even_odd")
                # Convert PyMuPDF objects to tuples for JSON serialization
                items = self._convert_drawing_items(d.get("items"))

            # Compute visible bbox considering clipping
            visible_bbox: BBox | None = None
            if "level" in d:  # Only available with extended=True
                level = d.get("level", 0)
                visible_bbox = self._compute_visible_bbox(nbbox, level, drawings, idx)
                logger.debug(
                    "Drawing %d at level %d: bbox=%s, visible_bbox=%s",
                    idx,
                    level,
                    nbbox,
                    visible_bbox,
                )

            drawing_blocks.append(
                Drawing(
                    bbox=visible_bbox if visible_bbox else nbbox,
                    fill_color=fill_color,
                    stroke_color=stroke_color,
                    line_width=line_width,
                    fill_opacity=fill_opacity,
                    stroke_opacity=stroke_opacity,
                    path_type=path_type,
                    dashes=dashes,
                    even_odd=even_odd,
                    items=items,
                    original_bbox=nbbox
                    if visible_bbox and visible_bbox != nbbox
                    else None,
                    id=self._get_next_id(),
                )
            )
            logger.debug(
                "Found drawing with bbox=%s, visible_bbox=%s", nbbox, visible_bbox
            )

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
        typed_blocks: list[Blocks] = []
        if "text" in include_types:
            typed_blocks.extend(self._extract_text_blocks(blocks))
        if "image" in include_types:
            typed_blocks.extend(self._extract_image_blocks(page))
        if "drawing" in include_types:
            # Use extended=True to get clipping hierarchy
            drawings = page.get_drawings(extended=True)
            typed_blocks.extend(self._extract_drawing_blocks(drawings, page))

        page_rect = page.rect
        page_bbox = BBox.from_tuple(
            (page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1)
        )

        return PageData(
            page_number=page_num,
            blocks=typed_blocks,
            bbox=page_bbox,
        )


# TODO Perhaps rename to more appropriate name
def extract_bounding_boxes(
    doc: pymupdf.Document,
    page_numbers: Sequence[int] | None = None,
    include_types: set[str] | None = None,
    *,
    include_metadata: bool = False,
) -> list[PageData]:
    """
    Extract bounding boxes for the selected pages of a PDF document.

    Args:
        doc: PyMuPDF Document object
        page_numbers: A sequence of 1-indexed page numbers to process. If None or
            empty, all pages are processed.
        include_types: Set of element types to include ("text", "image", "drawing").
            If None, defaults to all types.
        include_metadata: If True, extract additional metadata (colors, fonts,
            dimensions, etc.). If False, only extract core fields.

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
        extractor = Extractor(include_metadata=include_metadata)
        page_data = extractor.extract_page_blocks(page, page_num, include_types)
        pages.append(page_data)

    return pages
