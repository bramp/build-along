import logging
from collections.abc import Sequence

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
    ImageInfoDict,
    PointLikeTuple,
    RectLikeTuple,
)
from build_a_long.pdf_extract.utils import SerializationMixin

logger = logging.getLogger("extractor")


# Map bboxlog types to our block types (only images need bbox matching)
BBOXLOG_IMAGE_TYPES = frozenset({"fill-image"})


class BBoxLogTracker:
    """Tracks bboxlog entries and matches them to blocks.

    This class is used to determine the draw order of image blocks by matching
    them to entries in PyMuPDF's get_bboxlog() output. Text and drawing blocks
    get their draw order directly from seqno fields in get_texttrace() and
    get_drawings() respectively.
    """

    def __init__(
        self,
        bboxlog: list[tuple[str, tuple[float, float, float, float]]],
    ):
        """Initialize with bboxlog entries.

        Args:
            bboxlog: List of (type, bbox) tuples from page.get_bboxlog()
        """
        # Pre-index image entries for faster lookup
        self._image_entries: list[tuple[int, tuple[float, float, float, float]]] = []

        for idx, (entry_type, bbox) in enumerate(bboxlog):
            if entry_type in BBOXLOG_IMAGE_TYPES:
                self._image_entries.append((idx, bbox))

    def find_image_draw_order(
        self,
        bbox: tuple[float, float, float, float],
        tolerance: float = 0.5,
    ) -> int | None:
        """Find draw order for an image by matching bbox to bboxlog entries.

        Args:
            bbox: The bounding box to match (x0, y0, x1, y1)
            tolerance: Maximum difference allowed for bbox matching.

        Returns:
            The draw order (bboxlog index), or None if no match found
        """
        for idx, entry_bbox in self._image_entries:
            if all(abs(bbox[i] - entry_bbox[i]) < tolerance for i in range(4)):
                return idx
        return None


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


class ExtractionResult(SerializationMixin, BaseModel):
    """Top-level container for extracted PDF data."""

    pages: list[PageData]


class Extractor:
    """Handles extraction of elements from a single PDF page.

    Example usage:
        extractor = Extractor(page, page_num=1)
        text_blocks = extractor.extract_text_blocks()
        image_blocks = extractor.extract_image_blocks()
        # Or extract all at once:
        page_data = extractor.extract_page_data()
    """

    def __init__(
        self,
        page: pymupdf.Page,
        page_num: int,  # TODO I wonder if page_num is a property on page.
        *,
        include_metadata: bool = False,
    ) -> None:
        """Initialize the extractor for a specific page.

        Args:
            page: PyMuPDF page object to extract from.
            page_num: Page number (1-indexed) for the PageData result.
            include_metadata: If True, extract additional metadata (colors, fonts,
                dimensions, etc.). If False, only extract core fields.
        """
        self._page = page
        self._page_num = page_num
        self._include_metadata = include_metadata
        self._next_id = 0

        # Lazy-initialized BBoxLogTracker for draw order
        self._bboxlog_tracker: BBoxLogTracker | None = None

    @property
    def page(self) -> pymupdf.Page:
        """The PyMuPDF page this extractor is bound to."""
        return self._page

    @property
    def page_num(self) -> int:
        """The 1-indexed page number."""
        return self._page_num

    def _get_bboxlog_tracker(self) -> BBoxLogTracker:
        """Get or create the cached BBoxLogTracker.

        The tracker is lazily created on first use and cached for reuse
        across multiple block extractions.
        """
        if self._bboxlog_tracker is None:
            bboxlog = self._page.get_bboxlog()
            self._bboxlog_tracker = BBoxLogTracker(bboxlog)
        return self._bboxlog_tracker

    # TODO Maybe the ID should be based on the draw_order (which is also unique)
    def _get_next_id(self) -> int:
        """Get the next sequential ID and increment the counter."""
        current_id = self._next_id
        self._next_id += 1
        return current_id

    def reset_ids(self) -> None:
        """Reset the ID counter to 0.

        Call this if you need to re-extract with fresh IDs.
        """
        self._next_id = 0

    def _extract_text_blocks_from_texttrace(self, texttrace: list[dict]) -> list[Text]:
        """Extract text blocks from texttrace spans.

        Uses get_texttrace() which provides text in render order with seqno
        that directly corresponds to bboxlog index for draw_order.

        Args:
            texttrace: List of span dicts from page.get_texttrace()

        Returns:
            List of Text blocks with assigned IDs
        """
        text_blocks: list[Text] = []

        for span in texttrace:
            bbox = span.get("bbox")
            if not bbox:
                continue

            nbbox = BBox.from_tuple(bbox)

            # Assemble text from chars array
            # Each char is (unicode, glyph_id, origin, bbox)
            chars = span.get("chars", [])
            text = "".join(chr(c[0]) for c in chars)

            font_size: float | None = span.get("size")
            font_name: str | None = span.get("font")

            # seqno directly gives us the draw_order (bboxlog index)
            draw_order = span.get("seqno")

            # Extract additional metadata if requested
            font_flags: int | None = None
            color: int | None = None
            ascender: float | None = None
            descender: float | None = None
            origin: PointLikeTuple | None = None

            if self._include_metadata:
                font_flags = span.get("flags")
                # texttrace returns color as RGB tuple (0-1 range)
                # Convert to int like get_text("dict") does
                raw_color = span.get("color")
                if isinstance(raw_color, tuple) and len(raw_color) >= 3:
                    r, g, b = raw_color[:3]
                    color = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
                ascender = span.get("ascender")
                descender = span.get("descender")
                # Get origin from first char if available
                if chars:
                    origin = chars[0][2]  # origin is 3rd element of char tuple

            logger.debug(
                "Found text %r with bbox %s, font %s, size %s, draw_order %s",
                text,
                nbbox,
                font_name,
                font_size,
                draw_order,
            )

            text_blocks.append(
                Text(
                    id=self._get_next_id(),
                    bbox=nbbox,
                    draw_order=draw_order,
                    text=text,
                    font_name=font_name,
                    font_size=font_size,
                    font_flags=font_flags,
                    color=color,
                    ascender=ascender,
                    descender=descender,
                    origin=origin,
                )
            )

        return text_blocks

    def extract_image_blocks(self) -> list[Image]:
        """Extract image blocks from the page using get_image_info.

        This method uses page.get_image_info(xrefs=True) which provides:
        - xref: PDF cross-reference number for identifying unique/reused images
        - digest: MD5 hash for duplicate detection
        - Full bounding box and transform information

        Returns:
            List of Image blocks with assigned IDs
        """
        image_blocks: list[Image] = []
        tracker = self._get_bboxlog_tracker()

        # Use get_image_info with xrefs=True to get xref and digest (MD5 hash)
        # This is more comprehensive than extracting from rawdict
        image_infos: list[ImageInfoDict] = self._page.get_image_info(xrefs=True)  # type: ignore[assignment]

        for img_info in image_infos:
            bbox: RectLikeTuple = img_info.get("bbox", (0.0, 0.0, 0.0, 0.0))
            nbbox = BBox.from_tuple(bbox)

            bi = img_info.get("number")

            # Find draw order by matching bbox
            draw_order = tracker.find_image_draw_order(bbox)

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
                    draw_order=draw_order,
                )
            )
            logger.debug(
                "Found image %s with %s, xref=%s, draw_order=%s",
                bi,
                nbbox,
                xref,
                draw_order,
            )

        # Now get smask info from page.get_images() and update the image blocks
        # page.get_images() returns: (xref, smask, width, height, bpc, colorspace, ...)
        if image_blocks:
            page_images = self._page.get_images(full=True)
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

    def extract_drawing_blocks(self) -> list[Drawing]:
        """Extract drawing (vector path) blocks from the page.

        Returns:
            List of Drawing blocks with assigned IDs
        """
        # Use extended=True to get clipping hierarchy
        drawings = self._page.get_drawings(extended=True)

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

            # Get draw order directly from seqno (corresponds to bboxlog index)
            draw_order = d.get("seqno")

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
                    draw_order=draw_order,
                )
            )
            logger.debug(
                "Found drawing with bbox=%s, visible_bbox=%s, draw_order=%s",
                nbbox,
                visible_bbox,
                draw_order,
            )

        return drawing_blocks

    def extract_text_blocks(self) -> list[Text]:
        """Extract text blocks from the page.

        Uses get_texttrace() which provides text in render order with seqno
        for direct draw_order mapping.

        Returns:
            List of Text blocks with assigned IDs
        """
        # Use get_texttrace() for text extraction - it provides:
        # - Text in render order (not reading order)
        # - seqno field that directly maps to bboxlog index for draw_order
        # - All font/style information we need
        texttrace = self._page.get_texttrace()
        return self._extract_text_blocks_from_texttrace(texttrace)

    def extract_page_data(
        self,
        include_types: set[str] | None = None,
    ) -> PageData:
        """Extract all blocks from the page.

        Args:
            include_types: Set of element types to include ("text", "image",
                "drawing"). If None, defaults to all types.

        Returns:
            PageData with all extracted blocks (with sequential IDs)
        """
        include_types = include_types or {"text", "image", "drawing"}
        logger.debug("Processing page %s", self._page_num)

        # Extract blocks by type (IDs are assigned during creation)
        typed_blocks: list[Blocks] = []

        if "text" in include_types:
            typed_blocks.extend(self.extract_text_blocks())

        if "image" in include_types:
            typed_blocks.extend(self.extract_image_blocks())

        if "drawing" in include_types:
            typed_blocks.extend(self.extract_drawing_blocks())

        page_rect = self._page.rect
        page_bbox = BBox.from_tuple(
            (page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1)
        )

        return PageData(
            page_number=self._page_num,
            blocks=typed_blocks,
            bbox=page_bbox,
        )


def extract_page_data(
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
    pages: list[PageData] = []

    num_pages = len(doc)
    include_types = include_types or {"text", "image", "drawing"}

    # If not provided, process all pages (1-indexed numbers)
    if not page_numbers:
        page_numbers = [i + 1 for i in range(num_pages)]

    for page_num in page_numbers:
        page_index = page_num - 1  # convert to 0-indexed
        if page_index < 0 or page_index >= num_pages:
            # Out of bounds; skip defensively
            continue
        page = doc[page_index]

        # Create Extractor for this page
        extractor = Extractor(
            page=page,
            page_num=page_num,
            include_metadata=include_metadata,
        )
        page_data = extractor.extract_page_data(include_types=include_types)
        pages.append(page_data)

    return pages
