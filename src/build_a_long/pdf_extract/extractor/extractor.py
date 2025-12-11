import logging
from collections.abc import Sequence

import pymupdf
from pydantic import BaseModel

from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.drawing_utils import compute_visible_bbox

# Note: We intentionally do not build hierarchy here to avoid syncing issues
from build_a_long.pdf_extract.extractor.page_blocks import (
    Blocks,
    Drawing,
    Image,
    Text,
)
from build_a_long.pdf_extract.extractor.pymupdf_types import (
    ImageInfoDict,
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
        include_smask: bool = False,
    ) -> None:
        """Initialize the extractor for a specific page.

        Args:
            page: PyMuPDF page object to extract from.
            page_num: Page number (1-indexed) for the PageData result.
            include_metadata: If True, extract additional metadata (colors, fonts,
                dimensions, etc.). If False, only extract core fields.
            include_smask: If True, extract soft mask (alpha/transparency) xrefs
                for images. This requires an additional page.get_images() call.
                Defaults to False.
        """
        self._page = page
        self._page_num = page_num
        self._include_metadata = include_metadata
        self._include_smask = include_smask
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
            if not span.get("bbox"):
                continue

            text_block = Text.from_texttrace_span(
                span,  # type: ignore[arg-type]
                block_id=self._get_next_id(),
                include_metadata=self._include_metadata,
            )

            logger.debug(
                "Found text %r with bbox %s, font %s, size %s, draw_order %s",
                text_block.text,
                text_block.bbox,
                text_block.font_name,
                text_block.font_size,
                text_block.draw_order,
            )

            text_blocks.append(text_block)

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

            # Find draw order by matching bbox
            draw_order = tracker.find_image_draw_order(bbox)

            image_block = Image.from_image_info(
                img_info,
                block_id=self._get_next_id(),
                draw_order=draw_order,
                include_metadata=self._include_metadata,
            )

            image_blocks.append(image_block)
            logger.debug(
                "Found image %s with %s, xref=%s, draw_order=%s",
                image_block.image_id,
                image_block.bbox,
                image_block.xref,
                image_block.draw_order,
            )

        # Optionally get smask info from page.get_images() and update the image blocks
        # page.get_images() returns: (xref, smask, width, height, bpc, colorspace, ...)
        if self._include_smask and image_blocks:
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
                    img = img.with_smask(xref_to_smask[img.xref])
                updated_blocks.append(img)
            image_blocks = updated_blocks

        return image_blocks

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

            # Compute visible bbox considering clipping
            visible_bbox: BBox | None = None
            if "level" in d:  # Only available with extended=True
                level = d.get("level", 0)
                nbbox = BBox.from_tuple((drect.x0, drect.y0, drect.x1, drect.y1))
                visible_bbox = compute_visible_bbox(nbbox, level, drawings, idx)  # type: ignore[arg-type]
                logger.debug(
                    "Drawing %d at level %d: bbox=%s, visible_bbox=%s",
                    idx,
                    level,
                    nbbox,
                    visible_bbox,
                )

            try:
                drawing_block = Drawing.from_drawing_dict(
                    d,  # type: ignore[arg-type]
                    block_id=self._get_next_id(),
                    visible_bbox=visible_bbox,
                    include_metadata=self._include_metadata,
                )
            except ValueError as e:
                logger.warning("Drawing at index %d: %s, skipping", idx, e)
                continue

            drawing_blocks.append(drawing_block)
            logger.debug(
                "Found drawing with bbox=%s, visible_bbox=%s, draw_order=%s",
                drawing_block.unclipped_bbox,
                drawing_block.bbox if drawing_block.is_clipped else None,
                drawing_block.draw_order,
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
