"""
Typed data model for representing raw blocks extracted from PDF.

Each block represents exactly one primitive visual element (text, image, or drawing)
extracted from the PDF. These are the raw building blocks before classification.

For LEGO-specific structured elements (Part, PartCount, PartsList, StepNumber, etc.)
that are constructed by classifiers, see lego_page_elements.py.

These classes are intentionally small, immutable dataclasses with rich type
hints to keep them easy to test and reason about.
"""

from __future__ import annotations

from abc import ABC
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    field_serializer,
    field_validator,
)

from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.drawing_utils import convert_drawing_items
from build_a_long.pdf_extract.extractor.pymupdf_types import (
    DrawingDict,
    ImageInfoDict,
    PointLikeTuple,
    RectLikeTuple,
    TexttraceSpanDict,
    TransformTuple,
)
from build_a_long.pdf_extract.utils import SerializationMixin


class Block(SerializationMixin, BaseModel, ABC):
    """Base class for raw blocks extracted from PDF.

    Contract:
    - Every block has exactly one bounding box in page coordinates
      (same coordinate system produced by the extractor).
    - Every block must have a unique ID assigned by the Extractor.
    - Subclasses are small data holders.

    Blocks are frozen (immutable) and hashable based on field values,
    allowing use as dict keys. Two blocks with identical field values
    are considered equal.
    """

    model_config = ConfigDict(
        frozen=True,  # pyright: ignore[reportUnhashable] - Pyright doesn't recognize Pydantic frozen models or discriminated unions as hashable
    )

    bbox: BBox
    id: int
    """Unique block ID assigned by the Extractor."""

    draw_order: int | None = None
    """Index in the PDF's rendering order from get_bboxlog().

    This represents when this block was drawn relative to other blocks on the page.
    Lower values mean the block was drawn earlier (further back in z-order).
    None means the draw order could not be determined (e.g., block not found
    in bboxlog).
    """


class Drawing(Block):
    """A vector drawing block on the page.

    Stores metadata about vector paths including colors, line styles,
    and rendering properties.
    """

    tag: Literal["Drawing"] = Field(default="Drawing", alias="__tag__", frozen=True)

    fill_color: tuple[float, ...] | None = None  # fill color (RGB or CMYK)
    stroke_color: tuple[float, ...] | None = None  # stroke color (RGB or CMYK)
    line_width: float | None = None  # stroke line width
    fill_opacity: float | None = None  # fill transparency (0-1)
    stroke_opacity: float | None = None  # stroke transparency (0-1)
    path_type: str | None = None  # path type: "f", "s", "fs", "clip", "group"
    dashes: str | None = None  # dashed line specification
    even_odd: bool | None = None  # fill behavior for overlaps
    items: tuple[tuple, ...] | None = None  # list of draw commands
    original_bbox: BBox | None = None  # original bbox before clipping (only if clipped)

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    @property
    def unclipped_bbox(self) -> BBox:
        """Return the original unclipped bounding box.

        If the drawing was clipped, returns original_bbox.
        Otherwise, returns bbox (which is the original since no clipping occurred).
        """
        return self.original_bbox if self.original_bbox is not None else self.bbox

    @property
    def is_clipped(self) -> bool:
        """Return True if this drawing was clipped.

        Compares the visible bbox to the original bbox.
        """
        return self.original_bbox is not None and self.bbox != self.original_bbox

    @classmethod
    def from_drawing_dict(
        cls,
        d: DrawingDict,
        block_id: int,
        visible_bbox: BBox | None = None,
        *,
        include_metadata: bool = False,
    ) -> Drawing:
        """Create a Drawing block from a PyMuPDF drawing dict.

        Args:
            d: A drawing dict from page.get_drawings(extended=True)
            block_id: Unique ID to assign to this block
            visible_bbox: The visible bbox after clipping (if clipping was computed).
                If None, the original bbox from the drawing is used.
            include_metadata: If True, extract additional metadata (colors, items, etc.)

        Returns:
            A new Drawing block with the extracted data

        Raises:
            ValueError: If the drawing has no 'rect' field
        """
        drect = d.get("rect")
        if not drect:
            raise ValueError("Drawing has no 'rect' field")

        nbbox = BBox.from_rect(drect)

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

        if include_metadata:
            fill_color = d.get("fill")
            stroke_color = d.get("color")
            line_width = d.get("width")
            fill_opacity = d.get("fill_opacity")
            stroke_opacity = d.get("stroke_opacity")
            path_type = d.get("type")
            dashes = d.get("dashes")
            even_odd = d.get("even_odd")
            # Convert PyMuPDF objects to tuples for JSON serialization
            items = convert_drawing_items(d.get("items"))

        # Get draw order directly from seqno (corresponds to bboxlog index)
        draw_order = d.get("seqno")

        # Determine the final bbox and original_bbox
        final_bbox = visible_bbox if visible_bbox else nbbox
        original_bbox = nbbox if visible_bbox and visible_bbox != nbbox else None

        return cls(
            bbox=final_bbox,
            fill_color=fill_color,
            stroke_color=stroke_color,
            line_width=line_width,
            fill_opacity=fill_opacity,
            stroke_opacity=stroke_opacity,
            path_type=path_type,
            dashes=dashes,
            even_odd=even_odd,
            items=items,
            original_bbox=original_bbox,
            id=block_id,
            draw_order=draw_order,
        )

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"Drawing(bbox={str(self.bbox)})"


class Text(Block):
    """A text block on the page.

    Stores the actual text content extracted from the PDF along with metadata
    about fonts, colors, and text properties.
    """

    tag: Literal["Text"] = Field(default="Text", alias="__tag__", frozen=True)
    text: str
    font_name: str | None = None
    font_size: float | None = None

    font_flags: int | None = None  # font flags bitmap (superscript, italic, etc.)
    color: int | None = None  # text color as RGB integer
    ascender: float | None = None  # font ascender
    descender: float | None = None  # font descender
    origin: PointLikeTuple | None = None  # span origin (x, y)

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    @classmethod
    def from_texttrace_span(
        cls,
        span: TexttraceSpanDict,
        block_id: int,
        *,
        include_metadata: bool = False,
    ) -> Text:
        """Create a Text block from a PyMuPDF texttrace span.

        Args:
            span: A span dict from page.get_texttrace()
            block_id: Unique ID to assign to this block
            include_metadata: If True, extract additional metadata (color, flags, etc.)

        Returns:
            A new Text block with the extracted data
        """
        bbox = span.get("bbox")
        if not bbox:
            raise ValueError("Span has no bbox")

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

        if include_metadata:
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

        return cls(
            id=block_id,
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

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        text_preview = self.text[:30] + "..." if len(self.text) > 30 else self.text
        return f'Text(bbox={str(self.bbox)}, text="{text_preview}")'


class Image(Block):
    """An image block on the page (raster image from PDF).

    Stores metadata about the image including dimensions, format, and transform.
    """

    tag: Literal["Image"] = Field(default="Image", alias="__tag__", frozen=True)
    image_id: str | None = None

    width: int | None = None  # image width in pixels
    height: int | None = None  # image height in pixels
    ext: str | None = None  # image format (jpeg, png, etc.)
    colorspace: int | None = None  # colorspace component count
    xres: int | None = None  # horizontal resolution
    yres: int | None = None  # vertical resolution
    bpc: int | None = None  # bits per component
    size: int | None = None  # uncompressed size in bytes
    transform: TransformTuple | None = None  # transformation matrix
    xref: int | None = None  # PDF cross-reference number (identifies unique image)
    smask: int | None = None  # xref of soft mask (alpha/transparency) if any
    digest: bytes | None = None  # MD5 hash for duplicate detection

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    @classmethod
    def from_image_info(
        cls,
        img_info: ImageInfoDict,
        block_id: int,
        draw_order: int | None = None,
        *,
        include_metadata: bool = False,
    ) -> Image:
        """Create an Image block from PyMuPDF image info.

        Args:
            img_info: An image info dict from page.get_image_info(xrefs=True)
            block_id: Unique ID to assign to this block
            draw_order: The draw order from bboxlog (determined by caller)
            include_metadata: If True, extract additional metadata (dimensions, etc.)

        Returns:
            A new Image block with the extracted data
        """
        bbox: RectLikeTuple = img_info.get("bbox", (0.0, 0.0, 0.0, 0.0))
        nbbox = BBox.from_tuple(bbox)

        bi = img_info.get("number")

        # Get the xref - 0 means inline image (no xref)
        xref: int | None = img_info.get("xref")
        if xref == 0:
            xref = None  # Inline images have no xref

        # Extract additional metadata if requested
        width: int | None = None
        height: int | None = None
        colorspace: int | None = None
        xres: int | None = None
        yres: int | None = None
        bpc: int | None = None
        size: int | None = None
        transform: TransformTuple | None = None
        digest: bytes | None = None

        if include_metadata:
            width = img_info.get("width")
            height = img_info.get("height")
            colorspace = img_info.get("colorspace")
            xres = img_info.get("xres")
            yres = img_info.get("yres")
            bpc = img_info.get("bpc")
            size = img_info.get("size")
            transform = img_info.get("transform")
            digest = img_info.get("digest")

        return cls(
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
            smask=None,  # smask is set separately via with_smask()
            digest=digest,
            id=block_id,
            draw_order=draw_order,
        )

    def with_smask(self, smask: int) -> Image:
        """Return a copy of this Image with the smask set.

        Args:
            smask: The xref of the soft mask

        Returns:
            A new Image with smask set
        """
        return self.model_copy(update={"smask": smask})

    @field_validator("digest", mode="before")
    @classmethod
    def _parse_digest(cls, v: str | bytes | None) -> bytes | None:
        """Parse digest from hex string (for JSON deserialization) or bytes."""
        if v is None:
            return None
        if isinstance(v, bytes):
            return v
        # Assume hex string from JSON
        return bytes.fromhex(v)

    @field_serializer("digest")
    @classmethod
    def _serialize_digest(cls, v: bytes | None) -> str | None:
        """Serialize digest bytes to hex string for JSON output."""
        return v.hex() if v is not None else None

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        image_str = f", image_id={self.image_id}" if self.image_id else ""
        return f"Image(bbox={str(self.bbox)}{image_str})"


# Discriminated union type for polymorphic deserialization
# The Discriminator allows Pydantic to deserialize JSON into the correct
# subclass based on the "tag" field
Blocks = Annotated[Drawing | Text | Image, Discriminator("tag")]
