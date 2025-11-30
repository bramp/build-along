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
from build_a_long.pdf_extract.extractor.pymupdf_types import (
    PointLikeTuple,
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
