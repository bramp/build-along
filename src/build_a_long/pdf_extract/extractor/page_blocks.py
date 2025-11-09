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

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field

from build_a_long.pdf_extract.extractor.bbox import BBox


class _Block(BaseModel):
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


class Drawing(_Block):
    """A vector drawing block on the page.

    image_id can be used to tie back to a raster extracted by the pipeline
    when/if available.
    """

    tag: Literal["Drawing"] = Field(default="Drawing", alias="__tag__", frozen=True)
    image_id: str | None = None

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        image_str = f", image_id={self.image_id}" if self.image_id else ""
        return f"Drawing(bbox={str(self.bbox)}{image_str})"


class Text(_Block):
    """A text block on the page.

    Stores the actual text content extracted from the PDF.
    """

    tag: Literal["Text"] = Field(default="Text", alias="__tag__", frozen=True)
    text: str
    font_name: str | None = None
    font_size: float | None = None

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        text_preview = self.text[:30] + "..." if len(self.text) > 30 else self.text
        return f'Text(bbox={str(self.bbox)}, text="{text_preview}")'


class Image(_Block):
    """An image block on the page (raster image from PDF).

    image_id can be used to tie back to a raster extracted by the pipeline.
    """

    tag: Literal["Image"] = Field(default="Image", alias="__tag__", frozen=True)
    image_id: str | None = None

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        image_str = f", image_id={self.image_id}" if self.image_id else ""
        return f"Image(bbox={str(self.bbox)}{image_str})"


# Discriminated union type for polymorphic deserialization
# The Discriminator allows Pydantic to deserialize JSON into the correct
# subclass based on the "tag" field
Block = Annotated[Drawing | Text | Image, Discriminator("tag")]
