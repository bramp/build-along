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

from dataclasses import dataclass, field

from build_a_long.pdf_extract.extractor.bbox import BBox


@dataclass(eq=False, frozen=True)
class _Block:
    """Base class for raw blocks extracted from PDF.

    Contract:
    - Every block has exactly one bounding box in page coordinates
      (same coordinate system produced by the extractor).
    - Every block must have a unique ID assigned by the Extractor.
    - Subclasses are small data holders.
    """

    bbox: BBox
    id: int = field(kw_only=True)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass(eq=False, frozen=True)
class Drawing(_Block):
    """A vector drawing block on the page.

    image_id can be used to tie back to a raster extracted by the pipeline
    when/if available.
    """

    image_id: str | None = None

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        image_str = f", image_id={self.image_id}" if self.image_id else ""
        return f"Drawing(bbox={str(self.bbox)}{image_str})"


@dataclass(eq=False, frozen=True)
class Text(_Block):
    """A text block on the page.

    Stores the actual text content extracted from the PDF.
    """

    text: str
    font_name: str | None = None
    font_size: float | None = None

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        text_preview = self.text[:30] + "..." if len(self.text) > 30 else self.text
        return f'Text(bbox={str(self.bbox)}, text="{text_preview}")'


@dataclass(eq=False, frozen=True)
class Image(_Block):
    """An image block on the page (raster image from PDF).

    image_id can be used to tie back to a raster extracted by the pipeline.
    """

    image_id: str | None = None

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        image_str = f", image_id={self.image_id}" if self.image_id else ""
        return f"Image(bbox={str(self.bbox)}{image_str})"


# A helpful alias for heterogeneous collections of page elements
Block = Drawing | Text | Image
