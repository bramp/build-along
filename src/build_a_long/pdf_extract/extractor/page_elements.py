"""
Typed data model for representing nested page elements detected by the
bounding box extractor.

Each instance represents exactly one visual thing on the page and owns a
single bounding box. Complex structures are represented hierarchically, e.g.
PartsList -> Part -> PartCount.

These classes are intentionally small, immutable dataclasses with rich type
hints to keep them easy to test and reason about.

Note: Lego-specific structured elements (StepNumber, PartCount, Part, PartsList, etc.)
are defined in lego_page_elements.py to keep this module focused on raw extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from build_a_long.pdf_extract.extractor.bbox import BBox


@dataclass(eq=False)
class PageElement:
    """Base class for anything detected on a page.

    Contract:
    - Every element has exactly one bounding box in page coordinates
      (same coordinate system produced by the extractor).
    - Subclasses are small data holders.
    - label: The assigned classification label (e.g., 'page_number', 'step_number').
    - deleted: True if this element was removed during classification (e.g., duplicate).
    """

    bbox: BBox
    id: Optional[int] = field(default=None, kw_only=True)

    # Classification fields
    label: Optional[str] = field(default=None, kw_only=True)
    deleted: bool = field(default=False, kw_only=True)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass(eq=False)
class Drawing(PageElement):
    """A vector drawing on the page.

    image_id can be used to tie back to a raster extracted by the pipeline
    when/if available.
    """

    image_id: Optional[str] = None


@dataclass(eq=False)
class Text(PageElement):
    """A text element on the page.

    Stores the actual text content extracted from the PDF.
    """

    text: str
    font_name: Optional[str] = None
    font_size: Optional[float] = None


@dataclass(eq=False)
class Image(PageElement):
    """An image element on the page (raster image from PDF).

    image_id can be used to tie back to a raster extracted by the pipeline.
    """

    image_id: Optional[str] = None


# A helpful alias for heterogeneous collections of page elements
Element = Union[Drawing, Text, Image]
