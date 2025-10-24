"""
Typed data model for representing nested page elements detected by the
bounding box extractor.

Each instance represents exactly one visual thing on the page and owns a
single bounding box. Complex structures are represented hierarchically, e.g.
PartsList -> Part -> PartCount.

These classes are intentionally small, immutable dataclasses with rich type
hints to keep them easy to test and reason about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

from build_a_long.bounding_box_extractor.extractor.bbox import BBox


@dataclass
class PageElement:
    """Base class for anything detected on a page.

    Contract:
    - Every element has exactly one bounding box in page coordinates
      (same coordinate system produced by the extractor).
    - Subclasses are small data holders.
    """

    bbox: BBox


@dataclass
class Drawing(PageElement):
    """A vector drawing on the page.

    image_id can be used to tie back to a raster extracted by the pipeline
    when/if available.
    """

    image_id: Optional[str] = None
    children: List["Element"] = field(default_factory=list)


@dataclass
class Text(PageElement):
    """A text element on the page.

    Stores the actual text content extracted from the PDF.
    """

    text: str
    label: Optional[str] = None  # e.g., 'parts_list', 'instruction', etc.
    children: List["Element"] = field(default_factory=list)


@dataclass
class Image(PageElement):
    """An image element on the page (raster image from PDF).

    image_id can be used to tie back to a raster extracted by the pipeline.
    """

    image_id: Optional[str] = None
    children: List["Element"] = field(default_factory=list)


@dataclass
class Root(PageElement):
    """A root element that contains all top-level elements on a page.

    The bbox encompasses the entire page bounds.
    """

    children: List["Element"] = field(default_factory=list)


###
### Lego-specific elements ###
###


@dataclass
class StepNumber(PageElement):
    """A step number label, usually a small integer on the page."""

    value: int
    children: List["Element"] = field(default_factory=list)


@dataclass
class PartCount(PageElement):
    """The visual count label associated with a part entry (e.g., 'x3')."""

    count: int
    children: List["Element"] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("PartCount.count must be non-negative")


@dataclass
class Part(PageElement):
    """A single part entry within a parts list.

    name/number are optional metadata fields should we later OCR them.
    The count is modeled as its own element to keep a consistent
    'one element, one bbox' rule.
    """

    name: Optional[str]
    number: Optional[str]
    count: PartCount
    children: List["Element"] = field(default_factory=list)


@dataclass
class PartsList(PageElement):
    """A container of multiple parts for the page's parts list."""

    parts: List[Part]
    children: List["Element"] = field(default_factory=list)

    @property
    def total_items(self) -> int:
        """Total number of individual items accounting for counts.

        Example: if the list contains Part(count=2) and Part(count=5), this
        returns 7.
        """

        return sum(p.count.count for p in self.parts)


# A helpful alias if callers want to store a heterogeneous collection
Element = Union[PartsList, Part, PartCount, StepNumber, Drawing, Text, Image, Root]
