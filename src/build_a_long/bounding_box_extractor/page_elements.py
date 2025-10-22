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

from dataclasses import dataclass
from typing import List, Optional, Union, Tuple

from build_a_long.bounding_box_extractor.bbox import BBox


@dataclass(frozen=True)
class PageElement:
    """Base class for anything detected on a page.

    Contract:
    - Every element has exactly one bounding box in page coordinates
      (same coordinate system produced by the extractor).
    - Subclasses should remain immutable and small data holders.
    """

    bbox: BBox


@dataclass(frozen=True)
class StepNumber(PageElement):
    """A step number label, usually a small integer on the page."""

    value: int


@dataclass(frozen=True)
class Drawing(PageElement):
    """A main build drawing or image on the page.

    image_id can be used to tie back to a raster extracted by the pipeline
    when/if available.
    """

    image_id: Optional[str] = None


@dataclass(frozen=True)
class PartCount(PageElement):
    """The visual count label associated with a part entry (e.g., 'x3')."""

    count: int

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.count < 0:
            raise ValueError("PartCount.count must be non-negative")


@dataclass(frozen=True)
class Part(PageElement):
    """A single part entry within a parts list.

    name/number are optional metadata fields should we later OCR them.
    The count is modeled as its own element to keep a consistent
    'one element, one bbox' rule.
    """

    name: Optional[str]
    number: Optional[str]
    count: PartCount


@dataclass(frozen=True)
class PartsList(PageElement):
    """A container of multiple parts for the page's parts list."""

    parts: List[Part]

    @property
    def total_items(self) -> int:
        """Total number of individual items accounting for counts.

        Example: if the list contains Part(count=2) and Part(count=5), this
        returns 7.
        """

        return sum(p.count.count for p in self.parts)


@dataclass(frozen=True)
class Unknown(PageElement):
    """An element with unknown semantics for now.

    This lets us construct a hierarchy today and substitute a concrete
    class later when classification is available.

    - label/raw_type: carry through the original coarse-grained type or hint
        (e.g., 'text', 'image', 'parts_list').
    - content: any OCRed text content if known.
    - source_id: original block id to aid traceability.
    - children: optional nested elements fully contained in this bbox.
    """

    label: Optional[str] = None
    raw_type: Optional[str] = None
    content: Optional[str] = None
    source_id: Optional[str] = None
    children: Tuple["Element", ...] = ()


# A helpful alias if callers want to store a heterogeneous collection
Element = Union[PartsList, Part, PartCount, StepNumber, Drawing, Unknown]
