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
from typing import Dict, List, Optional, Union

from build_a_long.bounding_box_extractor.extractor.bbox import BBox


@dataclass(eq=False)
class PageElement:
    """Base class for anything detected on a page.

    Contract:
    - Every element has exactly one bounding box in page coordinates
      (same coordinate system produced by the extractor).
    - Subclasses are small data holders.
    - label: The assigned classification label (e.g., 'page_number', 'step_number').
    - label_scores: Probability scores (0.0 to 1.0) for each potential label.
    """

    bbox: BBox
    id: Optional[int] = field(default=None, kw_only=True)
    # Classification fields
    label: Optional[str] = field(default=None, kw_only=True)
    label_scores: Dict[str, float] = field(default_factory=dict, kw_only=True)

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


# Note: Previously, a synthetic Root element represented the full page bounds.
# We now model the page bounds on PageData.bbox instead and omit any synthetic root element.


###
### Lego-specific elements ###
###


@dataclass(eq=False)
class StepNumber(PageElement):
    """A step number label, usually a small integer on the page."""

    value: int


@dataclass(eq=False)
class PartCount(PageElement):
    """The visual count label associated with a part entry (e.g., 'x3')."""

    count: int

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("PartCount.count must be non-negative")


@dataclass(eq=False)
class Part(PageElement):
    """A single part entry within a parts list.

    name/number are optional metadata fields should we later OCR them.
    The count is modeled as its own element to keep a consistent
    'one element, one bbox' rule.
    """

    name: Optional[str]
    number: Optional[str]
    count: PartCount


@dataclass(eq=False)
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


# A helpful alias if callers want to store a heterogeneous collection
Element = Union[PartsList, Part, PartCount, StepNumber, Drawing, Text, Image]
