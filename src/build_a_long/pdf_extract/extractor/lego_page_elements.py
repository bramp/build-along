from dataclasses import dataclass, field
from typing import List, Optional

from dataclasses_json import DataClassJsonMixin, config

from build_a_long.pdf_extract.extractor.bbox import BBox, _bbox_decoder


@dataclass
class LegoPageElement(DataClassJsonMixin):
    """Base class for anything detected on a page.

    Contract:
    - Every element has exactly one bounding box in page coordinates
      (same coordinate system produced by the extractor).
    - Subclasses are small data holders.
    """

    bbox: BBox = field(metadata=config(decoder=_bbox_decoder))
    id: Optional[int] = field(default=None, kw_only=True)


@dataclass
class PageNumber(LegoPageElement):
    """The page number, usually a small integer on the page."""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("PageNumber.value must be non-negative")


@dataclass
class StepNumber(LegoPageElement):
    """A step number label."""

    value: int

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError("StepNumber.value must be positive")


@dataclass
class PartCount(LegoPageElement):
    """The visual count label associated with a part entry (e.g., '2x')."""

    count: int

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("PartCount.count must be non-negative")


@dataclass
class Part(LegoPageElement):
    """A single part entry within a parts list."""

    name: Optional[str]
    number: Optional[str]

    # TODO maybe add color?
    # TODO Some parts have a "shiny" highlight - maybe reference that image

    count: PartCount


@dataclass
class PartsList(LegoPageElement):
    """A container of multiple parts for the page's parts list."""

    parts: List[Part]

    @property
    def total_items(self) -> int:
        """Total number of individual items accounting for counts.

        Example: if the list contains Part(count=2) and Part(count=5), this
        returns 7.
        """

        return sum(p.count.count for p in self.parts)


@dataclass
class BagNumber(LegoPageElement):
    """The bag number, usually a small integer on the page."""

    value: int

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError("BagNumber.value must be positive")


@dataclass
class NewBag(LegoPageElement):
    """The graphic showing a new bag icon on the page."""

    bag: BagNumber


@dataclass
class Diagram(LegoPageElement):
    """The graphic showing how to complete the step."""


@dataclass
class Step(LegoPageElement):
    """A single instruction step on the page."""

    step_number: StepNumber
    parts_list: PartsList
    diagram: Diagram  # TODO maybe this should be a list?

    # TODO add other interesting callouts (such as rotate the element)


# TODO Add sub-assembly (or sub-step) element.
# TODO Add a final preview element.
# TODO Add a "information" element (for facts about the set).
# TODO Maybe add a progress bar element.
