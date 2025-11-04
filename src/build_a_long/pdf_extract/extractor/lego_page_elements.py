from dataclasses import dataclass, field
from typing import List, Optional

from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.extractor import PageData
from build_a_long.pdf_extract.extractor.page_elements import Drawing


@dataclass
class LegoPageElement:
    """Base class for anything detected on a page.

    Contract:
    - Every element has exactly one bounding box in page coordinates
      (same coordinate system produced by the extractor).
    - Subclasses are small data holders.
    """

    bbox: BBox
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

    # TODO We may wish to add the part this count refers to.

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("PartCount.count must be non-negative")


@dataclass
class Part(LegoPageElement):
    """A single part entry within a parts list."""

    count: PartCount
    diagram: Optional[Drawing] = None  # TODO Make this required

    # Name and Number are not directly extracted, but may be filled in later
    name: Optional[str] = None
    number: Optional[str] = None

    # TODO maybe add color?
    # TODO Some parts have a "shiny" highlight - maybe reference that image


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


@dataclass
class Page(LegoPageElement):
    """A complete page of LEGO instructions.

    This is the top-level element that contains all other elements on a page.
    It represents the structured, hierarchical view of the page after classification
    and hierarchy building.

    Attributes:
        page_number: The page number element, if found
        steps: List of Step elements on the page
        parts_lists: List of standalone PartsList elements (not within a Step)
        warnings: List of warnings generated during hierarchy building
        unprocessed_elements: Raw elements that were classified but couldn't be converted
    """

    page_data: PageData

    page_number: Optional[PageNumber] = None
    steps: List[Step] = field(default_factory=list)
    parts_lists: List[PartsList] = field(default_factory=list)

    # Metadata about the conversion process
    warnings: List[str] = field(default_factory=list)
    # Keep reference to raw elements that weren't converted (for debugging/analysis)
    unprocessed_elements: List = field(
        default_factory=list
    )  # List[Element] but avoiding import


# TODO Add sub-assembly (or sub-step) element.
# TODO Add a final preview element.
# TODO Add a "information" element (for facts about the set).
# TODO Maybe add a progress bar element.
# TODO Add part number (for use on the parts list page at the back of the book).
