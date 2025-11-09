from enum import Enum
from typing import Annotated

from annotated_types import Ge, Gt
from pydantic import BaseModel, ConfigDict, Field

from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Drawing


class LegoPageElement(BaseModel):
    """Base class for LEGO-specific structured elements constructed by classifiers.

    LegoPageElements are typically constructed from one or more Blocks during
    classification and are stored in Candidate.constructed, not in PageData.elements.

    Contract:
    - Every element has exactly one bounding box in page coordinates
      (same coordinate system produced by the extractor).
    - Subclasses are small data holders.
    - Inherits from Pydantic BaseModel to get automatic model_dump(), model_dump_json(),
      model_validate(), model_validate_json() methods.
    - Uses discriminated unions to add __tag__ field for polymorphic serialization.
    """

    # Note: page_data is excluded from serialization at dump time, not in config
    model_config = ConfigDict()

    bbox: BBox

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"{self.__class__.__name__}(bbox={str(self.bbox)})"


class PageNumber(LegoPageElement):
    """The page number, usually a small integer on the page."""

    value: Annotated[int, Ge(0)]

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"PageNumber(value={self.value})"


class StepNumber(LegoPageElement):
    """A step number label."""

    value: Annotated[int, Gt(0)]

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"StepNumber(value={self.value})"


class PartCount(LegoPageElement):
    """The visual count label associated with a part entry (e.g., '2x')."""

    count: Annotated[int, Ge(0)]

    # TODO We may wish to add the part this count refers to.

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"PartCount(count={self.count}x)"


class Part(LegoPageElement):
    """A single part entry within a parts list."""

    count: PartCount
    diagram: Drawing | None = None

    # Name and Number are not directly extracted, but may be filled in later
    name: str | None = None
    number: str | None = None

    # TODO maybe add color?
    # TODO Some parts have a "shiny" highlight - maybe reference that image

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        name_str = f'"{self.name}"' if self.name else "unnamed"
        number_str = self.number if self.number else "no-number"
        return f"Part(count={self.count.count}x, name={name_str}, number={number_str})"


class PartsList(LegoPageElement):
    """A container of multiple parts for the page's parts list."""

    parts: list[Part]

    @property
    def total_items(self) -> int:
        """Total number of individual items accounting for counts.

        Example: if the list contains Part(count=2) and Part(count=5), this
        returns 7.
        """

        return sum(p.count.count for p in self.parts)

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"PartsList(parts={len(self.parts)}, total_items={self.total_items})"


class BagNumber(LegoPageElement):
    """The bag number, usually a small integer on the page."""

    value: Annotated[int, Gt(0)]

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"BagNumber(value={self.value})"


class NewBag(LegoPageElement):
    """The graphic showing a new bag icon on the page."""

    bag: BagNumber

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"NewBag(bag={self.bag.value})"


class Diagram(LegoPageElement):
    """The graphic showing how to complete the step."""

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"Diagram(bbox={str(self.bbox)})"


class Step(LegoPageElement):
    """A single instruction step on the page."""

    step_number: StepNumber
    parts_list: PartsList
    diagram: Diagram  # TODO maybe this should be a list?

    # TODO add other interesting callouts (such as rotate the element)

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return (
            f"Step(number={self.step_number.value}, parts={len(self.parts_list.parts)})"
        )


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
        unprocessed_elements: Raw elements that were classified but couldn't
            be converted
    """

    class Category(Enum):
        INFO = 1
        INSTRUCTION = 2
        CATALOG = 3

    category: Category | None = None

    page_number: PageNumber | None = None
    steps: list[Step] = Field(default_factory=list)
    parts_lists: list[PartsList] = Field(default_factory=list)

    # Metadata about the conversion process
    warnings: list[str] = Field(default_factory=list)
    # Keep reference to raw elements that weren't converted (for debugging/analysis)
    unprocessed_elements: list = Field(
        default_factory=list
    )  # List[Element] but avoiding import

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        page_num = self.page_number.value if self.page_number else "unknown"
        return (
            f"Page(number={page_num}, steps={len(self.steps)}, "
            f"parts_lists={len(self.parts_lists)}, warnings={len(self.warnings)})"
        )


# TODO Add sub-assembly (or sub-step) element.
# TODO Add a final preview element.
# TODO Add a "information" element (for facts about the set).
# TODO Maybe add a progress bar element.
# TODO Add part number (for use on the parts list page at the back of the book).
