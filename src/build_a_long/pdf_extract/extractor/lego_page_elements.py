from abc import ABC
from enum import Enum
from typing import Annotated, Literal

from annotated_types import Ge, Gt
from pydantic import BaseModel, ConfigDict, Discriminator, Field

from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Drawing


class _LegoPageElement(BaseModel, ABC):
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
    model_config = ConfigDict(populate_by_name=True)

    bbox: BBox

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"{self.__class__.__name__}(bbox={str(self.bbox)})"


class PageNumber(_LegoPageElement):
    """The page number, usually a small integer on the page.

    Positional context: Typically located in the lower-left or lower-right corner
    of the page.

    See layout diagram: lego_page_layout.png
    """

    tag: Literal["PageNumber"] = Field(
        default="PageNumber", alias="__tag__", frozen=True
    )
    value: Annotated[int, Ge(0)]

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"PageNumber(value={self.value})"


class StepNumber(_LegoPageElement):
    """A step number label.

    Positional context: Located below the PartsList within a Step, left-aligned
    with the PartsList container.

    See layout diagram: lego_page_layout.png
    """

    tag: Literal["StepNumber"] = Field(
        default="StepNumber", alias="__tag__", frozen=True
    )
    value: Annotated[int, Gt(0)]

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"StepNumber(value={self.value})"


class PartCount(_LegoPageElement):
    """The visual count label associated with a part entry (e.g., '2x').

    Positional context: Positioned directly below the corresponding part image/diagram,
    left-aligned with the part image.

    See layout diagram: lego_page_layout.png
    """

    tag: Literal["PartCount"] = Field(default="PartCount", alias="__tag__", frozen=True)
    count: Annotated[int, Ge(0)]

    # TODO We may wish to add the part this count refers to.

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"PartCount(count={self.count}x)"


class Part(_LegoPageElement):
    """A single part entry within a parts list.

    Positional context: The part image/diagram appears first, with the PartCount
    label positioned directly below it, both left-aligned.

    See layout diagram: lego_page_layout.png
    """

    tag: Literal["Part"] = Field(default="Part", alias="__tag__", frozen=True)
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


class PartsList(_LegoPageElement):
    """A container of multiple parts for the page's parts list.

    Positional context: Contained within a Step. Located
    at the top of the step area, typically on the left side. Individual parts are
    arranged with their images first, followed by their count labels below.

    See layout diagram: lego_page_layout.png
    """

    tag: Literal["PartsList"] = Field(default="PartsList", alias="__tag__", frozen=True)
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


class BagNumber(_LegoPageElement):
    """The bag number, usually a small integer on the page."""

    tag: Literal["BagNumber"] = Field(default="BagNumber", alias="__tag__", frozen=True)
    value: Annotated[int, Gt(0)]

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"BagNumber(value={self.value})"


class NewBag(_LegoPageElement):
    """The graphic showing a new bag icon on the page."""

    tag: Literal["NewBag"] = Field(default="NewBag", alias="__tag__", frozen=True)
    bag: BagNumber

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"NewBag(bag={self.bag.value})"


class Diagram(_LegoPageElement):
    """The graphic showing how to complete the step.

    Positional context: The main diagram is positioned on the right side of the
    step area, occupying most of the horizontal space. It shows the assembly
    instructions for the step.

    See layout diagram: lego_page_layout.png
    """

    tag: Literal["Diagram"] = Field(default="Diagram", alias="__tag__", frozen=True)

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"Diagram(bbox={str(self.bbox)})"


class Step(_LegoPageElement):
    """A single instruction step on the page.

    Positional context: Steps are arranged vertically on the page, typically 1-2
    per page. Within each step:
    - PartsList is at the top-left
    - StepNumber is below the PartsList (left-aligned)
    - Main Diagram is on the right side, taking most of the space

    See layout diagram: lego_page_layout.png
    """

    tag: Literal["Step"] = Field(default="Step", alias="__tag__", frozen=True)

    step_number: StepNumber
    parts_list: PartsList
    diagram: Diagram  # TODO maybe this should be a list?

    # TODO add other interesting callouts (such as rotate the element)

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return (
            f"Step(number={self.step_number.value}, parts={len(self.parts_list.parts)})"
        )

    @property
    def value(self) -> int:
        """Return the step number value for convenience."""
        return self.step_number.value


class Page(_LegoPageElement):
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

    tag: Literal["Page"] = Field(default="Page", alias="__tag__", frozen=True)
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


LegoPageElement = Annotated[
    PageNumber
    | StepNumber
    | PartCount
    | Part
    | PartsList
    | BagNumber
    | NewBag
    | Diagram
    | Step
    | Page,
    Discriminator("tag"),
]


# TODO Add sub-assembly (or sub-step) element.
# TODO Add a final preview element.
# TODO Add a "information" element (for facts about the set).
# TODO Maybe add a progress bar element.
# TODO Add part number (for use on the parts list page at the back of the book).
