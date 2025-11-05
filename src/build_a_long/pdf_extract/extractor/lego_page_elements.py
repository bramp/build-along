from dataclasses import dataclass, field

from dataclass_wizard import JSONPyWizard

from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.extractor import PageData
from build_a_long.pdf_extract.extractor.page_elements import Drawing


@dataclass
class LegoPageElement(JSONPyWizard):
    """Base class for anything detected on a page.

    Contract:
    - Every element has exactly one bounding box in page coordinates
      (same coordinate system produced by the extractor).
    - Subclasses are small data holders.
    - Inherits from JSONPyWizard to get automatic to_dict(), to_json(),
      from_dict(), from_json() methods.
    - Uses auto_assign_tags to add __tag__ field for polymorphic serialization.
    """

    class _(JSONPyWizard.Meta):
        # Enable auto-tagging for polymorphic serialization
        auto_assign_tags = True
        # Exclude page_data from serialization to avoid circular references
        dump_exclude = {"page_data"}

    bbox: BBox

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"{self.__class__.__name__}(bbox={str(self.bbox)})"


@dataclass
class PageNumber(LegoPageElement):
    """The page number, usually a small integer on the page."""

    value: int = field(kw_only=True)

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("PageNumber.value must be non-negative")

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"PageNumber(value={self.value})"


@dataclass
class StepNumber(LegoPageElement):
    """A step number label."""

    value: int = field(kw_only=True)

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError("StepNumber.value must be positive")

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"StepNumber(value={self.value})"


@dataclass
class PartCount(LegoPageElement):
    """The visual count label associated with a part entry (e.g., '2x')."""

    count: int = field(kw_only=True)

    # TODO We may wish to add the part this count refers to.

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("PartCount.count must be non-negative")

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"PartCount(count={self.count})"


@dataclass
class Part(LegoPageElement):
    """A single part entry within a parts list."""

    count: PartCount = field(kw_only=True)
    diagram: Drawing | None = field(
        default=None, kw_only=True
    )  # TODO Make this required

    # Name and Number are not directly extracted, but may be filled in later
    name: str | None = field(default=None, kw_only=True)
    number: str | None = field(default=None, kw_only=True)

    # TODO maybe add color?
    # TODO Some parts have a "shiny" highlight - maybe reference that image

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        name_str = f'"{self.name}"' if self.name else "unnamed"
        number_str = self.number if self.number else "no-number"
        return f"Part(count={self.count.count}x, name={name_str}, number={number_str})"


@dataclass
class PartsList(LegoPageElement):
    """A container of multiple parts for the page's parts list."""

    parts: list[Part] = field(kw_only=True)

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


@dataclass
class BagNumber(LegoPageElement):
    """The bag number, usually a small integer on the page."""

    value: int = field(kw_only=True)

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError("BagNumber.value must be positive")

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"BagNumber(value={self.value})"


@dataclass
class NewBag(LegoPageElement):
    """The graphic showing a new bag icon on the page."""

    bag: BagNumber = field(kw_only=True)

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"NewBag(bag={self.bag.value})"


@dataclass
class Diagram(LegoPageElement):
    """The graphic showing how to complete the step."""

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"Diagram(bbox={str(self.bbox)})"


@dataclass
class Step(LegoPageElement):
    """A single instruction step on the page."""

    step_number: StepNumber = field(kw_only=True)
    parts_list: PartsList = field(kw_only=True)
    diagram: Diagram = field(kw_only=True)  # TODO maybe this should be a list?

    # TODO add other interesting callouts (such as rotate the element)

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return (
            f"Step(number={self.step_number.value}, parts={len(self.parts_list.parts)})"
        )


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

    # TODO Consider if we want to keep the page_data field here. It creates a circular
    # reference, and is not strictly necessary for the final LegoPageElement
    # representation.
    page_data: PageData = field(kw_only=True)

    page_number: PageNumber | None = field(default=None, kw_only=True)
    steps: list[Step] = field(default_factory=list, kw_only=True)
    parts_lists: list[PartsList] = field(default_factory=list, kw_only=True)

    # Metadata about the conversion process
    warnings: list[str] = field(default_factory=list, kw_only=True)
    # Keep reference to raw elements that weren't converted (for debugging/analysis)
    unprocessed_elements: list = field(
        default_factory=list, kw_only=True
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
