from __future__ import annotations

from abc import ABC
from collections.abc import Iterator
from enum import Enum
from typing import Annotated, Any, Literal

from annotated_types import Ge, Gt
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    PlainSerializer,
)

from build_a_long.pdf_extract.extractor.bbox import BBox


class LegoPageElement(BaseModel, ABC):
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

    def to_dict(self, **kwargs: Any) -> dict:
        """Serialize to dict with proper defaults (by_alias=True, exclude_none=True).

        Override by passing explicit kwargs if different behavior is needed.
        """
        defaults: dict[str, Any] = {"by_alias": True, "exclude_none": True}
        defaults.update(kwargs)
        return self.model_dump(**defaults)

    def to_json(self, **kwargs: Any) -> str:
        """Serialize to JSON with proper defaults (by_alias=True, exclude_none=True).

        Override by passing explicit kwargs if different behavior is needed.
        """
        defaults: dict[str, Any] = {"by_alias": True, "exclude_none": True}
        defaults.update(kwargs)
        return self.model_dump_json(**defaults)

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"{self.__class__.__name__}(bbox={str(self.bbox)})"

    def iter_elements(self) -> Iterator[LegoPageElement]:
        """Iterate over this element and all child elements.

        Default implementation yields only self. Subclasses with children
        should override to yield self first, then recursively yield children.

        Yields:
            This element and all descendant LegoPageElements
        """
        yield self


class PageNumber(LegoPageElement):
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


class StepNumber(LegoPageElement):
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


class PartCount(LegoPageElement):
    """The visual count label associated with a part entry (e.g., '2x').

    Positional context: Positioned directly below the corresponding part image/diagram,
    left-aligned with the part image.

    See layout diagram: lego_page_layout.png
    """

    tag: Literal["PartCount"] = Field(default="PartCount", alias="__tag__", frozen=True)
    count: Annotated[int, Ge(0)]

    matched_hint: Literal["part_count", "catalog_part_count"] | None = None
    """Which font size hint was matched during classification.
    
    - 'part_count': Standard instruction page part count
    - 'catalog_part_count': Catalog/inventory page part count
    """

    # TODO We may wish to add the part this count refers to.

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        hint_str = f", {self.matched_hint}" if self.matched_hint else ""
        return f"PartCount(count={self.count}x{hint_str})"


class PartNumber(LegoPageElement):
    """The element ID number for a part (catalog pages).

    Positional context: Located directly below the part count on catalog pages.
    This is a 4-8 digit number that identifies the specific LEGO element.

    See layout diagram: lego_page_layout.png
    """

    tag: Literal["PartNumber"] = Field(
        default="PartNumber", alias="__tag__", frozen=True
    )

    element_id: str
    """The LEGO element ID (4-8 digits, never starts with zero)."""

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"PartNumber(element_id={self.element_id})"


class PieceLength(LegoPageElement):
    """The length indicator for a LEGO piece (e.g., '4' for a 4-stud beam).

    Positional context: Located in the top-right area of a part image, typically
    surrounded by a circle or oval. Uses a smaller font size than step numbers.
    Can appear on any page type (instruction, catalog, or info pages).

    This is distinct from a step number - it indicates the physical length of
    the part being shown, not a construction step.

    See layout diagram: lego_page_layout.png
    """

    tag: Literal["PieceLength"] = Field(
        default="PieceLength", alias="__tag__", frozen=True
    )

    value: Annotated[int, Gt(0)]
    """The length value (number of studs or other measurement units)."""

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"PieceLength(value={self.value})"


class PartImage(LegoPageElement):
    """A candidate image that could represent a LEGO part.

    Positional context: These images typically appear in parts lists, positioned
    above their corresponding PartCount text and left-aligned.

    This element represents a validated image candidate that will be paired with
    a PartCount by PartsClassifier to construct Part elements.

    The bbox field (inherited from LegoPageElement) defines the image region.
    """

    tag: Literal["PartImage"] = Field(default="PartImage", alias="__tag__", frozen=True)

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"PartImage(bbox={self.bbox})"


class ProgressBar(LegoPageElement):
    """A progress bar showing building progress through the instruction book.

    Positional context: Typically located at the bottom of the page, spanning most
    of the page width, near the page number. Often consists of one or more
    Drawing/Image elements forming a horizontal bar with progress indicators.

    Note: The bbox is clipped to page boundaries for display purposes, but the
    original unclipped width is preserved in full_width for progress calculation.

    See layout diagram: lego_page_layout.png
    """

    tag: Literal["ProgressBar"] = Field(
        default="ProgressBar", alias="__tag__", frozen=True
    )

    progress: float | None = None
    """Optional progress percentage (0.0 to 1.0) if detectable from the visual."""

    full_width: float
    """The original unclipped width of the progress bar, used for progress calculation.
    
    When the progress bar bbox extends beyond page boundaries (a PDF extraction
    artifact), the bbox is clipped but this field preserves the original width
    that may be semantically meaningful for calculating progress percentage.
    """

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        progress_str = f", {self.progress:.1%}" if self.progress is not None else ""
        return f"ProgressBar(bbox={str(self.bbox)}{progress_str})"


class Part(LegoPageElement):
    """A single part entry within a parts list.

    Positional context: The part image/diagram appears first, with the PartCount
    label positioned directly below it, both left-aligned.

    See layout diagram: lego_page_layout.png
    See overview of all parts: https://brickarchitect.com/files/LEGO_BRICK_LABELS-CONTACT_SHEET.pdf
    """

    tag: Literal["Part"] = Field(default="Part", alias="__tag__", frozen=True)
    count: PartCount

    diagram: PartImage | None = None

    number: PartNumber | None = None
    """Optional part number used only on catalog pages."""

    length: PieceLength | None = None
    """Optional piece length indicator (e.g., '4' for a 4-stud axle).
    
    Appears in the top-right of the part image, surrounded by a circle.
    Can appear on any page type.
    """

    # TODO maybe add color?
    # TODO Some parts have a "shiny" highlight - maybe reference that image

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        number_str = f", number={self.number.element_id}" if self.number else ""
        length_str = f", len={self.length.value}" if self.length else ""
        return f"Part(count={self.count.count}x{number_str}{length_str})"

    def iter_elements(self) -> Iterator[LegoPageElement]:
        """Iterate over this Part and all child elements."""
        yield self
        yield self.count
        if self.diagram:
            yield from self.diagram.iter_elements()
        if self.number:
            yield from self.number.iter_elements()
        if self.length:
            yield from self.length.iter_elements()


class PartsList(LegoPageElement):
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

    def iter_elements(self) -> Iterator[LegoPageElement]:
        """Iterate over this PartsList and all child elements."""
        yield self
        for part in self.parts:
            yield from part.iter_elements()


class BagNumber(LegoPageElement):
    """The bag number, usually a small integer on the page."""

    tag: Literal["BagNumber"] = Field(default="BagNumber", alias="__tag__", frozen=True)
    value: Annotated[int, Gt(0)]

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"BagNumber(value={self.value})"


class NewBag(LegoPageElement):
    """The graphic showing a new bag icon on the page."""

    tag: Literal["NewBag"] = Field(default="NewBag", alias="__tag__", frozen=True)
    number: BagNumber

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"NewBag(bag={self.number.value})"

    def iter_elements(self) -> Iterator[LegoPageElement]:
        """Iterate over this Step and all child elements."""
        yield self
        yield from self.number.iter_elements()


class Diagram(LegoPageElement):
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


class Step(LegoPageElement):
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
    diagram: Diagram | None = None

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

    def iter_elements(self) -> Iterator[LegoPageElement]:
        """Iterate over this Step and all child elements."""
        yield self
        yield from self.step_number.iter_elements()
        yield from self.parts_list.iter_elements()
        if self.diagram:
            yield from self.diagram.iter_elements()


class Page(LegoPageElement):
    """A complete page of LEGO instructions.

    This is the top-level element that contains all other elements on a page.
    It represents the structured, hierarchical view of the page after classification
    and hierarchy building.

    Attributes:
        page_number: The page number element, if found
        steps: List of Step elements on the page (for INSTRUCTION pages)
        catalog: Parts list for catalog/inventory pages (for CATALOG pages)
        warnings: List of warnings generated during hierarchy building
        unprocessed_elements: Raw elements that were classified but couldn't
            be converted
    """

    class PageType(Enum):
        """Type of LEGO instruction page."""

        INSTRUCTION = "instruction"
        CATALOG = "catalog"
        INFO = "info"

    tag: Literal["Page"] = Field(default="Page", alias="__tag__", frozen=True)

    categories: Annotated[
        set[PageType],
        PlainSerializer(
            lambda cats: sorted(cat.value for cat in cats), return_type=list[str]
        ),
    ] = Field(default_factory=set)
    """Set of categories this page belongs to. A page can have multiple categories.
    
    For example, a page might be both INSTRUCTION and CATALOG if it contains
    both building steps and a parts catalog.
    
    Note: Serialized as a sorted list for deterministic JSON output.
    """

    page_number: PageNumber | None = None
    progress_bar: ProgressBar | None = None

    new_bags: list[NewBag] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    catalog: list[Part] = Field(default_factory=list)
    """List of parts for catalog pages. Empty list for non-catalog pages."""

    @property
    def is_instruction(self) -> bool:
        """Check if this page is an instruction page."""
        return Page.PageType.INSTRUCTION in self.categories

    @property
    def is_catalog(self) -> bool:
        """Check if this page is a catalog page."""
        return Page.PageType.CATALOG in self.categories

    @property
    def is_info(self) -> bool:
        """Check if this page is an info page."""
        return Page.PageType.INFO in self.categories

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        page_num = self.page_number.value if self.page_number else "unknown"
        categories_str = (
            f", categories=[{', '.join(c.name for c in self.categories)}]"
            if self.categories
            else ""
        )
        bags_str = f", bags={len(self.new_bags)}" if self.new_bags else ""
        catalog_str = f", catalog={len(self.catalog)} parts" if self.catalog else ""
        steps_str = f", steps={len(self.steps)}" if self.steps else ""
        return (
            f"Page(number={page_num}{categories_str}{bags_str}{catalog_str}{steps_str})"
        )

    def iter_elements(self) -> Iterator[LegoPageElement]:
        """Iterate over this Page and all child elements.

        Yields all elements in depth-first order: the Page itself, then all
        contained elements (page_number, progress_bar, steps and their children).

        Yields:
            This element and all descendant LegoPageElements
        """
        yield self

        if self.page_number:
            yield from self.page_number.iter_elements()
        if self.progress_bar:
            yield from self.progress_bar.iter_elements()

        for new_bag in self.new_bags:
            yield from new_bag.iter_elements()

        for part in self.catalog:
            yield from part.iter_elements()

        for step in self.steps:
            yield from step.iter_elements()


LegoPageElements = Annotated[
    PageNumber
    | StepNumber
    | PartCount
    | PartNumber
    | PieceLength
    | PartImage
    | ProgressBar
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
