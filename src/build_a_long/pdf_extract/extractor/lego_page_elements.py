from __future__ import annotations

import json
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
    model_validator,
)

from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.utils import SerializationMixin, remove_empty_lists


class LegoPageElement(SerializationMixin, BaseModel, ABC):
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

    def to_json(self, *, indent: str | int | None = None, **kwargs: Any) -> str:
        """Serialize to JSON with proper defaults (by_alias=True, exclude_none=True).

        Floats are rounded to 2 decimal places for consistent output.
        Empty lists are removed from the output.

        Args:
            indent: Optional indentation for pretty-printing (str like '\t', int, or None)
            **kwargs: Additional arguments passed to model_dump()
        """
        # Use to_dict() from mixin which rounds floats
        data = self.to_dict(**kwargs)
        cleaned_data = remove_empty_lists(data)

        # Use compact separators when not indented (matches Pydantic's behavior)
        separators = (",", ":") if indent is None else (",", ": ")
        return json.dumps(cleaned_data, indent=indent, separators=separators)

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
    count: Annotated[int, Gt(0)]

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


class StepCount(LegoPageElement):
    """The visual count label for a substep (e.g., '2x').

    Positional context: Positioned inside a substep callout box, indicating
    how many times to build that sub-assembly.

    This is similar to PartCount but uses a larger font size and appears
    in substep callout boxes rather than parts lists.

    See layout diagram: lego_page_layout.png
    """

    tag: Literal["StepCount"] = Field(default="StepCount", alias="__tag__", frozen=True)
    count: Annotated[int, Gt(0)]

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"StepCount(count={self.count}x)"


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


class Shine(LegoPageElement):
    """A visual 'shine' or 'star' effect indicating a shiny/metallic part.

    Positional context: Typically a small star-like drawing located in the
    top-right area of a part image.
    """

    tag: Literal["Shine"] = Field(default="Shine", alias="__tag__", frozen=True)

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"Shine(bbox={self.bbox})"


class PartImage(LegoPageElement):
    """A candidate image that could represent a LEGO part.

    Positional context: These images typically appear in parts lists, positioned
    above their corresponding PartCount text and left-aligned.

    This element represents a validated image candidate that will be paired with
    a PartCount by PartsClassifier to construct Part elements.

    The bbox field (inherited from LegoPageElement) defines the image region.
    """

    tag: Literal["PartImage"] = Field(default="PartImage", alias="__tag__", frozen=True)

    shine: Shine | None = None
    """Optional shine effect indicating a metallic part."""

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        shine_str = ", shiny" if self.shine else ""
        return f"PartImage(bbox={self.bbox}{shine_str})"

    def iter_elements(self) -> Iterator[LegoPageElement]:
        """Iterate over this PartImage and all child elements."""
        yield self
        if self.shine:
            yield from self.shine.iter_elements()


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


class RotationSymbol(LegoPageElement):
    """A symbol indicating the builder should rotate the assembled model.

    Positional context: Typically appears near diagram elements, often positioned
    beside or below the main instruction diagram. Can be either a small raster
    image (~40-80 pixels square) or a cluster of vector drawings forming curved
    arrows in a circular pattern.

    See layout diagram: lego_page_layout.png
    """

    tag: Literal["RotationSymbol"] = Field(
        default="RotationSymbol", alias="__tag__", frozen=True
    )

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"RotationSymbol(bbox={self.bbox})"


class Arrow(LegoPageElement):
    """An arrow indicating direction or relationship between elements.

    Arrows consist of a triangular arrowhead that points in a specific direction.
    In LEGO instructions, arrows typically:
    - Point from a main assembly to a sub-step callout
    - Indicate direction of motion or insertion
    - Connect related elements visually

    The bbox encompasses the arrowhead. The tip is where the arrow points TO,
    and the tail is where the arrow line originates FROM (the other end of the
    arrow shaft, if detected).

    Direction is measured in degrees where:
    - 0° = pointing right
    - 90° = pointing down
    - 180° or -180° = pointing left
    - -90° = pointing up
    """

    tag: Literal["Arrow"] = Field(default="Arrow", alias="__tag__", frozen=True)

    direction: float
    """Angle in degrees indicating where the arrow points.

    0° = right, 90° = down, 180° = left, -90° = up.
    """

    tip: tuple[float, float]
    """The tip point (x, y) of the arrowhead - where the arrow points TO."""

    tail: tuple[float, float] | None = None
    """The tail point (x, y) - where the arrow line originates FROM.
    
    This is the far end of the arrow shaft (not the arrowhead base).
    May be None if the arrow shaft was not detected.
    """

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"Arrow(bbox={self.bbox}, direction={self.direction:.0f}°)"


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
    """The graphic showing a new bag icon on the page.

    A NewBag can have an optional bag number. When present, the number indicates
    which specific bag to open (e.g., "Bag 1", "Bag 2"). When absent, the bag
    graphic indicates that all bags should be opened (no specific number).
    """

    tag: Literal["NewBag"] = Field(default="NewBag", alias="__tag__", frozen=True)
    number: BagNumber | None = None

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        if self.number:
            return f"NewBag(bag={self.number.value})"
        return "NewBag(all)"

    def iter_elements(self) -> Iterator[LegoPageElement]:
        """Iterate over this Step and all child elements."""
        yield self
        if self.number:
            yield from self.number.iter_elements()


class Diagram(LegoPageElement):
    """The graphic showing how to complete the step.

    Positional context: The main diagram is positioned on the right side of the
    step area, occupying most of the horizontal space. It shows the assembly
    instructions for the step.

    See layout diagram: lego_page_layout.png
    """

    tag: Literal["Diagram"] = Field(default="Diagram", alias="__tag__", frozen=True)

    # TODO Figure out where arrows fit in the hierarchy
    arrows: list[Arrow] = Field(default_factory=list)
    """Direction arrows indicating piece movement within the diagram."""

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        arrows_str = f", arrows={len(self.arrows)}" if self.arrows else ""
        return f"Diagram(bbox={str(self.bbox)}{arrows_str})"

    def iter_elements(self) -> Iterator[LegoPageElement]:
        """Iterate over this Diagram and all child elements (arrows)."""
        yield self
        for arrow in self.arrows:
            yield from arrow.iter_elements()


class SubStep(LegoPageElement):
    """A sub-step within a main step, typically shown in a callout box.

    Positional context: SubSteps appear as white/light-colored rectangular boxes
    with arrows pointing from them to the main diagram. They show smaller
    sub-assemblies that may need to be built multiple times (indicated by a count
    like "2x") before being attached to the main assembly.

    Structure:
    - A white/light rectangular box (detected via Drawing blocks)
    - A diagram showing the sub-assembly
    - An optional count indicating how many times to build it (e.g., "2x")

    Note: Arrows pointing from substeps to the main diagram are stored in the
    parent Step element's arrows field.

    See layout diagram: lego_page_layout.png
    """

    tag: Literal["SubStep"] = Field(default="SubStep", alias="__tag__", frozen=True)

    diagram: Diagram | None = None
    """The diagram showing the sub-assembly."""

    count: StepCount | None = None
    """Optional count indicating how many times to build this sub-assembly."""

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        count_str = f"count={self.count.count}x, " if self.count else ""
        diagram_str = "diagram" if self.diagram else "no diagram"
        return f"SubStep({count_str}{diagram_str})"

    def iter_elements(self) -> Iterator[LegoPageElement]:
        """Iterate over this SubStep and all child elements."""
        yield self
        if self.count:
            yield from self.count.iter_elements()
        if self.diagram:
            yield from self.diagram.iter_elements()


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
    parts_list: PartsList | None = None
    diagram: Diagram | None = None
    rotation_symbol: RotationSymbol | None = None
    """Optional rotation symbol indicating the builder should rotate the model."""

    arrows: list[Arrow] = Field(default_factory=list)
    """Arrows indicating direction or relationship between elements.
    
    These typically point from substep callout boxes to the main diagram,
    or indicate direction of motion/insertion for parts.
    """

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        rotation_str = ", rotation" if self.rotation_symbol else ""
        arrows_str = f", arrows={len(self.arrows)}" if self.arrows else ""
        parts_count = len(self.parts_list.parts) if self.parts_list else 0
        return (
            f"Step(number={self.step_number.value}, "
            f"parts={parts_count}{rotation_str}{arrows_str})"
        )

    @property
    def value(self) -> int:
        """Return the step number value for convenience."""
        return self.step_number.value

    def iter_elements(self) -> Iterator[LegoPageElement]:
        """Iterate over this Step and all child elements."""
        yield self
        yield from self.step_number.iter_elements()
        if self.parts_list:
            yield from self.parts_list.iter_elements()
        if self.diagram:
            yield from self.diagram.iter_elements()
        if self.rotation_symbol:
            yield from self.rotation_symbol.iter_elements()
        for arrow in self.arrows:
            yield from arrow.iter_elements()


class Page(LegoPageElement):
    """A complete page of LEGO instructions.

    This is the top-level element that contains all other elements on a page.
    It represents the structured, hierarchical view of the page after classification
    and hierarchy building.

    Attributes:
        pdf_page_number: The 1-indexed page number from the original PDF
        page_number: The LEGO page number element (printed on the page), if found
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

    pdf_page_number: int
    """The 1-indexed page number from the original PDF."""

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
    | StepCount
    | PartCount
    | PartNumber
    | PieceLength
    | PartImage
    | Shine
    | ProgressBar
    | RotationSymbol
    | Arrow
    | Part
    | PartsList
    | BagNumber
    | NewBag
    | Diagram
    | SubStep
    | Step
    | Page,
    Discriminator("tag"),
]


class Manual(SerializationMixin, BaseModel):
    """A complete LEGO instruction manual containing all pages.

    This is the top-level container that holds all pages from a PDF and provides
    cross-page analysis capabilities like finding unique parts, matching parts
    across pages by image digest, and navigating between pages.

    Pages are automatically sorted by PDF page number when the Manual is created.

    Attributes:
        pages: List of Page objects, sorted by pdf_page_number
        set_number: Optional LEGO set number (e.g., "75375")
        name: Optional name of the set (e.g., "Millennium Falcon")
    """

    model_config = ConfigDict(populate_by_name=True)

    tag: Literal["Manual"] = Field(default="Manual", alias="__tag__", frozen=True)

    # Set information
    set_number: str | None = None
    name: str | None = None

    # Source PDF metadata
    source_pdf: str | None = None
    """Path to the source PDF file."""

    source_size: int | None = None
    """Size of the source PDF file in bytes."""

    source_hash: str | None = None
    """Hash of the source PDF file (e.g. SHA256)."""

    # Main parsed contents
    pages: list[Page] = Field(default_factory=list)
    """List of Page objects, sorted by pdf_page_number."""

    unsupported_reason: str | None = None
    """If present, indicates why this manual could not be fully processed."""

    @model_validator(mode="after")
    def sort_pages(self) -> Manual:
        """Sort pages by PDF page number after initialization."""
        self.pages.sort(key=lambda p: p.pdf_page_number)
        return self

    def get_page(self, pdf_page_number: int) -> Page | None:
        """Get a page by its PDF page number.

        Args:
            pdf_page_number: The PDF page number to find (1-indexed)

        Returns:
            The Page at that PDF page number, or None if not found
        """
        for page in self.pages:
            if page.pdf_page_number == pdf_page_number:
                return page
        return None

    def get_page_by_lego_number(self, lego_page_number: int) -> Page | None:
        """Get a page by its LEGO page number (the number printed on the page).

        Args:
            lego_page_number: The LEGO page number to find

        Returns:
            The Page with the matching LEGO page number, or None if not found
        """
        for page in self.pages:
            if page.page_number and page.page_number.value == lego_page_number:
                return page
        return None

    @property
    def instruction_pages(self) -> list[Page]:
        """Get all instruction pages (pages with building steps)."""
        return [p for p in self.pages if p.is_instruction]

    @property
    def catalog_pages(self) -> list[Page]:
        """Get all catalog pages (pages with parts inventory)."""
        return [p for p in self.pages if p.is_catalog]

    @property
    def info_pages(self) -> list[Page]:
        """Get all info pages."""
        return [p for p in self.pages if p.is_info]

    @property
    def catalog_parts(self) -> list[Part]:
        """Get all parts from catalog pages.

        Returns:
            List of all Part objects from catalog pages, which typically have
            PartNumber (element_id) information for identification.
        """
        parts: list[Part] = []
        for page in self.catalog_pages:
            parts.extend(page.catalog)
        return parts

    @property
    def all_steps(self) -> list[Step]:
        """Get all steps from all instruction pages in order.

        Returns:
            List of all Step objects from instruction pages
        """
        steps: list[Step] = []
        for page in self.instruction_pages:
            steps.extend(page.steps)
        return steps

    @property
    def total_parts_count(self) -> int:
        """Get the total count of all parts across all steps.

        This sums up all part counts from all parts lists in all steps.
        """
        total = 0
        for step in self.all_steps:
            if step.parts_list:
                total += step.parts_list.total_items
        return total

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        set_str = f"set={self.set_number}, " if self.set_number else ""
        name_str = f'"{self.name}", ' if self.name else ""
        return (
            f"Manual({set_str}{name_str}"
            f"pages={len(self.pages)}, "
            f"steps={len(self.all_steps)}, "
            f"catalog_parts={len(self.catalog_parts)})"
        )

    def to_json(self, *, indent: str | int | None = None, **kwargs: Any) -> str:
        """Serialize to JSON with proper defaults (by_alias=True, exclude_none=True).

        Floats are rounded to 2 decimal places for consistent output.
        Empty lists are removed from the output.

        Args:
            indent: Optional indentation for pretty-printing (str like '\t', int, or None)
            **kwargs: Additional arguments passed to model_dump()
        """
        # Use to_dict() from mixin which rounds floats
        data = self.to_dict(**kwargs)
        cleaned_data = remove_empty_lists(data)

        # Use compact separators when not indented (matches Pydantic's behavior)
        separators = (",", ":") if indent is None else (",", ": ")
        return json.dumps(cleaned_data, indent=indent, separators=separators)


# TODO Add sub-assembly (or sub-step) element.
# TODO Add a final preview element.
# TODO Add a "information" element (for facts about the set).
