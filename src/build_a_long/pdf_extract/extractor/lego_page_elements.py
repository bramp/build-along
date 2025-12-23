from __future__ import annotations

import json
from abc import ABC
from collections.abc import Iterator, Sequence
from enum import Enum
from typing import Annotated, Any, ClassVar, Literal

from annotated_types import Ge, Gt
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    PlainSerializer,
    field_serializer,
    model_validator,
)

from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.sorting import (
    sort_by_columns,
    sort_by_rows,
    sort_left_to_right,
)
from build_a_long.pdf_extract.utils import (
    SerializationMixin,
    auto_id_field,
    remove_empty_lists,
)


class LegoPageElement(SerializationMixin, BaseModel, ABC):
    """Base class for LEGO-specific structured elements constructed by classifiers.

    LegoPageElements are typically constructed from one or more Blocks during
    classification and stored via ClassificationResult.build().

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

    id: int = Field(default_factory=auto_id_field, exclude=True)
    """Unique identifier for this element.

    Auto-generated at construction time. Use this for identity comparisons
    instead of id() which can differ across Pydantic deep-copies.
    Not serialized to JSON.
    """

    bbox: BBox

    def to_json(self, *, indent: str | int | None = None, **kwargs: Any) -> str:
        """Serialize to JSON with proper defaults (by_alias=True, exclude_none=True).

        Floats are rounded to 2 decimal places for consistent output.
        Empty lists are removed from the output.

        Args:
            indent: Optional indentation for pretty-printing (str like '\t', int,
                or None)
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

    # TODO Do we really need this ?
    matched_hint: Literal["part_count", "catalog_part_count"] | None = Field(
        default=None, exclude=True
    )
    """Which font size hint was matched during classification.

    - 'part_count': Standard instruction page part count
    - 'catalog_part_count': Catalog/inventory page part count
    """

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


class ScaleText(LegoPageElement):
    """The '1:1' text indicating the scale of the diagram.

    Positional context: Located within the Scale element's bounding box,
    usually to the left or right of the part diagram.
    """

    tag: Literal["ScaleText"] = Field(default="ScaleText", alias="__tag__", frozen=True)
    text: str = "1:1"

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"ScaleText(bbox={self.bbox}, text={self.text!r})"


class Scale(LegoPageElement):
    """A 1:1 scale indicator showing the actual size of a piece.

    Positional context: Typically appears at the bottom of the page, showing
    a piece bar/ruler with a piece length indicator, part image, and "1:1" text
    to indicate the printed scale matches the actual LEGO piece size.

    This helps builders verify piece lengths by measuring against the printed
    instruction manual.

    See layout diagram: lego_page_layout.png
    """

    tag: Literal["Scale"] = Field(default="Scale", alias="__tag__", frozen=True)

    length: PieceLength
    """The piece length indicator showing the measurement (e.g., 3 studs)."""

    text: ScaleText | None = None
    """The '1:1' text indicating scale."""

    diagram: PartImage | None = None
    """The part diagram shown at 1:1 scale."""

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"Scale(length={self.length.value})"

    def iter_elements(self) -> Iterator[LegoPageElement]:
        """Iterate over this Scale and all child elements."""
        yield self
        yield from self.length.iter_elements()
        if self.text:
            yield from self.text.iter_elements()
        if self.diagram:
            yield from self.diagram.iter_elements()


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

    # TODO: image_id, digest, and xref are temporary identifiers used for
    # classification, deduplication, and validation. They should eventually be
    # removed in favor of directly linking matching PartImage objects or using
    # a centralized AssetManager.
    image_id: str | None = Field(default=None, exclude=True)
    """Optional image ID from the source Image block (e.g., 'image_123')."""

    digest: bytes | None = Field(default=None, exclude=True)
    """MD5 digest of the image data (for deduplication)."""

    xref: int | None = Field(default=None, exclude=True)
    """PDF cross-reference ID (for deduplication within a PDF)."""

    @field_serializer("digest")
    @classmethod
    def _serialize_digest(cls, v: bytes | None) -> str | None:
        """Serialize digest bytes to hex string for JSON output."""
        return v.hex() if v is not None else None

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        shine_str = ", shiny" if self.shine else ""
        image_id_str = f", image_id={self.image_id}" if self.image_id else ""
        return f"PartImage(bbox={self.bbox}{shine_str}{image_id_str})"

    def iter_elements(self) -> Iterator[LegoPageElement]:
        """Iterate over this PartImage and all child elements."""
        yield self
        if self.shine:
            yield from self.shine.iter_elements()


class ProgressBarIndicator(LegoPageElement):
    """The movable indicator showing current progress within a progress bar.

    Positional context: Located within the ProgressBar, positioned horizontally
    to indicate how far through the instructions the reader has progressed.
    The indicator is typically a circular element that extends above and below
    the bar, sitting on top of the bar visually.

    See layout diagram: lego_page_layout.png
    """

    tag: Literal["ProgressBarIndicator"] = Field(
        default="ProgressBarIndicator", alias="__tag__", frozen=True
    )

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"ProgressBarIndicator(bbox={str(self.bbox)})"


class ProgressBarBar(LegoPageElement):
    """The bar portion of a progress bar, spanning across the page.

    Positional context: A long, thin horizontal element at the bottom of the page.
    This represents the track along which the indicator moves. It may consist of
    multiple overlapping Drawing/Image elements that form a single visual bar.

    See layout diagram: lego_page_layout.png
    """

    tag: Literal["ProgressBarBar"] = Field(
        default="ProgressBarBar", alias="__tag__", frozen=True
    )

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"ProgressBarBar(bbox={str(self.bbox)})"


class ProgressBar(LegoPageElement):
    """A progress bar showing building progress through the instruction book.

    Positional context: Typically located at the bottom of the page, spanning most
    of the page width, near the page number. Consists of two main components:

    - bar: The horizontal track (ProgressBarBar) spanning the page width
    - indicator: The circular marker (ProgressBarIndicator) showing current progress

    The indicator sits on top of the bar and extends above and below it vertically.

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

    bar: ProgressBarBar
    """The horizontal bar track spanning the page width."""

    indicator: ProgressBarIndicator | None = None
    """The progress indicator element, if detected."""

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        progress_str = f", {self.progress:.1%}" if self.progress is not None else ""
        return f"ProgressBar(bbox={str(self.bbox)}{progress_str})"

    def iter_elements(self) -> Iterator[LegoPageElement]:
        """Iterate over this ProgressBar and its child elements.

        Yields:
            This element, then the bar, then the indicator if present
        """
        yield self
        yield from self.bar.iter_elements()
        if self.indicator:
            yield from self.indicator.iter_elements()


class Divider(LegoPageElement):
    """A visual divider line separating sections of the page.

    Dividers are thin lines (typically white strokes) that run vertically or
    horizontally across a significant portion of the page (>40% of page
    height/width). They visually separate different instruction steps or
    sections on a page.

    Positional context: Can appear anywhere on the page, typically:
    - Vertical dividers separate left/right columns of steps
    - Horizontal dividers separate top/bottom sections

    See layout diagram: lego_page_layout.png
    """

    class Orientation(str, Enum):
        """Orientation of the divider line."""

        VERTICAL = "vertical"
        HORIZONTAL = "horizontal"

    tag: Literal["Divider"] = Field(default="Divider", alias="__tag__", frozen=True)

    orientation: Orientation
    """Whether the divider runs vertically or horizontally."""

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"Divider(bbox={str(self.bbox)}, {self.orientation.value})"


class Background(LegoPageElement):
    """The full-page background element, typically a colored rectangle.

    Background elements are large Drawing or Image blocks that cover most or all
    of the page. They are typically gray rectangles that form the visual
    backdrop for the instruction content.

    There should be at most one Background per page. This element collects
    all background-related blocks (full-page fills, page-edge lines) into
    a single logical element.

    Positional context: Covers the entire page or nearly the entire page.

    See layout diagram: lego_page_layout.png
    """

    tag: Literal["Background"] = Field(
        default="Background", alias="__tag__", frozen=True
    )

    fill_color: tuple[float, float, float] | None = None
    """RGB fill color of the background (0.0-1.0 per channel), if any."""

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        color_str = ""
        if self.fill_color:
            r, g, b = self.fill_color
            color_str = f", fill=[{r:.2f},{g:.2f},{b:.2f}]"
        return f"Background(bbox={str(self.bbox)}{color_str})"


class TriviaText(LegoPageElement):
    """Trivia or flavor text containing fun facts about the LEGO set.

    These are informational text blocks that appear on some pages, containing
    stories, facts, or background information about the set's theme. They are
    not part of the building instructions themselves.

    Characteristics:
    - Multiple lines of text in a smaller font (typically 8pt)
    - Often in multiple languages (English, French, Spanish, etc.)
    - May have an accompanying image or illustration
    - Usually located at the bottom portion of a page

    Positional context: Typically appears in the lower section of pages,
    spanning a significant width of the page.
    """

    tag: Literal["TriviaText"] = Field(
        default="TriviaText", alias="__tag__", frozen=True
    )

    text_lines: Sequence[str] = Field(default_factory=list, exclude=True)
    """The text content, split by line."""

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        preview = self.text_lines[0][:30] + "..." if self.text_lines else ""
        return f"TriviaText(lines={len(self.text_lines)}, {preview!r})"


class Decoration(LegoPageElement):
    """A decorative element on INFO pages (logos, graphics, etc.).

    Decorations are visual elements that appear on non-instruction pages
    like covers, credits, table of contents, and other informational pages.
    They are not part of the building instructions and exist primarily
    to consume blocks that would otherwise be left unconsumed.

    Characteristics:
    - Can be any visual element: logos, images, text, graphics
    - Appears on pages classified as INFO pages
    - Not semantically meaningful for instruction purposes

    Positional context: Can appear anywhere on INFO pages.
    """

    tag: Literal["Decoration"] = Field(
        default="Decoration", alias="__tag__", frozen=True
    )

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"Decoration(bbox={self.bbox})"


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


class ArrowHead(BaseModel):
    """A single arrowhead within an Arrow element.

    Represents one triangular arrowhead with its tip position and direction.
    """

    tip: tuple[float, float]
    """The tip point (x, y) - where this arrowhead points TO."""

    direction: float
    """Angle in degrees indicating where the arrowhead points.

    0° = right, 90° = down, 180° = left, -90° = up.
    """


class Arrow(LegoPageElement):
    """An arrow indicating direction or relationship between elements.

    Arrows consist of one or more triangular arrowheads that point in specific
    directions. In LEGO instructions, arrows typically:
    - Point from a main assembly to a sub-step callout
    - Indicate direction of motion or insertion
    - Connect related elements visually

    Arrows can have multiple heads (Y-shaped or branching arrows) that share a
    common tail/origin point. The bbox encompasses all arrowheads and the shaft.

    Direction is measured in degrees where:
    - 0° = pointing right
    - 90° = pointing down
    - 180° or -180° = pointing left
    - -90° = pointing up
    """

    tag: Literal["Arrow"] = Field(default="Arrow", alias="__tag__", frozen=True)

    heads: Sequence[ArrowHead]
    """List of arrowheads, each with its own tip and direction.

    Most arrows have a single head, but branching arrows (Y-shaped) can have
    multiple heads sharing the same tail/origin.
    """

    tail: tuple[float, float] | None = None
    """The tail point (x, y) - where the arrow line originates FROM.

    This is the far end of the arrow shaft (not the arrowhead base).
    May be None if the arrow shaft was not detected.
    For branching arrows, this is the common origin point.
    """

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        if len(self.heads) == 1:
            return f"Arrow(bbox={self.bbox}, direction={self.heads[0].direction:.0f}°)"
        return (
            f"Arrow(bbox={self.bbox}, heads={len(self.heads)}, "
            f"directions={[f'{h.direction:.0f}°' for h in self.heads]})"
        )


class Part(LegoPageElement):
    """A single part entry within a parts list.

    Positional context: The part image/diagram appears first, with the PartCount
    label positioned directly below it, both left-aligned.

    See layout diagram: lego_page_layout.png
    See overview of all parts: https://brickarchitect.com/files/LEGO_BRICK_LABELS-CONTACT_SHEET.pdf
    """

    __constraint_rules__: ClassVar[dict] = {
        "count": {"required": True},  # Count is mandatory
    }

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

    __constraint_rules__: ClassVar[dict] = {
        "parts": {"min_count": 1},  # Must have at least 1 part
    }

    tag: Literal["PartsList"] = Field(default="PartsList", alias="__tag__", frozen=True)
    parts: Sequence[Part]

    @model_validator(mode="after")
    def _sort_parts(self) -> PartsList:
        """Sort parts left-to-right, row by row."""
        object.__setattr__(self, "parts", sort_by_rows(self.parts))
        return self

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


class LoosePartSymbol(LegoPageElement):
    """An indicator symbol for a loose part that is not in the main bag.

    This is a cluster of drawings that appears next to an OpenBag circle
    when the bag contains a part instead of a bag number. The symbol indicates
    that an extra part is needed that's not found in the main bag.

    The symbol has a roughly square aspect ratio and consists of many small
    drawing elements that form a recognizable icon.
    """

    tag: Literal["LoosePartSymbol"] = Field(
        default="LoosePartSymbol", alias="__tag__", frozen=True
    )

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"LoosePartSymbol(bbox={str(self.bbox)})"


class OpenBag(LegoPageElement):
    """The graphic showing an open bag icon on the page.

    An OpenBag can contain either:
    - A bag number indicating which specific bag to open (e.g., "Bag 1", "Bag 2")
    - A part indicating a specific loose piece to find/use
    - Nothing, indicating that all bags should be opened

    When both number and part are None, the bag graphic indicates that all
    bags should be opened (no specific number or part highlighted).
    """

    tag: Literal["OpenBag"] = Field(default="OpenBag", alias="__tag__", frozen=True)
    number: BagNumber | None = None
    part: Part | None = None
    loose_part_symbol: LoosePartSymbol | None = None
    """Optional symbol indicating a loose part not in the main bag."""

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        if self.number:
            return f"OpenBag(bag={self.number.value})"
        if self.part:
            symbol_str = ", loose_part" if self.loose_part_symbol else ""
            return f"OpenBag(part={self.part.count.count}x{symbol_str})"
        return "OpenBag(all)"

    def iter_elements(self) -> Iterator[LegoPageElement]:
        """Iterate over this OpenBag and all child elements."""
        yield self
        if self.number:
            yield from self.number.iter_elements()
        if self.part:
            yield from self.part.iter_elements()
        if self.loose_part_symbol:
            yield from self.loose_part_symbol.iter_elements()


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


class SubStep(LegoPageElement):
    """A single sub-step within a step or sub-assembly callout box.

    SubSteps are mini-steps that appear either:
    - Inside SubAssembly callout boxes (numbered 1, 2, 3 within the box)
    - As "naked" substeps on the page alongside a main step

    Each has its own step number (typically starting at 1) and diagram showing
    that sub-step's construction.
    """

    tag: Literal["SubStep"] = Field(default="SubStep", alias="__tag__", frozen=True)

    step_number: StepNumber
    """The step number for this sub-step (typically 1, 2, 3, etc.)."""

    diagram: Diagram
    """The diagram showing this sub-step's construction."""

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        return f"SubStep(number={self.step_number.value}, diagram)"

    def iter_elements(self) -> Iterator[LegoPageElement]:
        """Iterate over this SubStep and all child elements."""
        yield self
        yield from self.step_number.iter_elements()
        yield from self.diagram.iter_elements()


class Preview(LegoPageElement):
    """A preview area showing a diagram or model preview.

    Preview elements are white rectangular areas containing a diagram.
    They typically appear on info pages to show what the completed model
    (or a section of it) will look like.

    Structure:
    - A white rectangular background (Drawing block with white fill)
    - A diagram inside (which may be composed of multiple image blocks)
    - May have optional border drawings or semi-transparent overlay

    Positional context: Can appear anywhere on the page, typically in areas
    outside the main instruction content.

    See layout diagram: lego_page_layout.png
    """

    tag: Literal["Preview"] = Field(default="Preview", alias="__tag__", frozen=True)

    diagram: Diagram | None = None
    """The diagram inside the preview area.
    
    The diagram may be composed of multiple underlying image blocks
    that have been merged into a single Diagram element.
    """

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        diagram_str = ", has_diagram" if self.diagram else ""
        return f"Preview(bbox={str(self.bbox)}{diagram_str})"

    def iter_elements(self) -> Iterator[LegoPageElement]:
        """Iterate over this Preview and all child elements."""
        yield self
        if self.diagram:
            yield from self.diagram.iter_elements()


class SubAssembly(LegoPageElement):
    """A sub-assembly within a main step, typically shown in a callout box.

    Positional context: SubAssemblies appear as white/light-colored rectangular
    boxes with arrows pointing from them to the main diagram. They show smaller
    sub-assemblies that may need to be built multiple times (indicated by a count
    like "2x") before being attached to the main assembly.

    Structure:
    - A white/light rectangular box (detected via Drawing blocks)
    - One or more steps, each with a step number and diagram
    - An optional count indicating how many times to build it (e.g., "2x")

    Note: Arrows pointing from subassemblies to the main diagram are stored in
    the parent Step element's arrows field.

    See layout diagram: lego_page_layout.png
    """

    tag: Literal["SubAssembly"] = Field(
        default="SubAssembly", alias="__tag__", frozen=True
    )

    steps: Sequence[SubStep] = Field(default_factory=list)
    """The steps within this sub-assembly, each with a step number and diagram."""

    diagram: Diagram | None = None
    """The main/final diagram showing the completed sub-assembly.
    
    This is used for simple subassemblies without internal steps. When steps
    are present, each step has its own diagram instead.
    """

    count: StepCount | None = None
    """Optional count indicating how many times to build this sub-assembly."""

    @model_validator(mode="after")
    def _sort_steps(self) -> SubAssembly:
        """Sort steps by step number."""
        sorted_steps = sorted(self.steps, key=lambda s: s.step_number.value)
        object.__setattr__(self, "steps", sorted_steps)
        return self

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        count_str = f"count={self.count.count}x, " if self.count else ""
        if self.steps:
            steps_str = f"steps={len(self.steps)}"
        elif self.diagram:
            steps_str = "diagram"
        else:
            steps_str = "no diagram"
        return f"SubAssembly({count_str}{steps_str})"

    def iter_elements(self) -> Iterator[LegoPageElement]:
        """Iterate over this SubAssembly and all child elements."""
        yield self
        if self.count:
            yield from self.count.iter_elements()
        for step in self.steps:
            yield from step.iter_elements()
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

    arrows: Sequence[Arrow] = Field(default_factory=list)
    """Arrows indicating direction or relationship between elements.
    
    These typically point from subassembly callout boxes to the main diagram,
    or indicate direction of motion/insertion for parts.
    """

    subassemblies: Sequence[SubAssembly] = Field(default_factory=list)
    """Sub-assemblies shown in callout boxes within this step.
    
    SubAssemblies show smaller sub-assemblies that may need to be built
    multiple times before being attached to the main assembly.
    """

    substeps: Sequence[SubStep] = Field(default_factory=list)
    """Sub-steps within this step that don't have a callout box.
    
    These are mini-steps (typically numbered 1, 2, 3, 4) that break down
    the main step into smaller pieces. Unlike subassemblies, these are not
    contained in a white callout box - they appear directly on the page.
    """

    @model_validator(mode="after")
    def _sort_elements(self) -> Step:
        """Sort child sequences: substeps by number, others by position."""
        # Sort substeps by step number
        sorted_substeps = sorted(self.substeps, key=lambda s: s.step_number.value)
        object.__setattr__(self, "substeps", sorted_substeps)

        # Sort arrows and subassemblies left-to-right
        object.__setattr__(self, "arrows", sort_left_to_right(self.arrows))
        object.__setattr__(
            self, "subassemblies", sort_left_to_right(self.subassemblies)
        )
        return self

    def __str__(self) -> str:
        """Return a single-line string representation with key information."""
        rotation_str = ", rotation" if self.rotation_symbol else ""
        arrows_str = f", arrows={len(self.arrows)}" if self.arrows else ""
        subassemblies_str = (
            f", subassemblies={len(self.subassemblies)}" if self.subassemblies else ""
        )
        substeps_str = f", substeps={len(self.substeps)}" if self.substeps else ""
        parts_count = len(self.parts_list.parts) if self.parts_list else 0
        return (
            f"Step(number={self.step_number.value}, "
            f"parts={parts_count}{rotation_str}{arrows_str}{subassemblies_str}{substeps_str})"
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
        for subassembly in self.subassemblies:
            yield from subassembly.iter_elements()
        for substep in self.substeps:
            yield from substep.iter_elements()


class InstructionContent(BaseModel):
    """Content specific to instruction pages.

    Contains the building steps and open bag indicators that are specific
    to pages showing how to build the LEGO set.
    """

    tag: Literal["InstructionContent"] = Field(
        default="InstructionContent", alias="__tag__", frozen=True
    )

    steps: Sequence[Step] = Field(default_factory=list)
    """List of Step elements on the page."""

    open_bags: Sequence[OpenBag] = Field(default_factory=list)
    """List of OpenBag elements indicating which bags to open."""

    @model_validator(mode="after")
    def _sort_elements(self) -> InstructionContent:
        """Sort steps by step number, open_bags by bag number then position."""
        # Sort steps by step number
        sorted_steps = sorted(self.steps, key=lambda s: s.step_number.value)
        object.__setattr__(self, "steps", sorted_steps)

        # Sort open_bags: by bag number if present, else by x position
        def open_bag_key(bag: OpenBag) -> tuple[int, float]:
            # Bags with numbers sort first by number, bags without sort by position
            if bag.number:
                return (0, bag.number.value)
            return (1, bag.bbox.x0)

        sorted_bags = sorted(self.open_bags, key=open_bag_key)
        object.__setattr__(self, "open_bags", sorted_bags)
        return self

    def iter_elements(self) -> Iterator[LegoPageElement]:
        """Iterate over all child elements."""
        for open_bag in self.open_bags:
            yield from open_bag.iter_elements()
        for step in self.steps:
            yield from step.iter_elements()


class CatalogContent(BaseModel):
    """Content specific to catalog/inventory pages.

    Contains parts that are displayed as a catalog or inventory listing,
    not associated with building steps.
    """

    tag: Literal["CatalogContent"] = Field(
        default="CatalogContent", alias="__tag__", frozen=True
    )

    parts: Sequence[Part] = Field(default_factory=list)
    """List of Part elements shown in the catalog."""

    @model_validator(mode="after")
    def _sort_parts(self) -> CatalogContent:
        """Sort parts top-to-bottom in columns."""
        object.__setattr__(self, "parts", sort_by_columns(self.parts))
        return self

    def iter_elements(self) -> Iterator[LegoPageElement]:
        """Iterate over all child elements."""
        for part in self.parts:
            yield from part.iter_elements()


class InfoContent(BaseModel):
    """Content specific to info pages.

    Contains decorative elements and other informational content that
    appears on pages without building instructions (e.g., cover pages,
    introduction pages).
    """

    tag: Literal["InfoContent"] = Field(
        default="InfoContent", alias="__tag__", frozen=True
    )

    decorations: Sequence[Decoration] = Field(default_factory=list)
    """List of decorative elements (logos, graphics, etc.)."""

    @model_validator(mode="after")
    def _sort_decorations(self) -> InfoContent:
        """Sort decorations left-to-right."""
        object.__setattr__(self, "decorations", sort_left_to_right(self.decorations))
        return self

    def iter_elements(self) -> Iterator[LegoPageElement]:
        """Iterate over all child elements."""
        for decoration in self.decorations:
            yield from decoration.iter_elements()


class Page(LegoPageElement):
    """A complete page of LEGO instructions.

    This is the top-level element that contains all other elements on a page.
    It represents the structured, hierarchical view of the page after classification
    and hierarchy building.

    Pages are composed of:
    - Common elements (background, page_number, progress_bar, etc.)
    - Optional content types based on page category:
      - instruction: Steps and open bags for building instructions
      - catalog: Parts list for inventory pages
      - info: Decorative content for informational pages

    A page can have multiple content types (e.g., both instruction and catalog).
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

    background: Background | None = None
    """The background element for the page, if detected."""

    page_number: PageNumber | None = None
    """The detected page number element on the page (not the PDF page number)."""

    progress_bar: ProgressBar | None = None
    """The detected progress bar element on the page, if present."""

    trivia_text: TriviaText | None = None
    """Trivia/flavor text on the page, if present."""

    dividers: Sequence[Divider] = Field(default_factory=list)
    """List of divider lines on the page separating sections."""

    scale: Scale | None = None
    """Scale indicator showing 1:1 printed size reference for piece lengths."""

    previews: Sequence[Preview] = Field(default_factory=list)
    """List of preview elements showing model diagrams."""

    unconsumed_blocks_count: int = 0
    """Count of blocks on the page that were not consumed by any element or filtered
    out."""

    # Content by page type (composition pattern)
    instruction: InstructionContent | None = None
    """Content for instruction pages (steps, open bags). None if not instruction."""

    catalog: CatalogContent | None = None
    """Content for catalog pages (parts listing). None if not catalog."""

    info: InfoContent | None = None
    """Content for info pages (decorations). None if not info."""

    @model_validator(mode="after")
    def _sort_elements(self) -> Page:
        """Sort dividers and previews left-to-right."""
        object.__setattr__(self, "dividers", sort_left_to_right(self.dividers))
        object.__setattr__(self, "previews", sort_left_to_right(self.previews))
        return self

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
        bags_str = (
            f", bags={len(self.instruction.open_bags)}"
            if self.instruction and self.instruction.open_bags
            else ""
        )
        catalog_str = (
            f", catalog={len(self.catalog.parts)} parts"
            if self.catalog and self.catalog.parts
            else ""
        )
        steps_str = (
            f", steps={len(self.instruction.steps)}"
            if self.instruction and self.instruction.steps
            else ""
        )
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

        if self.background:
            yield from self.background.iter_elements()
        if self.page_number:
            yield from self.page_number.iter_elements()
        if self.progress_bar:
            yield from self.progress_bar.iter_elements()
        if self.trivia_text:
            yield from self.trivia_text.iter_elements()

        for divider in self.dividers:
            yield from divider.iter_elements()

        if self.scale:
            yield from self.scale.iter_elements()

        for preview in self.previews:
            yield from preview.iter_elements()

        # Yield content from composed objects
        if self.instruction:
            yield from self.instruction.iter_elements()

        if self.catalog:
            yield from self.catalog.iter_elements()

        if self.info:
            yield from self.info.iter_elements()


LegoPageElements = Annotated[
    PageNumber
    | StepNumber
    | StepCount
    | PartCount
    | PartNumber
    | PieceLength
    | PartImage
    | Shine
    | Scale
    | ScaleText
    | ProgressBar
    | ProgressBarBar
    | ProgressBarIndicator
    | Background
    | TriviaText
    | Decoration
    | Divider
    | RotationSymbol
    | Arrow
    | Part
    | PartsList
    | BagNumber
    | OpenBag
    | LoosePartSymbol
    | Diagram
    | Preview
    | SubStep
    | SubAssembly
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
    def instruction_pages(self) -> Sequence[Page]:
        """Get all instruction pages (pages with building steps)."""
        return [p for p in self.pages if p.is_instruction]

    @property
    def catalog_pages(self) -> Sequence[Page]:
        """Get all catalog pages (pages with parts inventory)."""
        return [p for p in self.pages if p.is_catalog]

    @property
    def info_pages(self) -> Sequence[Page]:
        """Get all info pages."""
        return [p for p in self.pages if p.is_info]

    @property
    def catalog_parts(self) -> Sequence[Part]:
        """Get all parts from catalog pages.

        Returns:
            List of all Part objects from catalog pages, which typically have
            PartNumber (element_id) information for identification.
        """
        parts: Sequence[Part] = []
        for page in self.catalog_pages:
            if page.catalog:
                parts.extend(page.catalog.parts)
        return parts

    @property
    def all_steps(self) -> Sequence[Step]:
        """Get all steps from all instruction pages in order.

        Returns:
            List of all Step objects from instruction pages
        """
        steps: Sequence[Step] = []
        for page in self.instruction_pages:
            if page.instruction:
                steps.extend(page.instruction.steps)
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
            indent: Optional indentation for pretty-printing (str like '\t', int,
                or None)
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
