"""Type definitions for PyMuPDF (pymupdf) data structures.

See https://pymupdf.readthedocs.io/en/latest/textpage.html#page-dictionary
"""

from typing import NotRequired, Protocol, TypedDict

# Type alias for bounding box coordinates (x0, y0, x1, y1)
RectLikeTuple = tuple[float, float, float, float]


class RectLike(Protocol):
    """Protocol for PyMuPDF Rect-like objects with coordinate attributes."""

    x0: float
    y0: float
    x1: float
    y1: float


class CharDict(TypedDict):
    """Type definition for a char in PyMuPDF rawdict."""

    c: str


class SpanDict(TypedDict):
    """Type definition for a text span in PyMuPDF rawdict."""

    bbox: RectLikeTuple
    text: str
    chars: list[CharDict]


class LineDict(TypedDict):
    """Type definition for a text line in PyMuPDF rawdict."""

    spans: list[SpanDict]


class BlockDict(TypedDict):
    """Type definition for a block in PyMuPDF rawdict."""

    number: int
    type: int
    bbox: RectLikeTuple


class TextBlockDict(BlockDict):
    """Type definition for a text block in PyMuPDF rawdict."""

    lines: NotRequired[list[LineDict]]


class ImageBlockDict(BlockDict):
    """Type definition for a image block in PyMuPDF rawdict."""

    # TODO Should this be
    # https://pymupdf.readthedocs.io/en/latest/document.html#Document.get_page_images ?

    ext: str  # image type (e.g., "bmp", "gif", "jpeg", "jpx", "png", "tiff")
    width: int  # original image width
    height: int  # original image height
    colorspace: int  # colorspace component count
    xres: int  # resolution in x-direction (always 96)
    yres: int  # resolution in y-direction (always 96)
    bpc: int  # bits per component
    transform: tuple[float, float, float, float, float, float]  # image transform
    size: int  # size of the image in bytes
    image: bytes  # image content
    mask: NotRequired[bytes]  # image mask for transparent images


class DrawingDict(TypedDict):
    """Type definition for a drawing block from PyMuPDF page.get_drawings()."""

    rect: RectLike  # PyMuPDF Rect object with .x0, .y0, .x1, .y1 attributes

    closePath: NotRequired[bool]  # whether path is closed
    color: NotRequired[tuple[float, ...] | None]  # stroke color
    dashes: NotRequired[str | None]  # dashed line specification
    even_odd: NotRequired[bool]  # fill behavior for overlaps
    fill: NotRequired[tuple[float, ...] | None]  # fill color
    items: NotRequired[list[tuple]]  # list of draw commands
    lineCap: NotRequired[tuple[int, int, int]]  # line cap styles
    lineJoin: NotRequired[int]  # line join style
    fill_opacity: NotRequired[float]  # fill transparency
    stroke_opacity: NotRequired[float]  # stroke transparency
    layer: NotRequired[str]  # optional content group name
    level: NotRequired[int]  # hierarchy level
    seqno: NotRequired[int]  # command sequence number
    type: NotRequired[str]  # path type: "f", "s", "fs", "clip", "group"
    width: NotRequired[float]  # stroke line width


class RawDict(TypedDict):
    """Type definition for PyMuPDF page.get_text('rawdict') return value."""

    blocks: list[TextBlockDict | ImageBlockDict]
