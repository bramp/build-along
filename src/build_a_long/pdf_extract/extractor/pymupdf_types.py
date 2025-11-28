"""Type definitions for PyMuPDF (pymupdf) data structures.

See https://pymupdf.readthedocs.io/en/latest/textpage.html#page-dictionary
"""

from typing import NotRequired, Protocol, TypedDict

# Type alias for single coordinates (x, y)
PointLikeTuple = tuple[float, float]

# Type alias for bounding box coordinates (x0, y0, x1, y1)
RectLikeTuple = tuple[float, float, float, float]

# Type alias for transformation matrix (a, b, c, d, e, f)
# Represents an affine transformation: [a b 0; c d 0; e f 1]
TransformTuple = tuple[float, float, float, float, float, float]


class RectLike(Protocol):
    """Protocol for PyMuPDF Rect-like objects with coordinate attributes.

    See https://pymupdf.readthedocs.io/en/latest/rect.html
    """

    x0: float
    y0: float
    x1: float
    y1: float


class CharDict(TypedDict):
    """Type definition for a char in PyMuPDF rawdict.

    See https://pymupdf.readthedocs.io/en/latest/textpage.html#dictionary-structure-of-extractdict-and-extractrawdict
    """

    c: str
    origin: NotRequired[PointLikeTuple]  # character origin (x, y)
    bbox: NotRequired[RectLikeTuple]  # character bounding box


class SpanDict(TypedDict):
    """Type definition for a text span in PyMuPDF rawdict.

    See https://pymupdf.readthedocs.io/en/latest/textpage.html#dictionary-structure-of-extractdict-and-extractrawdict
    """

    bbox: RectLikeTuple
    text: str
    chars: list[CharDict]
    font: NotRequired[str]  # font name
    size: NotRequired[float]  # font size in points
    flags: NotRequired[int]  # font flags (bitmap: superscript, italic, serif, etc.)
    color: NotRequired[int]  # text color as RGB integer
    ascender: NotRequired[float]  # font ascender
    descender: NotRequired[float]  # font descender
    origin: NotRequired[PointLikeTuple]  # span origin (x, y)


class LineDict(TypedDict):
    """Type definition for a text line in PyMuPDF rawdict.

    See https://pymupdf.readthedocs.io/en/latest/textpage.html#dictionary-structure-of-extractdict-and-extractrawdict
    """

    spans: list[SpanDict]
    bbox: NotRequired[RectLikeTuple]  # line bounding box
    wmode: NotRequired[int]  # writing mode (0=horizontal, 1=vertical)
    dir: NotRequired[RectLikeTuple]  # writing direction vector


class BlockDict(TypedDict):
    """Type definition for a block in PyMuPDF rawdict.

    See https://pymupdf.readthedocs.io/en/latest/textpage.html#dictionary-structure-of-extractdict-and-extractrawdict
    """

    number: int
    type: int
    bbox: RectLikeTuple


class TextBlockDict(BlockDict):
    """Type definition for a text block in PyMuPDF rawdict.

    See https://pymupdf.readthedocs.io/en/latest/textpage.html#dictionary-structure-of-extractdict-and-extractrawdict
    """

    lines: NotRequired[list[LineDict]]


class ImageBlockDict(BlockDict):
    """Type definition for a image block in PyMuPDF rawdict.

    See https://pymupdf.readthedocs.io/en/latest/textpage.html#block-dictionaries
    """

    ext: str  # image type (e.g., "bmp", "gif", "jpeg", "jpx", "png", "tiff")
    width: int  # original image width
    height: int  # original image height
    colorspace: int  # colorspace component count
    xres: int  # resolution in x-direction (always 96)
    yres: int  # resolution in y-direction (always 96)
    bpc: int  # bits per component
    transform: TransformTuple  # image transform
    size: int  # size of the image in bytes
    image: bytes  # image content
    mask: NotRequired[bytes]  # image mask for transparent images
    name: NotRequired[str]  # image reference name


class ImageInfoDict(TypedDict):
    """Type definition for image info from page.get_image_info().

    See https://pymupdf.readthedocs.io/en/latest/page.html#Page.get_image_info
    """

    number: int  # block number
    bbox: RectLikeTuple  # image bbox on page
    width: int  # original image width
    height: int  # original image height
    colorspace: int  # colorspace.n (component count)
    cs_name: NotRequired[str]  # colorspace name (e.g., "DeviceRGB")
    xres: int  # resolution in x-direction
    yres: int  # resolution in y-direction
    bpc: int  # bits per component
    size: int  # storage occupied by image
    digest: NotRequired[bytes]  # MD5 hashcode (if hashes=True)
    xref: NotRequired[int]  # image xref (if xrefs=True), 0 if inline
    transform: TransformTuple  # transform matrix


class DrawingDict(TypedDict):
    """Type definition for a drawing block from PyMuPDF page.get_drawings().

    See https://pymupdf.readthedocs.io/en/latest/page.html#Page.get_drawings
    """

    rect: RectLike  # PyMuPDF Rect object with .x0, .y0, .x1, .y1 attributes

    closePath: NotRequired[bool]  # whether path is closed
    color: NotRequired[tuple[float, ...] | None]  # stroke color
    dashes: NotRequired[str | None]  # dashed line specification
    even_odd: NotRequired[bool]  # fill behavior for overlaps
    fill: NotRequired[tuple[float, ...] | None]  # fill color
    # list of draw commands: ("l", p1, p2) for lines,
    # ("c", p1, p2, p3, p4) for curves, ("re", rect, orientation) for rects,
    # ("qu", quad) for quads
    items: NotRequired[list[tuple]]
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
    """Type definition for PyMuPDF page.get_text('rawdict') return value.

    See https://pymupdf.readthedocs.io/en/latest/textpage.html#dictionary-structure-of-extractdict-and-extractrawdict
    """

    blocks: list[TextBlockDict | ImageBlockDict]
