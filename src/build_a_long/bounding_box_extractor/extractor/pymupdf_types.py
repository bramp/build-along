"""Type definitions for PyMuPDF (pymupdf) data structures.

See https://pymupdf.readthedocs.io/en/latest/textpage.html#page-dictionary
"""

from typing import NotRequired, TypedDict

# Type alias for bounding box coordinates (x0, y0, x1, y1)
BBoxTuple = tuple[float, float, float, float]


class SpanDict(TypedDict):
    """Type definition for a text span in PyMuPDF rawdict."""

    bbox: BBoxTuple
    text: str


class LineDict(TypedDict):
    """Type definition for a text line in PyMuPDF rawdict."""

    spans: list[SpanDict]


class BlockDict(TypedDict):
    """Type definition for a block in PyMuPDF rawdict."""

    number: int
    type: int
    bbox: BBoxTuple


class TextBlockDict(BlockDict):
    """Type definition for a text block in PyMuPDF rawdict."""

    lines: NotRequired[list[LineDict]]


class ImageBlockDict(BlockDict):
    """Type definition for a image block in PyMuPDF rawdict."""

    # TODO add all the fields as needed


class RawDict(TypedDict):
    """Type definition for PyMuPDF page.get_text('rawdict') return value."""

    blocks: list[BlockDict | TextBlockDict | ImageBlockDict]
