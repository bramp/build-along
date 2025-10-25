"""Dataclasses for representing LEGO set metadata."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DownloadUrl:
    """Holds the URL and preview URL for a download."""

    url: str
    preview_url: Optional[str]


@dataclass
class PdfEntry:
    """Represents a single instruction PDF file."""

    url: str
    filename: str
    preview_url: Optional[str] = None


@dataclass
class Metadata:
    """Complete metadata for a LEGO set's instructions."""

    set: str
    locale: str
    name: Optional[str] = None
    theme: Optional[str] = None
    age: Optional[str] = None
    pieces: Optional[int] = None
    year: Optional[int] = None
    set_image_url: Optional[str] = None
    pdfs: List[PdfEntry] = field(default_factory=list)
