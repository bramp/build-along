"""Dataclasses for representing LEGO set metadata.

This module uses dataclasses-json for (de)serialization so callers can use
``.to_dict()``, ``.to_json()``, ``from_dict`` and ``from_json`` directly.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from dataclasses_json import DataClassJsonMixin, config


@dataclass
class File(DataClassJsonMixin):
    """Represents a downloaded file with its path, size, and hash."""

    path: Path = field(metadata=config(encoder=str, decoder=Path))
    size: int
    hash: Optional[str]


@dataclass
class DownloadUrl(DataClassJsonMixin):
    """Holds the URL and preview URL for a download."""

    url: str
    preview_url: Optional[str]


@dataclass
class PdfEntry(DataClassJsonMixin):
    """Represents a single instruction PDF file."""

    url: str
    filename: str
    preview_url: Optional[str] = None
    filesize: Optional[int] = None
    filehash: Optional[str] = None  # SHA256 hash of the file content


@dataclass
class Metadata(DataClassJsonMixin):
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
