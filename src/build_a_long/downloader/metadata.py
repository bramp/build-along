"""Dataclasses for representing LEGO set metadata.

This module uses dataclasses-json for (de)serialization so callers can use
``.to_dict()``, ``.to_json()``, ``from_dict`` and ``from_json`` directly.
"""

from dataclasses import dataclass, field
from pathlib import Path

from dataclass_wizard import JSONPyWizard


@dataclass
class File(JSONPyWizard):
    """Represents a downloaded file with its path, size, and hash."""

    class _(JSONPyWizard.Meta):
        raise_on_unknown_json_key = True

    path: Path
    size: int
    hash: str | None


@dataclass
class DownloadUrl(JSONPyWizard):
    """Holds the URL and preview URL for a download."""

    url: str
    preview_url: str | None


@dataclass
class PdfEntry(JSONPyWizard):
    """Represents a single instruction PDF file."""

    url: str
    filename: str
    preview_url: str | None = None
    filesize: int | None = None
    filehash: str | None = None  # SHA256 hash of the file content


@dataclass
class Metadata(JSONPyWizard):
    """Complete metadata for a LEGO set's instructions."""

    set: str
    locale: str
    name: str | None = None
    theme: str | None = None
    age: str | None = None
    pieces: int | None = None
    year: int | None = None
    set_image_url: str | None = None
    pdfs: list[PdfEntry] = field(default_factory=list)
