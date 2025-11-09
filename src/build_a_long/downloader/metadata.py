"""Dataclasses for representing LEGO set metadata.

This module uses Pydantic for (de)serialization so callers can use
``.model_dump()``, ``.model_dump_json()``, ``model_validate`` and
``model_validate_json`` directly.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class DownloadedFile(BaseModel):
    """Represents a downloaded file with its path, size, and hash."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    size: int
    hash: str | None


class DownloadUrl(BaseModel):
    """Holds the URL and preview URL for a download."""

    url: str
    preview_url: str | None


class PdfEntry(BaseModel):
    """Represents a single instruction PDF file."""

    url: str
    filename: str
    preview_url: str | None = None
    filesize: int | None = None
    filehash: str | None = None  # SHA256 hash of the file content


class InstructionMetadata(BaseModel):
    """Complete metadata for a LEGO set's instructions."""

    set: str
    locale: str
    name: str | None = None
    theme: str | None = None
    age: str | None = None
    pieces: int | None = None
    year: int | None = None
    set_image_url: str | None = None
    pdfs: list[PdfEntry] = Field(default_factory=list)
