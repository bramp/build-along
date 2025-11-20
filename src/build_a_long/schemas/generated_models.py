# Generated from schemas/openapi.yaml - DO NOT EDIT MANUALLY
# Use 'pants run src/build_a_long/schemas:generate_models' to regenerate this file

from __future__ import annotations

from pathlib import Path

from pydantic import AnyUrl, BaseModel, Field


class DownloadedFile(BaseModel):
    path: Path = Field(..., description="The path to the downloaded file.")
    size: int = Field(..., description="The size of the file in bytes.")
    hash: str | None = Field(
        default=None, description="SHA256 hash of the file content, if available."
    )


class DownloadUrl(BaseModel):
    url: AnyUrl = Field(..., description="The primary download URL.")
    preview_url: AnyUrl | None = Field(
        default=None, description="An optional URL for a preview image or page."
    )


class PdfEntry(BaseModel):
    url: AnyUrl = Field(..., description="The URL to the PDF file.")
    filename: str = Field(..., description="The suggested filename for the PDF.")
    preview_url: AnyUrl | None = Field(
        default=None, description="An optional URL for a preview image of the PDF."
    )
    filesize: int | None = Field(
        default=None, description="The size of the PDF file in bytes."
    )
    filehash: str | None = Field(
        default=None, description="SHA256 hash of the PDF file content, if available."
    )


class InstructionMetadata(BaseModel):
    set: str = Field(..., description="The unique identifier for the LEGO set.")
    locale: str = Field(
        ..., description='The locale for the instructions (e.g., "en-US").'
    )
    name: str | None = Field(default=None, description="The name of the LEGO set.")
    theme: str | None = Field(
        default=None,
        description='The theme of the LEGO set (e.g., "City", "Star Wars").',
    )
    age: str | None = Field(
        default=None, description="The recommended age range for the set."
    )
    pieces: int | None = Field(
        default=None, description="The number of pieces in the set."
    )
    year: int | None = Field(default=None, description="The year the set was released.")
    set_image_url: AnyUrl | None = Field(
        default=None, description="URL to an image of the LEGO set."
    )
    pdfs: list[PdfEntry] = Field(
        default=[], description="A list of PDF instruction entries for the set."
    )
