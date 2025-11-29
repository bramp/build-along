"""Pydantic models for the LEGO instruction downloader.

This module contains both internal models (DownloadedFile, DownloaderStats) and
shared metadata models (InstructionMetadata, PdfEntry) that are used for serializing
instruction data to JSON files and generating schemas for other applications.
"""

from pathlib import Path

from pydantic import AnyUrl, BaseModel, Field, RootModel

# =============================================================================
# Shared metadata models (source of truth for JSON schema generation)
# =============================================================================


class PdfEntry(BaseModel):
    """Represents a single instruction PDF file.

    This model is serialized to JSON as part of InstructionMetadata and is
    used by both Python and other applications (via generated schemas).
    """

    url: AnyUrl = Field(..., description="The URL to the PDF file.")
    filename: str | None = Field(
        default=None, description="The suggested filename for the PDF."
    )
    filesize: int | None = Field(
        default=None, description="The size of the PDF file in bytes."
    )
    filehash: str | None = Field(
        default=None, description="SHA256 hash of the PDF file content, if available."
    )
    preview_url: AnyUrl | None = Field(
        default=None, description="An optional URL for a preview image of the PDF."
    )
    is_additional_info_booklet: bool | None = Field(
        default=None,
        description=(
            "Indicates if the PDF is an additional info booklet "
            "rather than the main instructions."
        ),
    )
    sequence_number: int | None = Field(
        default=None,
        description=(
            "The sequence number of the PDF in a multi-part instruction set "
            "(e.g., 1 for 1/4)."
        ),
    )
    sequence_total: int | None = Field(
        default=None,
        description=(
            "The total number of PDFs in a multi-part instruction set "
            "(e.g., 4 for 1/4)."
        ),
    )


class InstructionMetadata(BaseModel):
    """Complete metadata for a LEGO set's instructions.

    This is the main model serialized to metadata.json files in each set's
    data directory. It contains all information about a set and its PDFs.
    """

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


class YearlyIndexSummary(BaseModel):
    """Summary metadata for a single year's index file."""

    year: int = Field(..., description="The year of the index.")
    count: int = Field(..., description="The number of sets in the index.")
    filesize: int = Field(..., description="The size of the index file in bytes.")
    filename: str = Field(..., description="The filename of the index file.")


class MainIndex(RootModel[list[YearlyIndexSummary]]):
    """The main index file containing summaries for each yearly index."""

    root: list[YearlyIndexSummary] = Field(
        ..., description="A list of summaries for each yearly index."
    )


class YearlyIndex(RootModel[list[InstructionMetadata]]):
    """A yearly index file containing metadata for all sets from that year."""

    root: list[InstructionMetadata] = Field(
        ..., description="A list of instruction metadata for a single year."
    )


# =============================================================================
# Internal downloader models (not shared)
# =============================================================================


class DownloadedFile(BaseModel):
    """Represents a file that has been downloaded to disk."""

    path: Path = Field(..., description="The path to the downloaded file.")
    size: int = Field(..., description="The size of the file in bytes.")
    hash: str | None = Field(
        default=None, description="SHA256 hash of the file content, if available."
    )


class DownloadUrl(BaseModel):
    """A URL to download with optional metadata."""

    url: AnyUrl = Field(..., description="The primary download URL.")
    preview_url: AnyUrl | None = Field(
        default=None, description="An optional URL for a preview image or page."
    )
    sequence_number: int | None = Field(
        default=None,
        description="The sequence number of this instruction (e.g., 1 of 3).",
    )
    sequence_total: int | None = Field(
        default=None, description="The total number of instructions in the sequence."
    )
    is_additional_info_booklet: bool | None = Field(
        default=None,
        description=(
            "Indicates if the instruction is a supplemental or additional info booklet."
        ),
    )


class DownloaderStats(BaseModel):
    """Statistics for the downloader execution."""

    sets_processed: int = 0
    sets_not_found: int = 0
    sets_found: int = 0
    pdfs_found: int = 0
    pdfs_downloaded: int = 0
    pdfs_skipped: int = 0  # Cached or already exists
