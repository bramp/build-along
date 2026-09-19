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
    preview_alt: str | None = Field(
        default=None,
        description="The alt text for the preview cover image, if available.",
    )


class ImageEntry(BaseModel):
    """Represents a product image or asset from LEGO.com."""

    id: str | None = Field(default=None, description="The asset identifier.")
    url: AnyUrl = Field(..., description="The URL to the image.")
    alt_text: str | None = Field(
        default=None, description="Alt text or description of the image."
    )


class VideoEntry(BaseModel):
    """Represents a product video from LEGO.com."""

    id: str | None = Field(default=None, description="The video asset identifier.")
    title: str | None = Field(default=None, description="The video title.")
    description: str | None = Field(default=None, description="The video description.")
    url: AnyUrl | None = Field(
        default=None, description="The URL to the video file, if available."
    )
    quality: str | None = Field(
        default=None,
        description="The video stream quality (e.g., 'Highest').",
    )


class Dimensions(BaseModel):
    """Dimensions of the assembled LEGO model in centimeters."""

    height: float | None = Field(default=None, description="Height in centimeters.")
    width: float | None = Field(default=None, description="Width in centimeters.")
    depth: float | None = Field(default=None, description="Depth in centimeters.")


class InstructionMetadata(BaseModel):
    """Complete metadata for a LEGO set's instructions.

    This is the main model serialized to metadata.json files in each set's
    data directory. It contains all information about a set and its PDFs.
    """

    # TODO maybe rename 'set' to 'set_number' for clarity
    set: str = Field(..., description="The unique identifier for the LEGO set.")

    # TODO Should we add a 'element_id' field, to uniquely identify instructions
    # across sets?

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
    set_image_alt: str | None = Field(
        default=None, description="Alt text for the image of the LEGO set."
    )
    description: str | None = Field(
        default=None, description="The product description for the LEGO set."
    )
    features_text: str | None = Field(
        default=None,
        description="The features and details text (bullet points list, e.g. <ul>...</ul>) with any duplicate description prefix removed.",
    )
    meta_description: str | None = Field(
        default=None, description="A short summary description of the set."
    )
    meta_title: str | None = Field(
        default=None, description="The meta title of the set product page."
    )
    slug: str | None = Field(
        default=None, description="The URL slug of the set on LEGO.com."
    )
    hires_image_url: AnyUrl | None = Field(
        default=None,
        description="URL to a high-resolution image of the LEGO set.",
    )
    thumbnail_image_url: AnyUrl | None = Field(
        default=None,
        description="URL to a thumbnail image of the LEGO set.",
    )
    images: list[ImageEntry] = Field(
        default=[],
        description="List of all product and gallery images available for the set.",
    )
    videos: list[VideoEntry] = Field(
        default=[],
        description="List of all product videos available for the set.",
    )
    categories: list[str] = Field(
        default=[],
        description="Categories or tags associated with the set.",
    )
    brand: str | None = Field(
        default=None,
        description="The brand category of the set (e.g., 'Star Wars', 'Icons').",
    )
    minifigure_count: int | None = Field(
        default=None,
        description="The number of minifigures included in the set.",
    )
    dimensions: Dimensions | None = Field(
        default=None,
        description="Dimensions of the assembled set in centimeters.",
    )
    availability_status: str | None = Field(
        default=None,
        description="Availability status on LEGO.com (e.g. 'E_AVAILABLE', 'R_RETIRED').",
    )
    availability_text: str | None = Field(
        default=None,
        description="Human-readable availability text (e.g. 'Available now', 'Retired product').",
    )
    price_formatted: str | None = Field(
        default=None,
        description="Formatted price (e.g. '$999.99').",
    )
    price_cents: int | None = Field(
        default=None,
        description="Price in cents.",
    )
    currency: str | None = Field(
        default=None,
        description="Currency code (e.g. 'USD').",
    )
    rating: float | None = Field(
        default=None,
        description="Average customer review rating out of 5.",
    )
    sku: str | None = Field(
        default=None,
        description="The LEGO shop item/SKU number.",
    )
    flags: list[str] = Field(
        default=[],
        description="Special badges or flags (e.g. 'Exclusives', 'New').",
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
    filesize: int | None = Field(
        default=None,
        description="The size of the PDF file in bytes if available from metadata.",
    )
    preview_alt: str | None = Field(
        default=None,
        description="Alt text for the preview cover image, if available.",
    )


class DownloaderStats(BaseModel):
    """Statistics for the downloader execution."""

    sets_processed: int = 0
    sets_not_found: int = 0
    sets_found: int = 0
    pdfs_found: int = 0
    pdfs_downloaded: int = 0
    pdfs_skipped: int = 0  # Cached or already exists
