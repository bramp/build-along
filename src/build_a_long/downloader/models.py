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
    sequence_number: int | None = Field(
        default=None,
        description="The sequence number of this instruction (e.g., 1 of 3).",
    )
    sequence_total: int | None = Field(
        default=None, description="The total number of instructions in the sequence."
    )
    is_additional_info_booklet: bool | None = Field(
        default=None,
        description="Indicates if the instruction is a supplemental or additional info booklet.",
    )


class DownloaderStats(BaseModel):
    """Statistics for the downloader execution."""

    sets_processed: int = 0
    sets_not_found: int = 0
    sets_found: int = 0
    pdfs_found: int = 0
    pdfs_downloaded: int = 0
    pdfs_skipped: int = 0  # Cached or already exists
