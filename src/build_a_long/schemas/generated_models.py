# Generated from schemas/openapi.yaml - DO NOT EDIT MANUALLY
# Use 'pants run src/build_a_long/schemas:generate_models' to regenerate this file

from __future__ import annotations

from pydantic import AnyUrl, BaseModel, Field, RootModel


class PdfEntry(BaseModel):
    url: AnyUrl = Field(..., description="The URL to the PDF file.")
    filename: str = Field(..., description="The suggested filename for the PDF.")
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
        description="Indicates if the PDF is an additional info booklet rather than the main instructions.",
    )
    sequence_number: int | None = Field(
        default=None,
        description="The sequence number of the PDF in a multi-part instruction set (e.g., 1 for 1/4).",
    )
    sequence_total: int | None = Field(
        default=None,
        description="The total number of PDFs in a multi-part instruction set (e.g., 4 for 1/4).",
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


class YearlyIndexSummary(BaseModel):
    year: int = Field(..., description="The year of the index.")
    count: int = Field(..., description="The number of sets in the index.")
    filesize: int = Field(..., description="The size of the index file in bytes.")
    filename: str = Field(..., description="The filename of the index file.")


class MainIndex(RootModel[list[YearlyIndexSummary]]):
    root: list[YearlyIndexSummary] = Field(
        ..., description="A list of summaries for each yearly index."
    )


class YearlyIndex(RootModel[list[InstructionMetadata]]):
    root: list[InstructionMetadata] = Field(
        ..., description="A list of instruction metadata for a single year."
    )
