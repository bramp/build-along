"""Pydantic models for JSON output serialization."""

from pydantic import BaseModel

from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)


class DebugOutput(BaseModel):
    """Complete debug output including raw blocks and all classification candidates.

    This format is useful for debugging and understanding the classification process.
    It includes all intermediate data: raw blocks, all candidates (including rejected
    ones), removal reasons, and warnings.
    """

    pages: list[ClassificationResult]
