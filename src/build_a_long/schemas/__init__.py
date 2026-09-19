"""LEGO instruction schemas and models.

This module re-exports the shared metadata models from the downloader module
for backward compatibility.
"""

from build_a_long.downloader.models import (
    Dimensions,
    ImageEntry,
    InstructionMetadata,
    MainIndex,
    PdfEntry,
    VideoEntry,
    YearlyIndex,
    YearlyIndexSummary,
)

__all__ = [
    "Dimensions",
    "ImageEntry",
    "InstructionMetadata",
    "MainIndex",
    "PdfEntry",
    "VideoEntry",
    "YearlyIndex",
    "YearlyIndexSummary",
]
