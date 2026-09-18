"""LEGO instruction schemas and models.

This module re-exports the shared metadata models from the downloader module
for backward compatibility.
"""

from build_a_long.downloader.models import (
    ImageEntry,
    InstructionMetadata,
    MainIndex,
    PdfEntry,
    VideoEntry,
    YearlyIndex,
    YearlyIndexSummary,
)

__all__ = [
    "ImageEntry",
    "InstructionMetadata",
    "MainIndex",
    "PdfEntry",
    "VideoEntry",
    "YearlyIndex",
    "YearlyIndexSummary",
]
