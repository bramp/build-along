"""LEGO instruction schemas and models.

This module re-exports the shared metadata models from the downloader module
for backward compatibility.
"""

from build_a_long.downloader.models import (
    InstructionMetadata,
    MainIndex,
    PdfEntry,
    YearlyIndex,
    YearlyIndexSummary,
)

__all__ = [
    "InstructionMetadata",
    "MainIndex",
    "PdfEntry",
    "YearlyIndex",
    "YearlyIndexSummary",
]
