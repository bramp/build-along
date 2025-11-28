"""Batch classification result class."""

from __future__ import annotations

from pydantic import BaseModel

from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.text_histogram import TextHistogram


class BatchClassificationResult(BaseModel):
    """Results from classifying multiple pages together.

    This class holds both the per-page classification results and the
    global text histogram computed across all pages.
    """

    results: list[ClassificationResult]
    """Per-page classification results, one for each input page"""

    histogram: TextHistogram
    """Global text histogram computed across all pages"""
