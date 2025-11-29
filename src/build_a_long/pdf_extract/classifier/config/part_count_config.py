"""Configuration for part count classification."""

from __future__ import annotations

from pydantic import BaseModel

from build_a_long.pdf_extract.classifier.score import Weight


class PartCountConfig(BaseModel):
    """Configuration for part count classification.

    Part counts appear as text like "2x", "3X", or "5×" indicating how many
    of a particular piece are needed.
    """

    min_score: Weight = 0.5
    """Minimum score threshold for part count candidates."""

    text_weight: Weight = 0.7
    """Weight for text pattern matching score."""

    font_size_weight: Weight = 0.3
    """Weight for font size matching score."""
