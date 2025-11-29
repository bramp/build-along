"""Configuration for page number classification."""

from __future__ import annotations

from pydantic import BaseModel

from build_a_long.pdf_extract.classifier.score import Weight


class PageNumberConfig(BaseModel):
    """Configuration for page number classification.

    Page numbers typically appear in the bottom corners of LEGO instruction pages.
    """

    min_score: Weight = 0.5
    """Minimum score threshold for page number candidates."""

    text_weight: Weight = 0.7
    """Weight for text pattern matching score."""

    position_weight: Weight = 0.3
    """Weight for position score (proximity to bottom corners)."""

    position_scale: float = 50.0
    """Scale factor for exponential position scoring."""

    page_value_weight: Weight = 1.0
    """Weight for page value matching score (expected page number)."""

    font_size_weight: Weight = 0.1
    """Weight for font size matching score."""
