"""Configuration for bag number classification."""

from __future__ import annotations

from pydantic import BaseModel

from build_a_long.pdf_extract.classifier.score import Weight


class BagNumberConfig(BaseModel):
    """Configuration for bag number classification.

    Bag numbers are large numbers (1, 2, 3...) indicating which bag of parts to open.
    """

    min_score: Weight = 0.1
    """Minimum score threshold for bag number candidates."""

    text_weight: Weight = 0.4
    """Weight for text pattern matching score."""

    position_weight: Weight = 0.4
    """Weight for position score."""

    font_size_weight: Weight = 0.2
    """Weight for font size score."""
