"""Configuration for step number classification."""

from __future__ import annotations

from pydantic import BaseModel

from build_a_long.pdf_extract.classifier.score import Weight


class StepNumberConfig(BaseModel):
    """Configuration for step number classification.

    Step numbers appear on instruction pages to indicate the current build step.
    """

    min_score: Weight = 0.5
    """Minimum score threshold for step number candidates."""

    text_weight: Weight = 0.7
    """Weight for text pattern matching score."""

    font_size_weight: Weight = 0.3
    """Weight for font size matching score."""
