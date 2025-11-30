"""Configuration for the step count classifier."""

from pydantic import BaseModel

from build_a_long.pdf_extract.classifier.score import Weight


class StepCountConfig(BaseModel):
    """Configuration for step count (substep count) classification.

    StepCounts are count labels like "2x" that appear inside substep callout
    boxes. They use a larger font size than part counts (typically 16pt vs 8pt).
    """

    min_score: Weight = 0.5
    """Minimum score threshold for step count candidates."""

    text_weight: Weight = 0.6
    """Weight for text pattern matching in final score."""

    font_size_weight: Weight = 0.4
    """Weight for font size matching in final score."""
