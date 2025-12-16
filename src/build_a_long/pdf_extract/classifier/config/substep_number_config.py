"""Configuration for substep number classification."""

from __future__ import annotations

from pydantic import BaseModel, Field

from build_a_long.pdf_extract.classifier.score import Weight


class SubStepNumberConfig(BaseModel):
    """Configuration for substep number classification.

    Substep numbers are smaller step numbers that appear inside subassembly
    boxes or as naked substeps alongside main steps. They typically have a
    smaller font size than regular step numbers and lower values (1-10).
    """

    min_score: Weight = Field(
        default=0.5,
        description="Minimum score threshold for substep number candidates.",
    )

    text_weight: Weight = Field(
        default=0.4, description="Weight for text pattern matching score."
    )

    font_size_weight: Weight = Field(
        default=0.3, description="Weight for font size matching score."
    )

    value_weight: Weight = Field(
        default=0.3, description="Weight for step value score (lower is better)."
    )

    max_value: int = Field(
        default=10,
        description="Maximum value for a substep number (substeps are typically 1-10).",
    )

    size_ratio: float = Field(
        default=0.85,
        description=(
            "Font size ratio relative to step_number_size. "
            "Substep font must be smaller than step_number_size * size_ratio."
        ),
    )
