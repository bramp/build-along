"""Configuration for page number classification."""

from __future__ import annotations

from pydantic import BaseModel, Field

from build_a_long.pdf_extract.classifier.score import Weight


class PageNumberConfig(BaseModel):
    """Configuration for page number classification.

    Page numbers typically appear in the bottom corners of LEGO instruction pages.
    """

    min_score: Weight = Field(
        default=0.5, description="Minimum score threshold for page number candidates."
    )

    text_weight: Weight = Field(
        default=0.7, description="Weight for text pattern matching score."
    )

    position_weight: Weight = Field(
        default=0.3,
        description="Weight for position score (proximity to bottom corners).",
    )

    position_scale: float = Field(
        default=50.0, description="Scale factor for exponential position scoring."
    )

    page_value_weight: Weight = Field(
        default=1.0,
        description="Weight for page value matching score (expected page number).",
    )

    font_size_weight: Weight = Field(
        default=0.1, description="Weight for font size matching score."
    )
