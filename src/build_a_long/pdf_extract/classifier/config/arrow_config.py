"""Configuration for the arrow classifier."""

from __future__ import annotations

from pydantic import BaseModel, Field

from build_a_long.pdf_extract.classifier.score import Weight


class ArrowConfig(BaseModel):
    """Configuration for arrow (arrowhead) classification."""

    min_score: Weight = Field(
        default=0.5, description="Minimum score threshold for arrow candidates."
    )

    ideal_size: float = Field(
        default=12.0,
        description="Ideal width/height in points for arrowheads (typically 10-15px).",
    )

    min_size: float = Field(
        default=5.0, description="Minimum width/height in points for arrowheads."
    )

    max_size: float = Field(
        default=20.0, description="Maximum width/height in points for arrowheads."
    )

    min_aspect_ratio: float = Field(
        default=0.5, description="Minimum aspect ratio (allows elongated triangles)."
    )

    max_aspect_ratio: float = Field(
        default=2.0, description="Maximum aspect ratio (width/height) for arrowheads."
    )

    shape_weight: Weight = Field(
        default=0.7,
        description="Weight for shape score (triangle quality) in final score.",
    )

    size_weight: Weight = Field(
        default=0.3, description="Weight for size score in final arrow score."
    )
