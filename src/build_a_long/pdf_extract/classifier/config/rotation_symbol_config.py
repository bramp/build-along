"""Configuration for rotation symbol classification."""

from __future__ import annotations

from pydantic import BaseModel, Field

from build_a_long.pdf_extract.classifier.score import Weight


class RotationSymbolConfig(BaseModel):
    """Configuration for rotation symbol classification.

    Rotation symbols indicate that the builder should rotate the assembled model.
    They appear as small, isolated, square clusters of Drawing elements (~46px).
    """

    min_score: Weight = Field(
        default=0.3,
        description="Minimum score threshold for rotation symbol candidates.",
    )

    ideal_size: float = Field(
        default=46.0,
        description="Ideal width/height in points for rotation symbols (used for scoring).",
    )

    min_size: float = Field(
        default=25.0,
        description="Minimum width/height in points for rotation symbols.",
    )

    max_size: float = Field(
        default=50.6,
        description="Maximum width/height in points for rotation symbols (~50.6px).",
    )

    min_aspect_ratio: float = Field(
        default=0.95,
        description="Minimum aspect ratio (width/height) - must be close to square.",
    )

    max_aspect_ratio: float = Field(
        default=1.05,
        description="Maximum aspect ratio (width/height) - must be close to square.",
    )

    min_cluster_drawing_count: int = Field(
        default=4,
        description="Minimum number of drawings to form a rotation symbol cluster.",
    )

    max_cluster_drawing_count: int = Field(
        default=25,
        description="Maximum number of drawings to form a rotation symbol cluster.",
    )

    size_weight: Weight = Field(
        default=0.5, description="Weight for size score in final rotation symbol score."
    )

    aspect_weight: Weight = Field(
        default=0.3,
        description="Weight for aspect ratio score in final rotation symbol score.",
    )

    proximity_weight: Weight = Field(
        default=0.2,
        description="Weight for proximity to diagram score in final rotation symbol score.",
    )

    cluster_size_score: Weight = Field(
        default=0.7,
        description="Base size score for drawing clusters (slightly lower than images).",
    )

    proximity_close_distance: float = Field(
        default=100.0,
        description="Distance in points within which proximity score is 1.0.",
    )

    proximity_far_distance: float = Field(
        default=300.0,
        description="Distance in points beyond which proximity score is 0.0.",
    )
