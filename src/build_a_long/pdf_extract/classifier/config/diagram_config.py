"""Configuration for diagram classification."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DiagramConfig(BaseModel):
    """Configuration for diagram classification."""

    min_score: float = Field(
        default=0.6,
        description="Minimum score to be considered a valid diagram.",
    )

    max_area_ratio: float = Field(
        default=0.95,
        description=(
            "Maximum area as a ratio of page area. "
            "Images larger than this are filtered out (likely backgrounds)."
        ),
    )
