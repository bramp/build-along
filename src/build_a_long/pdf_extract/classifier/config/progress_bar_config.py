"""Configuration for progress bar classification."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProgressBarConfig(BaseModel):
    """Configuration for progress bar classification."""

    # TODO We should re-use this in ProgressIndicator, or make sure the values align..
    max_bottom_margin_ratio: float = Field(
        default=0.2,
        description="Maximum distance from bottom as a ratio of page height (0.0-1.0).",
    )

    max_page_number_proximity_ratio: float = Field(
        default=0.3,
        description="Maximum horizontal distance from page number as a ratio of page width.",
    )

    page_number_proximity_multiplier: float = Field(
        default=1.2,
        description="Score multiplier when close to page number.",
    )

    min_width_ratio: float = Field(
        default=0.3,
        description="Minimum width as a ratio of page width.",
    )

    max_score_width_ratio: float = Field(
        default=0.8,
        description="Width ratio at which width score is maximized.",
    )

    min_aspect_ratio: float = Field(
        default=3.0,
        description="Minimum width/height aspect ratio.",
    )

    ideal_aspect_ratio: float = Field(
        default=10.0,
        description="Aspect ratio at which aspect ratio score is maximized.",
    )

    # TODO We should re-use this in ProgressIndicator, or make sure the values align..
    indicator_search_margin: float = Field(
        default=10.0,
        description="Horizontal margin in pixels when searching for indicator.",
    )

    # TODO We should re-use this in ProgressIndicator, or make sure the values align..
    overlap_expansion_margin: float = Field(
        default=5.0,
        description="Vertical expansion in pixels when finding overlapping blocks.",
    )
