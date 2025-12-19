"""Configuration for the progress bar indicator classifier."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProgressBarIndicatorConfig(BaseModel):
    """Configuration for progress bar indicator classification.

    The progress bar indicator is a circular graphic on top of the progress bar
    that shows how far through the instructions the reader is. It's roughly
    square (circle) and typically 10-20 pixels in size.
    """

    min_size: float = Field(
        default=8.0,
        description=(
            "Minimum width/height for the indicator (filters out tiny elements)."
        ),
    )

    max_size: float = Field(
        default=25.0,
        description=(
            "Maximum width/height for the indicator (filters out large elements)."
        ),
    )

    max_aspect_ratio: float = Field(
        default=1.5,
        description=(
            "Maximum aspect ratio (width/height or height/width). 1.0 = perfect square."
        ),
    )

    max_bottom_margin_ratio: float = Field(
        default=0.25,
        description="Maximum distance from bottom of page as a ratio of page height.",
    )
