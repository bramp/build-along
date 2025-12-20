"""Configuration for progress bar classification."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProgressBarConfig(BaseModel):
    """Configuration for progress bar and progress bar indicator classification."""

    # --- Bar settings ---

    min_score: float = Field(
        default=0.6,
        description="Minimum score to be considered a valid progress bar.",
    )

    max_bottom_margin_ratio: float = Field(
        default=0.2,
        description="Maximum distance from bottom as a ratio of page height (0.0-1.0).",
    )

    max_page_number_proximity_ratio: float = Field(
        default=0.3,
        description=(
            "Maximum horizontal distance from page number as a ratio of page width."
        ),
    )

    page_number_proximity_multiplier: float = Field(
        default=1.2,
        description="Score multiplier when close to page number.",
    )

    min_width_ratio: float = Field(
        default=0.4,
        description=(
            "Minimum width as a ratio of page width for individual bar candidates. "
            "The build phase will merge other blocks at the same y-position."
        ),
    )

    min_merged_width_ratio: float = Field(
        default=0.9,
        description=(
            "Minimum width as a ratio of page width for the final merged bar. "
            "After merging, the combined bar must span at least this much."
        ),
    )

    max_score_width_ratio: float = Field(
        default=0.8,
        description="Width ratio at which width score is maximized.",
    )

    min_aspect_ratio: float = Field(
        default=3.0,
        description="Minimum width/height aspect ratio for the bar.",
    )

    ideal_aspect_ratio: float = Field(
        default=10.0,
        description="Aspect ratio at which aspect ratio score is maximized.",
    )

    indicator_search_margin: float = Field(
        default=10.0,
        description="Horizontal margin in pixels when searching for indicator.",
    )

    overlap_expansion_margin: float = Field(
        default=5.0,
        description="Vertical expansion in pixels when finding overlapping blocks.",
    )

    bar_merge_y_tolerance: float = Field(
        default=5.0,
        description="Max y-position difference (in points) for merging bar segments.",
    )

    bar_merge_height_tolerance: float = Field(
        default=2.0,
        description="Maximum height difference (in points) for merging bar segments.",
    )

    # --- Indicator settings ---

    indicator_min_size: float = Field(
        default=8.0,
        description=(
            "Minimum width/height for the indicator (filters out tiny elements)."
        ),
    )

    indicator_max_size: float = Field(
        default=25.0,
        description=(
            "Maximum width/height for the indicator (filters out large elements)."
        ),
    )

    indicator_min_aspect_ratio: float = Field(
        default=0.9,
        description=(
            "Minimum aspect ratio (width/height) for indicator. "
            "Allows tolerance for floating point in nearly-square circles."
        ),
    )

    indicator_max_aspect_ratio: float = Field(
        default=1.5,
        description=(
            "Maximum aspect ratio (width/height or height/width) for indicator. "
            "1.0 = perfect square."
        ),
    )

    indicator_max_bottom_margin_ratio: float = Field(
        default=0.25,
        description="Max distance from page bottom as ratio of page height.",
    )

    indicator_shadow_margin: float = Field(
        default=5.0,
        description="Margin (in points) for finding shadow blocks around indicator.",
    )
