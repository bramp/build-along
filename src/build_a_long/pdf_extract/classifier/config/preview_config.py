"""Configuration for the preview classifier."""

from __future__ import annotations

from pydantic import BaseModel, Field

from build_a_long.pdf_extract.classifier.score import Weight


class PreviewConfig(BaseModel):
    """Configuration for preview element classification.

    Previews are white rectangular areas containing diagrams that show
    what the completed model (or a section of it) will look like.
    They are identified by:
    - A white/light rectangular box (Drawing element with white fill)
    - One or more images inside forming a diagram
    - Located outside the main instruction/step areas
    """

    min_score: Weight = Field(
        default=0.5, description="Minimum score threshold for preview candidates."
    )

    # Size constraints
    min_width: float = Field(
        default=50.0, description="Minimum width of a Preview element in points."
    )

    min_height: float = Field(
        default=50.0, description="Minimum height of a Preview element in points."
    )

    max_page_width_ratio: float = Field(
        default=0.6,
        description="Maximum width of a Preview as a ratio of the page width.",
    )

    max_page_height_ratio: float = Field(
        default=0.6,
        description="Maximum height of a Preview as a ratio of the page height.",
    )

    # Fill color detection
    white_threshold: float = Field(
        default=0.95,
        description="Minimum RGB value (0-1) for each channel to be considered white.",
    )

    # Score weights
    box_shape_weight: Weight = Field(
        default=0.3, description="Weight for box shape score (rectangular quality)."
    )

    fill_color_weight: Weight = Field(
        default=0.3, description="Weight for fill color score (whiteness)."
    )

    diagram_weight: Weight = Field(
        default=0.4, description="Weight for diagram/image presence inside box."
    )
