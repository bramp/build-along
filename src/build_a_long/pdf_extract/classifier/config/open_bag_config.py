"""Configuration for open bag classification."""

from __future__ import annotations

from pydantic import BaseModel, Field

from build_a_long.pdf_extract.classifier.score import Weight


class OpenBagConfig(BaseModel):
    """Configuration for open bag classification.

    OpenBag elements mark when a numbered bag of pieces should be opened.
    They consist of an optional bag number surrounded by a cluster of images
    forming a bag icon graphic, typically at the top-left of a page.
    """

    min_score: Weight = Field(
        default=0.5, description="Minimum score threshold for open bag candidates."
    )

    min_circle_size: float = Field(
        default=150.0,
        description=(
            "Minimum width/height in points for circular bag icon outlines. "
            "Circles range from ~176 to ~240 points. Using 150 to catch smaller ones. "
            "Also used as minimum size for image cluster fallback detection."
        ),
    )

    min_icon_aspect_ratio: float = Field(
        default=0.90,
        description="Minimum aspect ratio (width/height) for bag icon images.",
    )

    max_icon_aspect_ratio: float = Field(
        default=1.10,
        description="Maximum aspect ratio (width/height) for bag icon images.",
    )

    max_icon_x_ratio: float = Field(
        default=0.15,
        description=(
            "Maximum x position as ratio of page width. Bag icons are typically "
            "positioned very close to the top-left corner, around x=14 on a 552pt wide "
            "page (~2.5%). Using 15% to allow some margin."
        ),
    )

    max_icon_y_ratio: float = Field(
        default=0.15,
        description=(
            "Maximum y position as ratio of page height. Bag icons are typically "
            "positioned very close to the top-left corner, around y=14 on a 496pt tall "
            "page (~2.8%). Using 15% to allow some margin."
        ),
    )
