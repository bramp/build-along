"""Configuration for background classification."""

from __future__ import annotations

from pydantic import BaseModel, Field

from build_a_long.pdf_extract.classifier.score import Weight


class BackgroundConfig(BaseModel):
    """Configuration for background classification.

    Backgrounds are large drawing or image blocks that cover most or all
    of the page, forming the visual backdrop for instruction content.
    """

    min_score: Weight = Field(
        default=0.4, description="Minimum score threshold for background candidates."
    )

    min_coverage_ratio: float = Field(
        default=0.85,
        description=(
            "Minimum coverage as a ratio of page area. A background element must cover "
            "at least this much of the page area to be considered valid. Default is "
            "0.85 (85% of page area)."
        ),
    )

    edge_margin: float = Field(
        default=4.0,
        description=(
            "Margin in points for background edge detection. Full-page backgrounds "
            "are expected to touch or be near page edges within this margin. Also, "
            "elements contained entirely within this margin from any page edge are "
            "classified as background (borders, bleed lines, artifacts)."
        ),
    )
