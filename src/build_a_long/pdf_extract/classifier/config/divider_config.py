"""Configuration for divider classification."""

from __future__ import annotations

from pydantic import BaseModel, Field

from build_a_long.pdf_extract.classifier.score import Weight


class DividerConfig(BaseModel):
    """Configuration for divider classification.

    Dividers are thin lines that separate sections of a page. They run
    vertically or horizontally across a significant portion of the page.
    """

    min_score: Weight = Field(
        default=0.4, description="Minimum score threshold for divider candidates."
    )

    min_length_ratio: float = Field(
        default=0.4,
        description=(
            "Minimum length as a ratio of page height (vertical) or width "
            "(horizontal). A divider must span at least this much of the page to be "
            "considered valid. Default is 0.4 (40% of page dimension)."
        ),
    )

    max_thickness: float = Field(
        default=5.0,
        description=(
            "Maximum thickness in points for the divider line. Dividers are thin "
            "lines, so this limits how thick they can be. A value of 0 indicates a "
            "stroke-only line (no fill width)."
        ),
    )

    # TODO Use hints from the page to determine the margin.
    edge_margin: float = Field(
        default=5.0,
        description=(
            "Margin in points from page edge to ignore dividers. Dividers within this "
            "distance from the page boundary are considered page borders and are "
            "filtered out. This prevents detecting page border lines as content "
            "dividers."
        ),
    )
