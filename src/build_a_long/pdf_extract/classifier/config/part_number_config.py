"""Configuration for part number (element ID) classification."""

from __future__ import annotations

from pydantic import BaseModel, Field

from build_a_long.pdf_extract.classifier.score import Weight


class PartNumberConfig(BaseModel):
    """Configuration for part number (element ID) classification.

    Part numbers are typically 6-7 digit LEGO element IDs that appear
    on catalog pages below part counts.
    """

    min_score: Weight = Field(
        default=0.4, description="Minimum score threshold for part number candidates."
    )
