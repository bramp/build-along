"""Configuration for parts list classification."""

from __future__ import annotations

from pydantic import BaseModel, Field

from build_a_long.pdf_extract.classifier.score import Weight


class PartsListConfig(BaseModel):
    """Configuration for parts list classification.

    Parts lists are drawing regions that contain Part elements,
    representing the pieces needed for a build step.
    """

    min_score: Weight = 0.5
    """Minimum score threshold for parts list candidates."""

    max_area_ratio: float = Field(default=0.75, ge=0.0, le=1.0)
    """Maximum ratio of page area a parts list can occupy (0.0-1.0).
    
    Drawings larger than this fraction of the page are rejected as they're
    likely the entire page background rather than actual parts lists.
    Default is 0.75 (75% of page area).
    """
