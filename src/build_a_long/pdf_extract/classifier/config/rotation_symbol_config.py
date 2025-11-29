"""Configuration for rotation symbol classification."""

from __future__ import annotations

from pydantic import BaseModel

from build_a_long.pdf_extract.classifier.score import Weight


class RotationSymbolConfig(BaseModel):
    """Configuration for rotation symbol classification.

    Rotation symbols indicate that the builder should rotate the assembled model.
    They appear as small, isolated, square clusters of Drawing elements (~46px).
    """

    min_score: Weight = 0.3
    """Minimum score threshold for rotation symbol candidates."""

    ideal_size: float = 46.0
    """Ideal width/height in points for rotation symbols (used for scoring)."""

    min_size: float = 46.0 * 0.90
    """Minimum width/height in points for rotation symbols (~41.4px)."""

    max_size: float = 46.0 * 1.10
    """Maximum width/height in points for rotation symbols (~50.6px)."""

    min_aspect: float = 0.95
    """Minimum aspect ratio (width/height) - must be close to square."""

    max_aspect: float = 1.05
    """Maximum aspect ratio (width/height) - must be close to square."""

    min_drawings_in_cluster: int = 4
    """Minimum number of drawings to form a rotation symbol cluster."""

    max_drawings_in_cluster: int = 25
    """Maximum number of drawings to form a rotation symbol cluster."""

    size_weight: Weight = 0.5
    """Weight for size score in final rotation symbol score."""

    aspect_weight: Weight = 0.3
    """Weight for aspect ratio score in final rotation symbol score."""

    proximity_weight: Weight = 0.2
    """Weight for proximity to diagram score in final rotation symbol score."""

    cluster_size_score: float = 0.7
    """Base size score for drawing clusters (slightly lower than images)."""

    proximity_close_distance: float = 100.0
    """Distance in points within which proximity score is 1.0."""

    proximity_far_distance: float = 300.0
    """Distance in points beyond which proximity score is 0.0."""
