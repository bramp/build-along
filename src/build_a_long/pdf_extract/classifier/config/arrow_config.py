"""Configuration for the arrow classifier."""

from pydantic import BaseModel

from build_a_long.pdf_extract.classifier.score import Weight


class ArrowConfig(BaseModel):
    """Configuration for arrow (arrowhead) classification."""

    min_score: Weight = 0.5
    """Minimum score threshold for arrow candidates."""

    ideal_size: float = 12.0
    """Ideal width/height in points for arrowheads (typically 10-15px)."""

    min_size: float = 5.0
    """Minimum width/height in points for arrowheads."""

    max_size: float = 20.0
    """Maximum width/height in points for arrowheads."""

    min_aspect: float = 0.5
    """Minimum aspect ratio (allows elongated triangles)."""

    max_aspect: float = 2.0
    """Maximum aspect ratio (width/height) for arrowheads."""

    shape_weight: Weight = 0.7
    """Weight for shape score (triangle quality) in final score."""

    size_weight: Weight = 0.3
    """Weight for size score in final arrow score."""
