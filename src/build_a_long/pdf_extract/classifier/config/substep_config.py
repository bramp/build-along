"""Configuration for the substep classifier."""

from pydantic import BaseModel

from build_a_long.pdf_extract.classifier.score import Weight


class SubStepConfig(BaseModel):
    """Configuration for substep (callout box) classification.

    SubSteps are small callout boxes showing sub-assemblies to build multiple times.
    They are identified by:
    - A white/light rectangular box (Drawing element)
    - A count label inside (e.g., "2x") with larger font than part counts
    - A diagram/image of the sub-assembly
    - Often an arrow pointing to the main diagram
    """

    min_score: Weight = 0.5
    """Minimum score threshold for substep candidates."""

    # Score weights
    box_shape_weight: Weight = 0.3
    """Weight for box shape score (rectangular quality)."""

    count_weight: Weight = 0.4
    """Weight for count label presence and quality."""

    diagram_weight: Weight = 0.3
    """Weight for diagram/image presence inside box."""
