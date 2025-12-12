"""Configuration for the subassembly classifier."""

from __future__ import annotations

from pydantic import BaseModel, Field

from build_a_long.pdf_extract.classifier.score import Weight


class SubAssemblyConfig(BaseModel):
    """Configuration for subassembly (callout box) classification.

    SubAssemblies are small callout boxes showing sub-assemblies to build multiple
    times. They are identified by:
    - A white/light rectangular box (Drawing element)
    - A count label inside (e.g., "2x") with larger font than part counts
    - A diagram/image of the sub-assembly
    - Often an arrow pointing to the main diagram
    """

    min_score: Weight = Field(
        default=0.5, description="Minimum score threshold for subassembly candidates."
    )

    # Minimum size configuration
    min_part_width: float = Field(
        default=20.0, description="Minimum width of a Part element in points."
    )

    min_part_height: float = Field(
        default=15.0, description="Minimum height of a Part element in points."
    )

    min_size_part_multiplier: float = Field(
        default=1.5,
        description="SubAssembly must be at least this many times larger than the minimum Part size.",
    )

    max_page_width_ratio: float = Field(
        default=0.5,
        description="Maximum width of a SubAssembly as a ratio of the page width.",
    )

    max_page_height_ratio: float = Field(
        default=0.5,
        description="Maximum height of a SubAssembly as a ratio of the page height.",
    )

    # Score weights
    box_shape_weight: Weight = Field(
        default=0.3, description="Weight for box shape score (rectangular quality)."
    )

    count_weight: Weight = Field(
        default=0.4, description="Weight for count label presence and quality."
    )

    diagram_weight: Weight = Field(
        default=0.3, description="Weight for diagram/image presence inside box."
    )

    bbox_group_tolerance: float = Field(
        default=5.0,
        description=(
            "Maximum coordinate difference when grouping similar Drawing "
            "blocks into a single subassembly candidate. Drawings with "
            "bboxes that differ by at most this amount are treated as "
            "representing the same logical box (e.g., fill and stroke layers)."
        ),
    )

    @property
    def min_subassembly_width(self) -> float:
        """Minimum width for a SubAssembly box."""
        return self.min_part_width * self.min_size_part_multiplier

    @property
    def min_subassembly_height(self) -> float:
        """Minimum height for a SubAssembly box."""
        return self.min_part_height * self.min_size_part_multiplier
