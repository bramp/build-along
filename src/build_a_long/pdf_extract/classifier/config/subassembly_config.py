"""Configuration for the subassembly classifier."""

from pydantic import BaseModel

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

    min_score: Weight = 0.5
    """Minimum score threshold for subassembly candidates."""

    # Minimum size configuration
    # TODO The min_part sizes should be moved to a common config used by both
    min_part_width: float = 20.0
    """Minimum width of a Part element in points."""

    min_part_height: float = 15.0
    """Minimum height of a Part element in points."""

    min_size_multiplier: float = 1.5
    """SubAssembly must be at least this many times larger than the minimum Part size."""

    # Score weights
    box_shape_weight: Weight = 0.3
    """Weight for box shape score (rectangular quality)."""

    count_weight: Weight = 0.4
    """Weight for count label presence and quality."""

    diagram_weight: Weight = 0.3
    """Weight for diagram/image presence inside box."""

    @property
    def min_subassembly_width(self) -> float:
        """Minimum width for a SubAssembly box."""
        return self.min_part_width * self.min_size_multiplier

    @property
    def min_subassembly_height(self) -> float:
        """Minimum height for a SubAssembly box."""
        return self.min_part_height * self.min_size_multiplier
