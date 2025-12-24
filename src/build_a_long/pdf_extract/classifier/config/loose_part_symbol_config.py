"""Configuration for loose part symbol classification."""

from __future__ import annotations

from pydantic import BaseModel, Field

from build_a_long.pdf_extract.classifier.score import Weight


class LoosePartSymbolConfig(BaseModel):
    """Configuration for loose part symbol classification.

    LoosePartSymbol elements are small, square-ish symbols that appear next to
    OpenBag circles containing a Part instead of a bag number. The symbol
    indicates that an extra part is needed that's not found in the main bag.
    """

    min_score: Weight = Field(
        default=0.4,
        description="Minimum score threshold for loose part symbol candidates.",
    )

    ideal_size: float = Field(
        default=68.5,
        description=(
            "Ideal average size (width + height / 2) in points for the symbol cluster. "
            "Based on observed symbols which are approximately 68.5 points square."
        ),
    )

    size_tolerance: float = Field(
        default=0.10,
        description=(
            "Tolerance for size deviation as a fraction of ideal_size. "
            "A value of 0.10 means ±10% of ideal_size is acceptable."
        ),
    )
