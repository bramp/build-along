"""Rules for scoring visual properties."""

from __future__ import annotations

from build_a_long.pdf_extract.classifier.rules.base import Rule, RuleContext
from build_a_long.pdf_extract.extractor.page_blocks import Block, Drawing


class StrokeColorScore(Rule):
    """Rule that scores a drawing block based on its stroke color.

    Dividers are typically white lines on LEGO instruction pages.
    """

    def __init__(
        self,
        weight: float = 1.0,
        name: str = "StrokeColor",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not isinstance(block, Drawing):
            return 0.0
        drawing_block = block  # type: Drawing

        if drawing_block.stroke_color is not None:
            r, g, b = drawing_block.stroke_color
            # Check if it's white or very light (all channels > 0.9)
            if r > 0.9 and g > 0.9 and b > 0.9:
                return 1.0
            # Light gray is also acceptable
            if r > 0.7 and g > 0.7 and b > 0.7:
                return 0.7
            # Any other stroke color gets a lower score
            return 0.3

        if drawing_block.fill_color is not None:
            r, g, b = drawing_block.fill_color
            if r > 0.9 and g > 0.9 and b > 0.9:
                return 0.8
            if r > 0.7 and g > 0.7 and b > 0.7:
                return 0.5

        return 0.0


class CurveCountRule(Rule):
    """Rule that scores based on the number of bezier curves in a drawing.

    Useful for detecting circles (which typically consist of 4 bezier curves).
    """

    def __init__(
        self,
        min_count: int,
        weight: float = 1.0,
        name: str = "CurveCount",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required
        self.min_count = min_count

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not isinstance(block, Drawing):
            return 0.0

        if not block.items:
            return 0.0

        # 'c' indicates a curve operation in the drawing path
        curve_count = sum(1 for item in block.items if item[0] == "c")

        return 1.0 if curve_count >= self.min_count else 0.0
