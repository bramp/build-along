"""Rules for scoring text blocks."""

from __future__ import annotations

import re
from re import Pattern

from build_a_long.pdf_extract.classifier.rules.base import Rule, RuleContext
from build_a_long.pdf_extract.classifier.rules.scale import ScaleFunction
from build_a_long.pdf_extract.classifier.text import (
    extract_bag_number_value,
    extract_element_id,
    extract_page_number_value,
    extract_part_count_value,
    extract_step_number_value,
)
from build_a_long.pdf_extract.extractor.page_blocks import Block, Text


class RegexMatch(Rule):
    """Rule that checks if text matches a regex pattern."""

    def __init__(
        self,
        regex: str | Pattern,
        weight: float = 1.0,
        name: str = "RegexMatch",
        required: bool = False,
    ):
        self.name = name
        self.regex = re.compile(regex) if isinstance(regex, str) else regex
        self.weight = weight
        self.required = required

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not isinstance(block, Text):
            return 0.0
        return 1.0 if self.regex.match(block.text.strip()) else 0.0


class FontSizeMatch(Rule):
    """Rule that scores based on font size using a scale function.

    The scale should be configured with absolute font size thresholds.
    """

    def __init__(
        self,
        scale: ScaleFunction,
        weight: float = 1.0,
        name: str = "FontSizeMatch",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required
        self.scale = scale

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not isinstance(block, Text):
            return 0.0

        size = block.bbox.height if block.font_size is None else block.font_size

        # Pass the actual font size to the scale
        return self.scale(size)


class FontSizeRangeRule(Rule):
    """Rule that scores based on font size using the provided scale."""

    def __init__(
        self,
        scale: ScaleFunction,
        weight: float = 1.0,
        name: str = "FontSizeRange",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required
        self.scale = scale

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not isinstance(block, Text) or block.font_size is None:
            return 0.0

        return self.scale(block.font_size)


class PageNumberValueMatch(Rule):
    """Rule that scores how well text matches the expected page number."""

    def __init__(
        self,
        scale: ScaleFunction,
        weight: float = 1.0,
        name: str = "PageNumberValueMatch",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required
        self.scale = scale

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not isinstance(block, Text):
            return 0.0

        value = extract_page_number_value(block.text)
        if value is None:
            return 0.0

        expected = context.page_data.page_number
        diff = abs(value - expected)
        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, self.scale(diff)))


class PageNumberTextRule(Rule):
    """Rule that checks if text looks like a page number (1-3 digits)."""

    def __init__(
        self,
        weight: float = 1.0,
        name: str = "PageNumberTextMatch",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not isinstance(block, Text):
            return 0.0
        text = block.text.strip()
        # Zero prefixed numbers are allowed but score slightly lower
        if re.match(r"^0+\d{1,3}$", text):
            return 0.90
        # 1-3 digit numbers are just fine
        if re.match(r"^\d{1,3}$", text):
            return 1.0
        return 0.0


class PartCountTextRule(Rule):
    """Rule that checks if text looks like a part count (e.g. '2x')."""

    def __init__(
        self,
        weight: float = 1.0,
        name: str = "PartCountTextMatch",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not isinstance(block, Text):
            return 0.0
        if extract_part_count_value(block.text) is not None:
            return 1.0
        return 0.0


class PartNumberTextRule(Rule):
    """Rule that checks if text matches a part number (element ID) pattern.

    Includes scoring based on length distribution (6-7 digits preferred).
    """

    # Empirical distribution of element ID lengths
    LENGTH_DISTRIBUTION = {
        4: 0.0002,
        5: 0.0050,
        6: 0.0498,
        7: 0.9447,
        8: 0.0003,
    }

    def __init__(
        self,
        weight: float = 1.0,
        name: str = "PartNumberTextMatch",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not isinstance(block, Text):
            return 0.0

        element_id = extract_element_id(block.text)
        if element_id is None:
            return 0.0

        num_digits = len(element_id)
        # Base score (0.5) + distribution bonus (scaled to 0.0-0.5 range)
        distribution_score = self.LENGTH_DISTRIBUTION.get(num_digits, 0.0)
        return 0.5 + (0.5 * distribution_score)


class BagNumberTextRule(Rule):
    """Rule that checks if text matches a bag number pattern."""

    def __init__(
        self,
        weight: float = 1.0,
        name: str = "BagNumberTextMatch",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not isinstance(block, Text):
            return 0.0
        if extract_bag_number_value(block.text) is not None:
            return 1.0
        return 0.0


class BagNumberFontSizeRule(Rule):
    """Rule that scores bag numbers based on absolute font size ranges."""

    def __init__(
        self,
        scale: ScaleFunction,
        weight: float = 1.0,
        name: str = "BagNumberFontSizeScore",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required
        self.scale = scale

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not isinstance(block, Text):
            return 0.0

        font_size = block.font_size
        if font_size is None:
            return 0.0

        # Bag numbers must be quite large (at least 35 points)
        if font_size < 35:
            return 0.0
        elif font_size <= 60:
            return 1.0
        else:
            # Very large is okay but not preferred
            # Clamp to [0.5, 1.0]
            return max(0.5, min(1.0, self.scale(font_size)))


class StepNumberTextRule(Rule):
    """Rule that checks if text matches a step number pattern."""

    def __init__(
        self,
        weight: float = 1.0,
        name: str = "StepNumberTextMatch",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not isinstance(block, Text):
            return 0.0
        if extract_step_number_value(block.text) is not None:
            return 1.0
        return 0.0


class PieceLengthValueRule(Rule):
    """Rule that checks if text represents a valid piece length (1-32)."""

    def __init__(
        self,
        weight: float = 1.0,
        name: str = "PieceLengthValue",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not isinstance(block, Text):
            return 0.0

        try:
            value = int(block.text.strip())
            if 1 <= value <= 32:
                return 1.0
        except ValueError:
            pass
        return 0.0


class StepValueMaxFilter(Rule):
    """Filter that rejects step numbers above a maximum value.

    Returns 1.0 for values <= max_value, 0.0 for values > max_value.
    Used to distinguish substep numbers (small values like 1-10) from
    main step numbers (larger values like 100-500).
    """

    def __init__(
        self,
        max_value: int,
        weight: float = 1.0,
        name: str = "StepValueMax",
        required: bool = False,
    ):
        self.name = name
        self.max_value = max_value
        self.weight = weight
        self.required = required

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not isinstance(block, Text):
            return 0.0

        value = extract_step_number_value(block.text)
        if value is None:
            return 0.0

        if value <= self.max_value:
            return 1.0
        return 0.0


class FontSizeSmallerThanRule(Rule):
    """Rule that scores based on font size being smaller than a reference size.

    Scores 1.0 if font size is clearly smaller (below threshold_ratio).
    Scores 0.7 if font size is similar (between threshold_ratio and 1.0).
    Scores 0.4 if font size is larger (above 1.0).

    Used for substep numbers which should be smaller than main step numbers.
    """

    def __init__(
        self,
        reference_size: float | None,
        threshold_ratio: float = 0.85,
        weight: float = 1.0,
        name: str = "FontSizeSmallerThan",
        required: bool = False,
    ):
        self.name = name
        self.reference_size = reference_size
        self.threshold_ratio = threshold_ratio
        self.weight = weight
        self.required = required

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not isinstance(block, Text):
            return 0.0

        # If no reference size available, skip this rule
        if self.reference_size is None:
            return None

        font_size = block.font_size if block.font_size else block.bbox.height

        ratio = font_size / self.reference_size

        if ratio < self.threshold_ratio:
            # Good - clearly smaller than reference
            return 1.0
        elif ratio < 1.0:
            # Similar size - could go either way
            return 0.7
        else:
            # Larger than reference - less likely but still possible
            return 0.4
