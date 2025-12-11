"""Rules for scoring text blocks."""

from __future__ import annotations

import re
from re import Pattern

from build_a_long.pdf_extract.classifier.rules.base import Rule, RuleContext
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
    """Rule that scores based on font size match with a hint."""

    def __init__(
        self,
        target_size: float | None,
        weight: float = 1.0,
        name: str = "FontSizeMatch",
        required: bool = False,
    ):
        self.name = name
        self.target_size = target_size
        self.weight = weight
        self.required = required

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not isinstance(block, Text):
            return 0.0

        # If no hint available, skip this rule
        if self.target_size is None:
            return None

        if block.font_size is None:
            size = block.bbox.height
        else:
            size = block.font_size

        if self.target_size == 0:
            return 0.0

        diff_ratio = abs(size - self.target_size) / self.target_size
        # Linear penalty: score = 1.0 - (diff_ratio * 2.0)
        return max(0.0, 1.0 - (diff_ratio * 2.0))


class FontSizeRangeRule(Rule):
    """Rule that scores based on font size being within a range.

    Scores 1.0 if strictly between min and max (with tolerance).
    Scores 0.7 if within tolerance of min (indicating close to smaller type).
    Scores 0.0 if outside range.
    """

    def __init__(
        self,
        min_size: float | None,
        max_size: float | None,
        tolerance: float = 1.0,
        weight: float = 1.0,
        name: str = "FontSizeRange",
        required: bool = False,
    ):
        self.name = name
        self.min_size = min_size
        self.max_size = max_size
        self.tolerance = tolerance
        self.weight = weight
        self.required = required

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not isinstance(block, Text) or block.font_size is None:
            return 0.0

        if self.min_size is None or self.max_size is None:
            return 0.5  # Neutral if hints missing

        font_size = block.font_size

        if font_size < self.min_size - self.tolerance:
            return 0.0
        if font_size > self.max_size + self.tolerance:
            return 0.0

        if font_size > self.min_size + self.tolerance:
            return 1.0

        return 0.7


class PageNumberValueMatch(Rule):
    """Rule that scores how well text matches the expected page number."""

    def __init__(
        self,
        weight: float = 1.0,
        name: str = "PageNumberValueMatch",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not isinstance(block, Text):
            return 0.0

        value = extract_page_number_value(block.text)
        if value is None:
            return 0.0

        expected = context.page_data.page_number
        diff = abs(value - expected)
        return max(0.0, 1.0 - 0.1 * diff)


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
        weight: float = 1.0,
        name: str = "BagNumberFontSizeScore",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required

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
            return max(0.5, 1.0 - (font_size - 60) / 60.0)


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
