"""
Rules for rule-based classification.

This module defines the Rule interface and common rules used to score
blocks for classification.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from re import Pattern

from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.classifier.text import (
    extract_bag_number_value,
    extract_element_id,
    extract_page_number_value,
    extract_part_count_value,
    extract_step_number_value,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.page_blocks import Block, Text, Drawing


@dataclass
class RuleContext:
    """Context passed to rules during evaluation."""

    page_data: PageData
    config: ClassifierConfig


class Rule(ABC):
    """Abstract base class for scoring rules.

    Attributes:
        name: A descriptive name for the rule (used in debug logs/score details).
        weight: The default weight (0.0 or greater) contribution of this rule to the
            total score.
        required: If True, a calculated score of 0.0 will immediately disqualify
            the candidate, regardless of how high other rule scores are.
            Useful for hard filters (e.g. "Must be Text").
    """

    name: str
    weight: float = 1.0

    # If true, a score of 0.0 from this rule will immediately disqualify the candidate
    # regardless of other scores.
    required: bool = False

    @abstractmethod
    def calculate(self, block: Block, context: RuleContext) -> float | None:
        """Calculate a score between 0.0 and 1.0 for the given block.

        Returns:
            float: Score between 0.0 and 1.0
            None: If the rule cannot be evaluated (e.g. missing hints) and should
                  be ignored (not contributing to the weighted sum).
        """
        pass


class Filter(Rule):
    """Base class for filters (rules that don't contribute to score)."""

    weight: float = 0.0
    required: bool = True


class IsInstanceFilter(Filter):
    """Filter that checks if a block is of a specific type."""

    def __init__(self, block_type: type[Block], name: str = ""):
        self.name = name if name else f"IsInstance({block_type.__name__})"
        self.block_type = block_type

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        return 1.0 if isinstance(block, self.block_type) else 0.0


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


class InBottomBandFilter(Filter):
    """Filter that checks if a block is in the bottom band of the page."""

    def __init__(
        self,
        threshold_ratio: float = 0.1,
        name: str = "InBottomBand",
        invert: bool = False,
    ):
        self.name = name
        self.threshold_ratio = threshold_ratio
        self.invert = invert

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        page_bbox = context.page_data.bbox
        assert page_bbox is not None

        bottom_threshold = page_bbox.y1 - (page_bbox.height * self.threshold_ratio)
        element_center_y = (block.bbox.y0 + block.bbox.y1) / 2

        is_in_band = element_center_y >= bottom_threshold

        if self.invert:
            return 0.0 if is_in_band else 1.0
        return 1.0 if is_in_band else 0.0


class CornerDistanceScore(Rule):
    """Rule that scores based on distance to bottom corners."""

    def __init__(
        self,
        scale: float,
        weight: float = 1.0,
        name: str = "CornerDistanceScore",
        required: bool = False,
    ):
        self.name = name
        self.scale = scale
        self.weight = weight
        self.required = required

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        page_bbox = context.page_data.bbox
        assert page_bbox is not None

        element_center_x = (block.bbox.x0 + block.bbox.x1) / 2
        element_center_y = (block.bbox.y0 + block.bbox.y1) / 2

        dist_bottom_left = math.sqrt(
            (element_center_x - page_bbox.x0) ** 2
            + (element_center_y - page_bbox.y1) ** 2
        )
        dist_bottom_right = math.sqrt(
            (element_center_x - page_bbox.x1) ** 2
            + (element_center_y - page_bbox.y1) ** 2
        )
        min_dist = min(dist_bottom_left, dist_bottom_right)

        return math.exp(-min_dist / self.scale)


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


class MaxScoreRule(Rule):
    """Rule that returns the maximum score from a list of sub-rules.

    Useful when there are multiple ways to satisfy a criteria (e.g. matching
    one of several font size hints).
    """

    def __init__(
        self,
        rules: list[Rule],
        weight: float = 1.0,
        name: str = "MaxScore",
        required: bool = False,
    ):
        self.name = name
        self.rules = rules
        self.weight = weight
        self.required = required

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        scores = []
        for rule in self.rules:
            score = rule.calculate(block, context)
            if score is not None:
                scores.append(score)

        if not scores:
            return None

        return max(scores)


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


class TopLeftPositionScore(Rule):
    """Rule that scores based on position in top-left area (common for Bag numbers)."""

    def __init__(
        self,
        weight: float = 1.0,
        name: str = "TopLeftPositionScore",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        page_bbox = context.page_data.bbox
        assert page_bbox is not None

        # Check vertical position (should be in top 40% of page)
        center_y = (block.bbox.y0 + block.bbox.y1) / 2
        vertical_ratio = (center_y - page_bbox.y0) / page_bbox.height

        if vertical_ratio > 0.4:
            # Too far down the page
            return 0.0

        # Score higher for positions closer to the top
        vertical_score = 1.0 - (vertical_ratio / 0.4)

        # Check horizontal position (prefer left half)
        center_x = (block.bbox.x0 + block.bbox.x1) / 2
        horizontal_ratio = (center_x - page_bbox.x0) / page_bbox.width

        # Favor left side (1.0), but don't completely exclude right side (0.3)
        horizontal_score = 1.0 if horizontal_ratio <= 0.5 else 0.3

        # Combine scores (70% vertical, 30% horizontal)
        return 0.7 * vertical_score + 0.3 * horizontal_score


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


class SizeRangeRule(Rule):
    """Rule that checks if block dimensions are within specified ranges."""

    def __init__(
        self,
        min_width: float | None = None,
        max_width: float | None = None,
        min_height: float | None = None,
        max_height: float | None = None,
        weight: float = 1.0,
        name: str = "SizeRange",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        width = block.bbox.width
        height = block.bbox.height

        if self.min_width is not None and width < self.min_width:
            return 0.0
        if self.max_width is not None and width > self.max_width:
            return 0.0
        if self.min_height is not None and height < self.min_height:
            return 0.0
        if self.max_height is not None and height > self.max_height:
            return 0.0

        return 1.0


class AspectRatioRule(Rule):
    """Rule that checks if block aspect ratio (width/height) is within range."""

    def __init__(
        self,
        min_ratio: float,
        max_ratio: float,
        weight: float = 1.0,
        name: str = "AspectRatio",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if block.bbox.height <= 0:
            return 0.0

        ratio = block.bbox.width / block.bbox.height
        if self.min_ratio <= ratio <= self.max_ratio:
            return 1.0
        return 0.0


class CoverageRule(Rule):
    """Rule that scores based on page area coverage."""

    def __init__(
        self,
        min_ratio: float = 0.85,
        weight: float = 1.0,
        name: str = "Coverage",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required
        self.min_ratio = min_ratio

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        page_bbox = context.page_data.bbox
        assert page_bbox is not None
        page_area = page_bbox.area
        if page_area <= 0:
            return 0.0

        block_area = block.bbox.area
        coverage_ratio = block_area / page_area

        if coverage_ratio < self.min_ratio:
            return 0.0

        # Score increases from min_ratio to 1.0
        # Map [min_ratio, 1.0] -> [0.5, 1.0]
        normalized = (coverage_ratio - self.min_ratio) / (1.0 - self.min_ratio)
        return min(1.0, normalized * 0.5 + 0.5)


class EdgeProximityRule(Rule):
    """Rule that scores based on proximity to page edges."""

    def __init__(
        self,
        threshold: float = 10.0,
        weight: float = 1.0,
        name: str = "EdgeProximity",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required
        self.threshold = threshold

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        page_bbox = context.page_data.bbox
        assert page_bbox is not None

        # Calculate distance from each edge
        left_dist = abs(block.bbox.x0 - page_bbox.x0)
        right_dist = abs(block.bbox.x1 - page_bbox.x1)
        top_dist = abs(block.bbox.y0 - page_bbox.y0)
        bottom_dist = abs(block.bbox.y1 - page_bbox.y1)

        avg_edge_dist = (left_dist + right_dist + top_dist + bottom_dist) / 4.0

        if avg_edge_dist <= self.threshold:
            return 1.0

        # Decrease score as distance increases
        # Decay over 50 units
        return max(0.0, 1.0 - (avg_edge_dist - self.threshold) / 50.0)


class IsVerticalDividerRule(Rule):
    """Rule that checks if a block is a vertical divider."""

    def __init__(
        self,
        max_thickness: float,
        min_length_ratio: float,
        edge_margin: float,
        weight: float = 1.0,
        name: str = "IsVerticalDivider",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required
        self.max_thickness = max_thickness
        self.min_length_ratio = min_length_ratio
        self.edge_margin = edge_margin

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not isinstance(block, Drawing):
            return 0.0
        drawing_block = block  # type: Drawing

        page_bbox = context.page_data.bbox
        assert page_bbox is not None

        bbox = drawing_block.bbox
        width = bbox.width
        height = bbox.height
        page_height = page_bbox.height

        # Check for vertical divider (thin width, tall height)
        if (
            width <= self.max_thickness
            and height >= page_height * self.min_length_ratio
        ):
            # Reject vertical dividers at left or right page edges
            at_left_edge = bbox.x0 <= page_bbox.x0 + self.edge_margin
            at_right_edge = bbox.x1 >= page_bbox.x1 - self.edge_margin
            if at_left_edge or at_right_edge:
                return 0.0
            return 1.0
        return 0.0


class IsHorizontalDividerRule(Rule):
    """Rule that checks if a block is a horizontal divider."""

    def __init__(
        self,
        max_thickness: float,
        min_length_ratio: float,
        edge_margin: float,
        weight: float = 1.0,
        name: str = "IsHorizontalDivider",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required
        self.max_thickness = max_thickness
        self.min_length_ratio = min_length_ratio
        self.edge_margin = edge_margin

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not isinstance(block, Drawing):
            return 0.0
        drawing_block = block  # type: Drawing

        page_bbox = context.page_data.bbox
        assert page_bbox is not None

        bbox = drawing_block.bbox
        width = bbox.width
        height = bbox.height
        page_width = page_bbox.width

        # Check for horizontal divider (thin height, wide width)
        if height <= self.max_thickness and width >= page_width * self.min_length_ratio:
            # Reject horizontal dividers at top or bottom page edges
            at_top_edge = bbox.y0 <= page_bbox.y0 + self.edge_margin
            at_bottom_edge = bbox.y1 >= page_bbox.y1 - self.edge_margin
            if at_top_edge or at_bottom_edge:
                return 0.0
            return 1.0
        return 0.0


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

        # No stroke color - could still be a divider via fill
        if drawing_block.fill_color is not None:
            r, g, b = drawing_block.fill_color
            if r > 0.9 and g > 0.9 and b > 0.9:
                return 0.8
            if r > 0.7 and g > 0.7 and b > 0.7:
                return 0.5

        return 0.0
