"""Rules for scoring geometry and position."""

from __future__ import annotations

import math

from build_a_long.pdf_extract.classifier.rules.base import Filter, Rule, RuleContext
from build_a_long.pdf_extract.classifier.rules.scoring import (
    score_linear,
    score_triangular,
)
from build_a_long.pdf_extract.extractor.page_blocks import Block, Drawing, Text


class InBottomBandFilter(Rule):
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
        self.weight = 0.0
        self.required = True

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


class SizePreferenceScore(Rule):
    """Rule that scores based on size with a target preference.

    Checks that both width and height are within [min_size, max_size], then
    scores based on average size relative to target:
    - At target_size: score = 1.0
    - At min_size: score = min_score (default 0.75)
    - Outside min/max range (either dimension): score = 0.0

    Linear interpolation between min and target.
    """

    def __init__(
        self,
        min_size: float,
        target_size: float,
        max_size: float,
        min_score: float = 0.75,
        weight: float = 1.0,
        name: str = "SizePreference",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required
        self.min_size = min_size
        self.target_size = target_size
        self.max_size = max_size
        self.min_score = min_score

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        width = block.bbox.width
        height = block.bbox.height

        # Both dimensions must be within range
        if width < self.min_size or width > self.max_size:
            return 0.0
        if height < self.min_size or height > self.max_size:
            return 0.0

        # Score based on average size relative to target
        avg_size = (width + height) / 2

        # Interpolate: min_score at min_size, 1.0 at target_size
        if avg_size <= self.target_size:
            if self.target_size == self.min_size:
                return 1.0
            t = (avg_size - self.min_size) / (self.target_size - self.min_size)
            return self.min_score + t * (1.0 - self.min_score)
        else:
            # Beyond target but still within max - score decreases
            if self.max_size == self.target_size:
                return 1.0
            t = (avg_size - self.target_size) / (self.max_size - self.target_size)
            return 1.0 - t * (1.0 - self.min_score)


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
        return score_linear(
            coverage_ratio,
            min_val=self.min_ratio,
            max_val=1.0,
            min_score=0.5,
            max_score=1.0,
        )


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


class PageEdgeFilter(Filter):
    """Filter that passes if a block is entirely within the edge margin of the page.

    Used to capture page-edge artifacts (borders, bleed lines) that should be
    classified as background. Unlike EdgeProximityRule which checks if a block
    *touches* page edges, this filter checks if a block is entirely *contained*
    within the margin zone at any page edge.
    """

    def __init__(
        self,
        margin: float = 4.0,
        name: str = "PageEdge",
    ):
        self.name = name
        self.margin = margin

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        page_bbox = context.page_data.bbox
        assert page_bbox is not None
        bbox = block.bbox

        # Check if entirely at left edge (x1 <= margin from left)
        if bbox.x1 <= page_bbox.x0 + self.margin:
            return 1.0

        # Check if entirely at right edge (x0 >= page_width - margin)
        if bbox.x0 >= page_bbox.x1 - self.margin:
            return 1.0

        # Check if entirely at top edge (y1 <= margin from top)
        if bbox.y1 <= page_bbox.y0 + self.margin:
            return 1.0

        # Check if entirely at bottom edge (y0 >= page_height - margin)
        if bbox.y0 >= page_bbox.y1 - self.margin:
            return 1.0

        return 0.0


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


class TextContainerFitRule(Rule):
    """Rule that scores how well a text block fits inside a containing drawing.

    Finds the smallest drawing containing the text and scores based on the
    ratio of drawing area to text area. Ideal for piece length indicators
    (numbers inside circles).
    """

    def __init__(
        self,
        weight: float = 1.0,
        name: str = "TextContainerFit",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not isinstance(block, Text):
            return 0.0

        text_area = block.bbox.area
        if text_area <= 0:
            return 0.0

        # Find all drawings
        drawings = [b for b in context.page_data.blocks if isinstance(b, Drawing)]

        best_score = 0.0
        # Maximum ratio of drawing area to text area to consider
        MAX_AREA_RATIO = 6.0

        for drawing in drawings:
            # Check if text bbox is fully contained in drawing bbox
            if drawing.bbox.contains(block.bbox):
                drawing_area = drawing.bbox.area

                # Skip drawings that are way too large
                ratio = drawing_area / text_area
                if ratio > MAX_AREA_RATIO:
                    continue

                # Score based on ratio
                # Ideal ratio: 2-4x (circle slightly larger than text)
                score = 0.0
                if 2.0 <= ratio <= 4.0:
                    score = 1.0
                elif 1.0 <= ratio < 2.0:
                    score = 0.8
                elif 4.0 < ratio <= 10.0:
                    # Should be covered by MAX_AREA_RATIO check above,
                    # but for completeness
                    score = 0.6
                else:
                    score = 0.1

                if score > best_score:
                    best_score = score

        return best_score


class SizeRatioRule(Rule):
    """Rule that scores based on block size relative to page dimensions.

    Calculates width and height ratios relative to page size and scores them
    using a triangular function peaking at `ideal_ratio` and decaying to 0.0
    at `min_ratio` and `max_ratio`.
    """

    def __init__(
        self,
        ideal_ratio: float,
        min_ratio: float,
        max_ratio: float,
        weight: float = 1.0,
        name: str = "SizeRatio",
        required: bool = False,
    ):
        self.name = name
        self.ideal_ratio = ideal_ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.weight = weight
        self.required = required

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        page_bbox = context.page_data.bbox
        assert page_bbox is not None
        if page_bbox.width <= 0 or page_bbox.height <= 0:
            return 0.0

        width_ratio = block.bbox.width / page_bbox.width
        height_ratio = block.bbox.height / page_bbox.height

        w_score = score_triangular(
            width_ratio, self.min_ratio, self.ideal_ratio, self.max_ratio
        )
        h_score = score_triangular(
            height_ratio, self.min_ratio, self.ideal_ratio, self.max_ratio
        )
        return (w_score + h_score) / 2.0


# TODO I think we can merge this with IsBottomBandFilter
class BottomPositionScore(Rule):
    """Rule that scores based on position at bottom of page.

    Scores higher for elements closer to the bottom edge.
    """

    def __init__(
        self,
        max_bottom_margin_ratio: float,
        weight: float = 1.0,
        name: str = "BottomPositionScore",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required
        self.max_bottom_margin_ratio = max_bottom_margin_ratio

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        page_bbox = context.page_data.bbox
        assert page_bbox is not None
        page_height = page_bbox.height

        element_bottom = block.bbox.y1
        bottom_distance = page_bbox.y1 - element_bottom
        bottom_margin_ratio = bottom_distance / page_height

        # Should be in bottom margin area
        if bottom_margin_ratio > self.max_bottom_margin_ratio:
            return 0.0

        # Score based on proximity to bottom (closer = better)
        return 1.0 - (bottom_margin_ratio / self.max_bottom_margin_ratio)


class PageNumberProximityScore(Rule):
    """Rule that scores based on proximity to the page number.

    Boosts score if the element is horizontally near the detected page number.
    """

    def __init__(
        self,
        proximity_ratio: float,
        weight: float = 1.0,
        name: str = "PageNumberProximityScore",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required
        self.proximity_ratio = proximity_ratio

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if not context.classification_result:
            return None

        page_bbox = context.page_data.bbox
        assert page_bbox is not None

        page_number_candidates = context.classification_result.get_scored_candidates(
            "page_number"
        )
        if not page_number_candidates:
            return None

        # Assume best candidate is page number
        pn_bbox = page_number_candidates[0].bbox
        horizontal_distance = min(
            abs(block.bbox.x0 - pn_bbox.x1),
            abs(block.bbox.x1 - pn_bbox.x0),
        )

        if horizontal_distance < page_bbox.width * self.proximity_ratio:
            return 1.0

        return 0.0


class WidthCoverageScore(Rule):
    """Rule that scores based on how much of the page width the element spans."""

    def __init__(
        self,
        min_width_ratio: float,
        max_score_width_ratio: float,
        weight: float = 1.0,
        name: str = "WidthCoverageScore",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required
        self.min_width_ratio = min_width_ratio
        self.max_score_width_ratio = max_score_width_ratio

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        page_bbox = context.page_data.bbox
        assert page_bbox is not None
        if page_bbox.width <= 0:
            return 0.0

        width_ratio = block.bbox.width / page_bbox.width

        # Penalize elements that are too narrow
        if width_ratio < self.min_width_ratio:
            return 0.0

        # Score increases with width, maxing at max_score_width_ratio
        if width_ratio >= self.max_score_width_ratio:
            return 1.0

        # Linear interpolation
        return score_linear(
            width_ratio,
            min_val=self.min_width_ratio,
            max_val=self.max_score_width_ratio,
            min_score=0.0,
            max_score=1.0,
        )


class ContinuousAspectRatioScore(Rule):
    """Rule that scores aspect ratio using linear interpolation.

    Scores 0.0 below min_ratio, 1.0 above ideal_ratio, and linear in between.
    """

    def __init__(
        self,
        min_ratio: float,
        ideal_ratio: float,
        weight: float = 1.0,
        name: str = "ContinuousAspectRatioScore",
        required: bool = False,
    ):
        self.name = name
        self.weight = weight
        self.required = required
        self.min_ratio = min_ratio
        self.ideal_ratio = ideal_ratio

    def calculate(self, block: Block, context: RuleContext) -> float | None:
        if block.bbox.height <= 0:
            return 0.0

        aspect_ratio = block.bbox.width / block.bbox.height

        if aspect_ratio < self.min_ratio:
            return 0.0

        if aspect_ratio >= self.ideal_ratio:
            return 1.0

        return score_linear(
            aspect_ratio,
            min_val=self.min_ratio,
            max_val=self.ideal_ratio,
            min_score=0.0,
            max_score=1.0,
        )
