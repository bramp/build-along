"""Utility functions for classifiers."""

from collections.abc import Sequence

from build_a_long.pdf_extract.extractor.page_blocks import Drawing


def score_white_fill(block: Drawing, white_threshold: float = 0.9) -> float:
    """Score a drawing block based on having white fill.

    Args:
        block: The Drawing block to analyze.
        white_threshold: Threshold for considering a channel "white" (0.0-1.0).

    Returns:
        Score from 0.0 to 1.0 where 1.0 is white fill.
    """
    if block.fill_color is None:
        return 0.0

    r, g, b = block.fill_color
    # Check if it's white (all channels above threshold)
    if r >= white_threshold and g >= white_threshold and b >= white_threshold:
        return 1.0

    # Light gray is also acceptable
    # Using 0.7 as a reasonable baseline for "light gray" based on existing classifiers
    gray_threshold = 0.7
    if r > gray_threshold and g > gray_threshold and b > gray_threshold:
        # Scale score based on how close to white it is
        # 0.7 -> 0.6, 0.9 -> 1.0
        min_score = 0.6
        return min_score + (1.0 - min_score) * (min(r, g, b) - gray_threshold) / (
            1.0 - gray_threshold
        )

    return 0.0


def colors_match(
    color1: Sequence[float],
    color2: Sequence[float],
    tolerance: float = 0.1,
) -> bool:
    """Check if two colors match within a tolerance.

    Args:
        color1: First color as RGB tuple (0.0-1.0).
        color2: Second color as RGB tuple (0.0-1.0).
        tolerance: Maximum difference per channel.

    Returns:
        True if colors match within tolerance.
    """
    if len(color1) != len(color2):
        return False
    return all(abs(c1 - c2) <= tolerance for c1, c2 in zip(color1, color2, strict=True))


def extract_unique_points(
    line_items: Sequence[tuple], precision: int = 1
) -> list[tuple[float, float]]:
    """Extract unique points from line items.

    Args:
        line_items: List of line items, each ('l', (x1, y1), (x2, y2)).
        precision: Decimal places for rounding to determine uniqueness.

    Returns:
        List of unique (x, y) points.
    """
    points: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()

    for item in line_items:
        # item is ('l', (x1, y1), (x2, y2))
        # Ensure it is a line item
        if item[0] != "l":
            continue

        p1, p2 = item[1], item[2]
        for p in [p1, p2]:
            key = (round(p[0], precision), round(p[1], precision))
            if key not in seen:
                seen.add(key)
                points.append((p[0], p[1]))

    return points
