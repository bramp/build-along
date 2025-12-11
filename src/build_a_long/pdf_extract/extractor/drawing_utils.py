"""Utilities for processing PyMuPDF drawing data.

This module contains functions for:
- Converting PyMuPDF drawing items to JSON-serializable tuples
- Computing visible bounding boxes considering clip paths
"""

import logging

from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.pymupdf_types import DrawingDict

logger = logging.getLogger(__name__)


def convert_drawing_items(items: list[tuple] | None) -> tuple[tuple, ...] | None:
    """Convert PyMuPDF objects in drawing items to JSON-serializable tuples.

    Drawing items can contain Rect, Point, and Quad objects which need
    to be converted to tuples for JSON serialization.

    Args:
        items: List of drawing command tuples from PyMuPDF

    Returns:
        Tuple of tuples with PyMuPDF objects converted to tuples, or None
    """
    if items is None:
        return None

    converted_items: list[tuple] = []
    for item in items:
        if not isinstance(item, tuple) or len(item) == 0:
            converted_items.append(item)
            continue

        # Convert tuple elements that are PyMuPDF objects
        converted_elements: list = [item[0]]  # Keep the command string
        for element in item[1:]:
            # Check if element has x0, y0, x1, y1 (Rect-like)
            if hasattr(element, "x0"):
                converted_elements.append(
                    (element.x0, element.y0, element.x1, element.y1)
                )
            # Check if element has x, y (Point-like)
            elif hasattr(element, "x") and hasattr(element, "y"):
                converted_elements.append((element.x, element.y))
            # Check if it's a Quad (has ul, ur, ll, lr)
            elif hasattr(element, "ul"):
                converted_elements.append(
                    (
                        (element.ul.x, element.ul.y),
                        (element.ur.x, element.ur.y),
                        (element.ll.x, element.ll.y),
                        (element.lr.x, element.lr.y),
                    )
                )
            else:
                # Keep as-is if not a recognized PyMuPDF object
                converted_elements.append(element)

        converted_items.append(tuple(converted_elements))

    return tuple(converted_items)


def compute_visible_bbox(
    bbox: BBox,
    level: int,
    drawings: list[DrawingDict],
    current_index: int,
) -> BBox:
    """Compute visible bbox by intersecting with applicable clip paths.

    In PyMuPDF's drawing hierarchy:
    - A clip at level L applies to all subsequent drawings at level > L
    - A clip's scope ends when we see a non-clip at level <= L
    - We walk backwards to find all clips that apply to the current drawing

    Args:
        bbox: The original bounding box
        level: The hierarchy level of this drawing
        drawings: Full list of drawings (with extended=True)
        current_index: Index of current drawing in the list

    Returns:
        The visible bbox after applying all relevant clips
    """
    visible = bbox

    # Build the clip stack by walking backwards
    # A clip at level L applies to drawings at level > L
    # A clip's scope ends when we see a non-clip at level <= L
    clip_stack: dict[int, BBox] = {}  # level -> clip bbox

    for i in range(current_index - 1, -1, -1):
        prev = drawings[i]
        prev_level = prev.get("level", 0)
        prev_type = prev.get("type")

        # If we see a non-clip at or below our level, we can stop
        # (we've exited all relevant clip scopes)
        if prev_type != "clip" and prev_level < level:
            logger.debug("  Stop at drawing %d (L%d non-clip)", i, prev_level)
            break

        # If it's a clip at a level less than ours, it applies
        # Only add if we haven't seen a clip at this level yet
        # (we're walking backwards, so first encountered is most recent)
        if prev_type == "clip" and prev_level < level and prev_level not in clip_stack:
            scissor = prev.get("scissor")
            if scissor:
                # Skip inverted/invalid clip rectangles - they don't make
                # geometric sense as clipping regions and are likely PDF
                # artifacts
                if scissor.x0 > scissor.x1 or scissor.y0 > scissor.y1:
                    logger.debug(
                        "  Skipping inverted clip from drawing %d (L%d): %s",
                        i,
                        prev_level,
                        scissor,
                    )
                    continue

                clip_bbox = BBox.from_tuple(
                    (scissor.x0, scissor.y0, scissor.x1, scissor.y1)
                )
                clip_stack[prev_level] = clip_bbox
                logger.debug(
                    "  Adding clip from drawing %d (L%d): %s",
                    i,
                    prev_level,
                    clip_bbox,
                )

    # Apply all clips by intersecting
    for clip_bbox in clip_stack.values():
        visible = visible.intersect(clip_bbox)

    return visible
