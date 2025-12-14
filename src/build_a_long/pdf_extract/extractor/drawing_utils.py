"""Utilities for processing PyMuPDF drawing data.

This module contains functions for converting PyMuPDF drawing items
to JSON-serializable tuples.

For clip path computation, use the clip module.
"""

from build_a_long.pdf_extract.extractor.pymupdf_types import DrawingDict  # noqa: F401


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
