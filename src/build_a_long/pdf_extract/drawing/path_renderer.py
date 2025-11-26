"""Rendering of vector drawing paths from PyMuPDF."""

import logging

from PIL import ImageDraw

logger = logging.getLogger(__name__)


def draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[float, float, float, float],
    outline: str | None = None,
    width: int = 0,
    dash_length: int = 5,
) -> None:
    """Draw a dashed rectangle.

    Args:
        draw: PIL ImageDraw object
        bbox: Bounding box as (x0, y0, x1, y1)
        outline: Color for the dashed line
        width: Line width
        dash_length: Length of each dash segment
    """
    x0, y0, x1, y1 = bbox
    # Top edge
    for i in range(int(x0), int(x1), dash_length * 2):
        draw.line([(i, y0), (min(i + dash_length, x1), y0)], fill=outline, width=width)
    # Bottom edge
    for i in range(int(x0), int(x1), dash_length * 2):
        draw.line([(i, y1), (min(i + dash_length, x1), y1)], fill=outline, width=width)
    # Left edge
    for i in range(int(y0), int(y1), dash_length * 2):
        draw.line([(x0, i), (x0, min(i + dash_length, y1))], fill=outline, width=width)
    # Right edge
    for i in range(int(y0), int(y1), dash_length * 2):
        draw.line([(x1, i), (x1, min(i + dash_length, y1))], fill=outline, width=width)


def draw_path_items(
    draw: ImageDraw.ImageDraw,
    items: tuple[tuple, ...],
    scale_x: float,
    scale_y: float,
    color: str = "cyan",
) -> None:
    """Draw the actual vector path from drawing items.

    Note: Bezier curves are rendered as straight lines to their endpoints
    rather than true curves, for simplicity.

    Args:
        draw: PIL ImageDraw object
        items: Drawing path items (PyMuPDF get_drawings), converted to tuples
        scale_x: X scaling factor
        scale_y: Y scaling factor
        color: Color for the path lines (default: cyan)
    """
    # Convert path items to coordinates
    path_points: list[tuple[float, float]] = []

    commands_processed = 0

    for item in items:
        if not isinstance(item, tuple):
            logger.warning(
                "Drawing item is not a tuple: %s (type=%s)", item, type(item)
            )
            continue

        if len(item) == 0:
            logger.warning("Drawing item is empty tuple")
            continue

        cmd = item[0]
        if not isinstance(cmd, str):
            logger.warning("Command is not a string: %s (type=%s)", cmd, type(cmd))
            continue

        commands_processed += 1

        if cmd == "l":  # line to
            if len(item) >= 2:
                point = item[1]
                # Point could be tuple or list after JSON deserialization
                if isinstance(point, tuple | list) and len(point) == 2:
                    if not all(isinstance(v, int | float) for v in point):
                        logger.warning("Line point has non-numeric values: %s", point)
                    else:
                        path_points.append((point[0] * scale_x, point[1] * scale_y))
                else:
                    logger.warning(
                        "Line command unexpected point format: %s",
                        item,
                    )
            else:
                logger.warning("Line command with insufficient items: %s", item)
        elif cmd == "c":  # curve (bezier) - simplified to endpoint only
            if len(item) >= 2:
                # NOTE: This renders as straight line to endpoint, not true bezier
                # Last point is the end of the curve
                point = item[-1] if isinstance(item[-1], tuple | list) else item[1]
                if isinstance(point, tuple | list) and len(point) == 2:
                    if not all(isinstance(v, int | float) for v in point):
                        logger.warning("Curve point has non-numeric values: %s", point)
                    else:
                        path_points.append((point[0] * scale_x, point[1] * scale_y))
                else:
                    logger.warning(
                        "Curve command unexpected point format: %s",
                        item,
                    )
            else:
                logger.warning("Curve command with insufficient items: %s", item)
        elif cmd == "re" and len(item) >= 2:  # rectangle
            point = item[1]
            # For rectangles, we get corner coordinates as (x0, y0, x1, y1)
            # Note: Could be list or tuple after JSON deserialization
            if isinstance(point, tuple | list) and len(point) == 4:
                if not all(isinstance(v, int | float) for v in point):
                    logger.warning(
                        "Rectangle coords have non-numeric values: %s", point
                    )
                else:
                    # Draw all four corners of the rectangle
                    x0, y0, x1, y1 = point
                    path_points.extend(
                        [
                            (x0 * scale_x, y0 * scale_y),
                            (x1 * scale_x, y0 * scale_y),
                            (x1 * scale_x, y1 * scale_y),
                            (x0 * scale_x, y1 * scale_y),
                            (x0 * scale_x, y0 * scale_y),  # Close the rectangle
                        ]
                    )
            else:
                logger.warning(
                    "Rectangle command unexpected point format: %s",
                    item,
                )
        else:
            logger.warning(
                "Encountered unknown drawing commands: %s (add support if needed)",
                sorted(cmd),
            )

    # Draw the path as connected line segments
    if len(path_points) >= 2:
        draw.line(path_points, fill=color, width=1)
    elif len(path_points) == 1:
        # Single point - draw as a small dot
        x, y = path_points[0]
        draw.ellipse([(x - 1, y - 1), (x + 1, y + 1)], fill=color)
    elif len(path_points) == 0 and commands_processed > 0:
        # Commands were processed but no points generated - log for debugging
        logger.debug(
            "Processed %d drawing commands but generated 0 path points",
            commands_processed,
        )
