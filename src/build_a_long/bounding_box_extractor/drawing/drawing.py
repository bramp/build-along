import logging
from pathlib import Path
from typing import Tuple

import pymupdf
from PIL import Image, ImageDraw

from build_a_long.bounding_box_extractor.extractor.page_elements import (
    Drawing,
    Element,
    Image as ImageElement,
    Text,
)

logger = logging.getLogger(__name__)


def draw_and_save_bboxes(
    page: pymupdf.Page,
    hierarchy: Tuple[Element, ...],
    output_path: Path,
) -> None:
    """
    Draws bounding boxes from a hierarchy on the PDF page image and saves it.
    Colors are based on nesting depth, and element types are labeled.
    """
    image_dpi = 150

    # Render page to an image
    pix = page.get_pixmap(colorspace=pymupdf.csRGB, dpi=image_dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    draw = ImageDraw.Draw(img)

    # Get page dimensions for scaling
    page_rect = page.rect
    scale_x = pix.width / page_rect.width
    scale_y = pix.height / page_rect.height

    # Colors for different nesting depths (cycles through this list)
    depth_colors = ["red", "green", "blue", "yellow", "purple", "orange"]

    def _draw_element(element: Element, depth: int) -> None:
        """Recursively draw an element and its children."""
        bbox = element.bbox

        # Scale the bounding box
        scaled_bbox = (
            bbox.x0 * scale_x,
            bbox.y0 * scale_y,
            bbox.x1 * scale_x,
            bbox.y1 * scale_y,
        )

        # Determine color based on depth
        color = depth_colors[depth % len(depth_colors)]

        # Draw the bounding box
        draw.rectangle(scaled_bbox, outline=color, width=2)

        # Draw the element type text
        label = element.__class__.__name__
        if isinstance(element, Drawing):
            if element.image_id:
                label = f"{label} ({element.image_id})"
        elif isinstance(element, ImageElement):
            if element.image_id:
                label = f"{label} ({element.image_id})"
        elif isinstance(element, Text):
            # For Text elements, show the actual text content
            content = element.content.strip()
            if len(content) > 50:  # Truncate long text
                content = content[:47] + "..."
            label = f"{label}: {content}"

        # Below bottom-left
        text_position = (scaled_bbox[0], scaled_bbox[3] + 2)
        draw.text(text_position, label, fill=color)

        # Recursively draw children
        for child in element.children:
            _draw_element(child, depth + 1)

    # Start traversal from root elements
    for root_element in hierarchy:
        _draw_element(root_element, 0)

    img.save(output_path)
    logger.info("Saved image with bboxes to %s", output_path)
