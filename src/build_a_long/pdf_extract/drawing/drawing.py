import logging
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

from build_a_long.pdf_extract.classifier.types import ClassificationResult
from build_a_long.pdf_extract.extractor.page_elements import (
    Drawing,
    PageElement,
    Image as ImageElement,
    Text,
)
from build_a_long.pdf_extract.extractor.hierarchy import ElementTree

logger = logging.getLogger(__name__)


def draw_and_save_bboxes(
    page: pymupdf.Page,
    hierarchy: ElementTree,
    result: ClassificationResult,
    output_path: Path,
    *,
    draw_deleted: bool = False,
) -> None:
    """
    Draws bounding boxes from a hierarchy on the PDF page image and saves it.
    Colors are based on nesting depth, and element types are labeled.

    Args:
        page: PyMuPDF page to render
        hierarchy: ElementTree containing the element hierarchy
        result: ClassificationResult containing labels for elements
        output_path: Where to save the output image
        draw_deleted: If True, also render elements marked as deleted.
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

    def _draw_element(element: PageElement, depth: int) -> None:
        """Recursively draw an element and its children."""
        if element.deleted and not draw_deleted:
            return

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

        # If element is deleted, use a lighter/grayed out color and dashed style
        if element.deleted:
            # Draw dashed outline for deleted elements
            # TODO Fix this, as we don't currently draw dashed, OR with lower opacity
            # PIL doesn't support dashed lines directly, so we'll draw with lower opacity
            draw.rectangle(scaled_bbox, outline=color, width=1)
        else:
            # Draw the bounding box normally
            draw.rectangle(scaled_bbox, outline=color, width=2)

        # Draw the element type text
        label_prefix = "[REMOVED] " if element.deleted else ""
        element_label = result.get_label(element)  # type: ignore[arg-type]
        label = f"ID: {element.id} {label_prefix}" + (
            element_label or element.__class__.__name__
        )
        if isinstance(element, Drawing):
            if element.image_id:
                label = f"{label} ({element.image_id})"
        elif isinstance(element, ImageElement):
            if element.image_id:
                label = f"{label} ({element.image_id})"
        elif isinstance(element, Text):
            # For Text elements, show the actual text content
            content = element.text.strip()
            if len(content) > 50:  # Truncate long text
                content = content[:47] + "..."
            label = f"{label}: {content}"

        # Below bottom-left
        if depth % 2 == 0:  # Even depth, left align
            text_position = (scaled_bbox[0], scaled_bbox[3] + 2)
            draw.text(text_position, label, fill=color)
        else:  # Odd depth, right align
            # Calculate text width for right alignment
            text_bbox = draw.textbbox((0, 0), label)
            text_width = text_bbox[2] - text_bbox[0]
            text_position = (scaled_bbox[2] - text_width, scaled_bbox[3] + 2)
            draw.text(text_position, label, fill=color)

        # Recursively draw children using the hierarchy
        children = hierarchy.get_children(element)
        for child in children:
            _draw_element(child, depth + 1)

    # Start traversal from root elements
    for root_element in hierarchy.roots:
        _draw_element(root_element, 0)

    img.save(output_path)
    logger.info("Saved image with bboxes to %s", output_path)
